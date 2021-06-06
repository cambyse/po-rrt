use itertools::{all, enumerate, izip, merge, zip};

use crate::common::*;
use crate::nearest_neighbor::*;
use crate::map_io::*; // tests only
use crate::prm_graph::*;
use std::collections::HashSet;
use crate::prm_reachability::*;
use crate::belief_graph::*;
use crate::sample_space::*;
use crate::prm::*;
use queues::*;
use bitvec::prelude::*;
use priority_queue::PriorityQueue;


#[derive(Clone, Copy)]
struct ParentLink {
	pub id: usize,
	pub dist: f64,
}

#[derive(Clone, Copy)]
pub struct Edge {
	pub id: usize,
	pub cost: f64,
}

#[derive(Clone, Copy)]
pub struct RefinmentNode<const N: usize> {
	pub state: [f64; N],
	pub parent: Option<Edge>,
	pub belief_graph_id: usize
}

pub struct RefinmentTree<const N: usize> {
	pub nodes: Vec<RefinmentNode<N>>,
	pub belief_state_id: usize,
	pub leaf: usize
}

impl<'a, const N: usize> RefinmentTree<N> { 
	fn add_node(&mut self, state: [f64; N], parent: Option<Edge>, belief_graph_id: usize) -> usize {
		let node = RefinmentNode { state, parent, belief_graph_id };
		self.nodes.push(node);
		self.nodes.len() - 1
	}

	fn dist_from_root(&self, id: usize) -> f64 {
		let mut node_id = id;
		let mut cost = 0.0;
		while let Some(parent_link) = &self.nodes[node_id].parent {
			cost += parent_link.cost;
			node_id = parent_link.id;
		}

		cost
	}
}

pub struct PRMPolicyRefiner <'a, F: PRMFuncs<N>, const N: usize> {
	pub policy: &'a Policy<N>,
	pub fns: &'a F,
	pub belief_graph: &'a BeliefGraph<N>,
	pub compatibilities: Vec<Vec<bool>> 
}

impl <'a, F: PRMFuncs<N>, const N: usize> PRMPolicyRefiner<'a, F, N> {	
	pub fn new(policy: &'a Policy<N>, fns: &'a F, belief_graph: &'a BeliefGraph<N>) -> Self {
		Self{
			policy,
			fns,
			belief_graph,
			compatibilities: compute_compatibility(&belief_graph.reachable_belief_states, &fns.world_validities())
		}
	}

	pub fn refine_solution(&mut self, radius: f64) -> (Policy<N>, Vec<RefinmentTree<N>>) {
		// option 1: extract traj pieces, refine each piece independently
		// -> each piece can be converted to tree + reparenting + resamplning?
		// option 2: extract tube in belief graph, 
		// -> add samples in belief space directly + re-extract cond dijkstra

		// option 1
		println!("refine policy..");

		let (path_pieces, skeleton) = self.policy.decompose();
		let mut trees = vec![];
		let mut kdtrees = vec![];

		for (_belief_state, path) in &path_pieces {
			let (mut tree, kdtree) = self.build_tree(path, radius);

			self.reparent(&mut tree, &kdtree, 0.5 * radius);

			trees.push(tree);
			kdtrees.push(kdtree);
		}

		let policy = self.recompose(&trees, &skeleton);
		println!("success!");

		(policy, trees)
	}

	fn build_tree(&self, path: &[usize], radius: f64) -> (RefinmentTree<N>, KdTree<N>) {
		let mut visited = HashSet::new();

		// add first nodes
		let root_belief_graph_id = self.policy.nodes[*path.first().expect("path shouldn't be empty!")].original_node_id;
		let root_belief_node = &self.belief_graph.nodes[root_belief_graph_id];

		let mut tree = RefinmentTree {
			nodes: vec![],
			belief_state_id: 0,
			leaf: 0
		};
		tree.add_node(root_belief_node.state.clone(), None, root_belief_graph_id);
		let mut kdtree =  KdTree::new(root_belief_node.state.clone());
		visited.insert(root_belief_graph_id);

		// initialize tree with path
		for (previous_policy_id, next_policy_id) in pairwise_iter(path) {
			let previous_belief_graph_id = self.policy.nodes[*previous_policy_id].original_node_id;
			let previous_belief_node = &self.belief_graph.nodes[previous_belief_graph_id];

			let next_belief_graph_id = self.policy.nodes[*next_policy_id].original_node_id;
			let next_belief_node = &self.belief_graph.nodes[next_belief_graph_id];

			let edge = Edge{
				id: tree.nodes.len() - 1,
				cost: self.fns.cost_evaluator(&previous_belief_node.state, &next_belief_node.state)
			};
			let id = tree.add_node(next_belief_node.state.clone(), Some(edge), next_belief_graph_id);
			kdtree.add(next_belief_node.state, id);
			visited.insert(next_belief_graph_id);
		}
		tree.belief_state_id = root_belief_node.belief_id;
		tree.leaf = tree.nodes.len() - 1;

		// add offsprings if in radius
		let nodes = tree.nodes.clone();
		for (node_id, node) in nodes.iter().enumerate() {
			let mut q: Queue<(usize, usize)> = queue![];
			q.add((node_id, node.belief_graph_id)).unwrap();

			while q.size() > 0 {
				let (tree_id, belief_graph_id) = q.remove().unwrap();
				let belief_node = &self.belief_graph.nodes[belief_graph_id];

				for &child_id in &belief_node.children {
					let child_belief_node = &self.belief_graph.nodes[child_id];

					if !visited.contains(&child_id) && norm2(&node.state, &child_belief_node.state) <= radius {

						let edge = Edge{
							id: tree_id,
							cost: self.fns.cost_evaluator(&node.state, &child_belief_node.state)
						};

						let new_tree_id = tree.add_node(child_belief_node.state, Some(edge), child_id);
						kdtree.add(child_belief_node.state, new_tree_id);
						visited.insert(child_id);

						for &child_child_id in &child_belief_node.children {
							if !visited.contains(&child_child_id) {
								q.add((new_tree_id, child_child_id)).unwrap();
							}
						}
					}
				}
			}
		}

		(tree, kdtree)
	}

	fn reparent(&self, tree: &mut RefinmentTree<N>, kdtree: &KdTree<N>, radius: f64) {
		let mut q = PriorityQueue::new();

		// push all nodes
		for (node_id, _) in tree.nodes.iter().enumerate() {
			q.push(node_id, Priority{prio: tree.dist_from_root(node_id)});
		}

		while !q.is_empty() {
			let (node_id, _) = q.pop().unwrap();
			let node_state = tree.nodes[node_id].state.clone();

			let kd_neighbors = kdtree.nearest_neighbors(node_state, radius);
			let neighbor_ids: Vec<usize> = kd_neighbors.iter()
				.filter(|&kdnode| self.is_transition_valid(&node_state, &kdnode.state, tree.belief_state_id))
				.map(|&kdnode| kdnode.id)
				.collect();

			let distance_from_root = tree.dist_from_root(node_id);
			let neighbor_distances_from_root: Vec<(usize, f64)> = neighbor_ids.iter()
				.map(|id|(*id, tree.dist_from_root(*id)))
				.collect();

			for (neighbor_id, neighbor_dist_from_root) in neighbor_distances_from_root {
				let neighbor = &tree.nodes[neighbor_id];
				let cost = self.fns.cost_evaluator(&node_state, &neighbor.state);

				if distance_from_root + cost < neighbor_dist_from_root {
					// node_id is better parent! -> reparent
					let edge = Edge{
						id: node_id,
						cost
					};

					tree.nodes[neighbor_id].parent = Some(edge);

					// update priority
					q.push(neighbor_id, Priority{prio: distance_from_root + cost});
				}
			}			
		}	
	}

	fn recompose(&self, trees: &Vec<RefinmentTree<N>>, _skeleton: &Vec<Vec<usize>>) -> Policy<N> {
		// TODO: correct branching
		let mut policy = Policy {
			nodes: vec![],
			leafs: vec![]
		};

		for tree in trees {
			let mut node = tree.nodes[tree.leaf];
			let belief_node = &self.belief_graph.nodes[node.belief_graph_id];
			policy.add_node(&node.state, &belief_node.belief_state, node.belief_graph_id, false);

			while let Some(parent) = node.parent {
				node = tree.nodes[parent.id];
				let belief_node = &self.belief_graph.nodes[node.belief_graph_id];
				let id = policy.add_node(&node.state, &belief_node.belief_state, node.belief_graph_id, false);
				policy.add_edge(id -1, id);
			}
		}

		policy
	}

	fn is_transition_valid(&self, from: &[f64; N], to: &[f64; N], belief_state_id: usize) -> bool {
		let from_validity = self.fns.state_validity(from);
		let to_validity = self.fns.state_validity(to);
		
		if let (Some(from_validity_id), Some(to_validity_id)) = (from_validity, to_validity) {
			let from_node = PRMNode{
				state: from.clone(),
				validity_id: from_validity_id,
				parents: vec![],
				children: vec![]
			};

			let to_node = PRMNode{
				state: to.clone(),
				validity_id: to_validity_id,
				parents: vec![],
				children: vec![]
			};

			let validity = self.fns.transition_validator(&from_node, &to_node);

			match validity {
				Some(validity_id) => {return self.compatibilities[belief_state_id][validity_id]},
				None => {return false;}
			}
		}

		false
	}
}

#[cfg(test)]
mod tests {
    use super::*;

	#[test]
	fn test_plan_on_map2_pomdp() {
		let mut m = Map::open("data/map2.pgm", [-1.0, -1.0], [1.0, 1.0]);
		m.add_zones("data/map2_zone_ids.pgm", 0.2);

		fn goal(state: &[f64; 2]) -> WorldMask {
			bitvec![if (state[0] - 0.55).abs() < 0.05 && (state[1] - 0.9).abs() < 0.05 { 1 } else { 0 }; 4]
		}

		let mut prm = PRM::new(ContinuousSampler::new([-1.0, -1.0], [1.0, 1.0]),
							DiscreteSampler::new(),
							&m);

		prm.grow_graph(&[0.55, -0.8], goal, 0.1, 5.0, 2000, 100000).expect("graph not grown up to solution");
		prm.print_summary();
		let policy = prm.plan_belief_space(&vec![0.1, 0.1, 0.1, 0.7]);

		let mut policy_refiner = PRMPolicyRefiner::new(&policy, &m, &prm.belief_graph);

		let (policy, trees) = policy_refiner.refine_solution(0.3);

		let mut m2 = m.clone();
		m2.resize(5);
		m2.draw_full_graph(&prm.graph);
		m2.draw_zones_observability();
		m2.draw_refinment_trees(&trees);
		m2.draw_policy(&policy);
		m2.save("results/test_prm_on_map2_pomdp_refined");
	}
}