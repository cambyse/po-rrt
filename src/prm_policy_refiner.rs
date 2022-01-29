use itertools::{all, enumerate, izip, merge, zip};

use crate::common::*;
use crate::nearest_neighbor::*;
use crate::map_io::*; // tests only
use crate::map_shelves_io::*; // tests only
use crate::prm_graph::*;
use std::collections::HashSet;
use crate::prm_reachability::*;
use crate::belief_graph::*;
use crate::sample_space::*;
use crate::prm::*;
use queues::*;
use bitvec::prelude::*;
use priority_queue::PriorityQueue;

pub enum RefinmentStrategy {
    Reparent(f64),
    PartialShortCut(usize),
}

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
	pub compatibilities: Vec<Vec<bool>>,
}

impl <'a, F: PRMFuncs<N>, const N: usize> PRMPolicyRefiner<'a, F, N> {	
	pub fn new(policy: &'a Policy<N>, fns: &'a F, belief_graph: &'a BeliefGraph<N>) -> Self {
		Self{
			policy,
			fns,
			belief_graph,
			compatibilities: compute_compatibility(&belief_graph.reachable_belief_states, &fns.world_validities()),
		}
	}

	pub fn refine_solution(&mut self, strategy: RefinmentStrategy) -> (Policy<N>, Vec<RefinmentTree<N>>) {
		// option 1: extract traj pieces, refine each piece independently
		// -> each piece can be converted to tree + reparenting + resampling?
		// option 2: extract tube in belief graph, 
		// -> add samples in belief space directly + re-extract cond dijkstra

		// option 1
		println!("refine policy..");

		let (path_pieces, skeleton) = self.policy.decompose();
		let mut trees = vec![];

		for (_belief_state, path) in &path_pieces {
			let tree = match strategy {
				RefinmentStrategy::Reparent(radius) => {
					let (mut tree, kdtree) = self.build_tree(path, radius);
					self.reparent(&mut tree, &kdtree, 0.5 * radius);
					tree
				},
				RefinmentStrategy::PartialShortCut(n_iterations) => {
					let mut path_piece = self.build_path_piece(path);
					self.partial_shortcut(&mut path_piece, n_iterations);
					path_piece
				}
			};


			trees.push(tree);
		}

		let policy = self.recompose(&trees, &skeleton);

		println!("success! (number of nodes:{}, number of leafs:{})", policy.nodes.len(), policy.leafs.len());

		(policy, trees)
	}

	// Shortcut strategy
	fn build_path_piece(&self, path: &[usize]) -> RefinmentTree<N> {
		// add first node
		let root_belief_graph_id = self.policy.nodes[*path.first().expect("path shouldn't be empty!")].original_node_id;
		let root_belief_node = &self.belief_graph.nodes[root_belief_graph_id];

		let mut tree = RefinmentTree {
			nodes: vec![],
			belief_state_id: 0,
			leaf: 0
		};
		tree.add_node(root_belief_node.state, None, root_belief_graph_id);

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
			tree.add_node(next_belief_node.state, Some(edge), next_belief_graph_id);
		}
		tree.belief_state_id = root_belief_node.belief_id;
		tree.leaf = tree.nodes.len() - 1;
		tree
	}

	fn partial_shortcut(&self, tree: &mut RefinmentTree<N>, n_iterations: usize) {
		fn interpolate(a: f64, b: f64, lambda: f64) -> f64 {
			a * (1.0 - lambda) + b * lambda
		}

		if tree.nodes.len() <= 2 {
			return; // can't shortcut with only 2 states or less
		}

		// Implement partial shortcut, see "Creating_High-quality_Paths_for_Motion_Planning"	
		let joint_dim = tree.nodes.first().unwrap().state.len();
		let mut sampler = DiscreteSampler::new();

		for _i in 0..n_iterations {
			let joint = sampler.sample(joint_dim);
			let interval_start = sampler.sample(tree.nodes.len() - 2);
			let interval_end = interval_start + 2 + sampler.sample(tree.nodes.len() - interval_start - 2);

			assert!(interval_end < tree.nodes.len());
			assert!(interval_end - interval_start >= 2);

			let interval_start_state = &tree.nodes[interval_start].state;
			let interval_end_state = &tree.nodes[interval_end].state;

			// create shortcut states (interpolated on a particular joint)
			let mut shortcut_states = vec![];
			shortcut_states.reserve(interval_end - interval_start);
			for j in interval_start..interval_end {
				let lambda = (j - interval_start) as f64 / (interval_end - interval_start) as f64;
				let mut shortcut_state = tree.nodes[j].state;
				shortcut_state[joint] = interpolate(interval_start_state[joint], interval_end_state[joint], lambda);
				shortcut_states.push(shortcut_state);
			}

			// check validities
			let mut should_commit = true;
			for (from, to) in pairwise_iter(&shortcut_states) {
				should_commit = should_commit && self.is_transition_valid(from, to, tree.belief_state_id); // TODO: can be optimized to avoid rechecking 2 times the nodes
			}

			// commit if valid
			if should_commit {
				for j in interval_start..interval_end {
					tree.nodes[j].state = shortcut_states[j - interval_start];
				}
			}
		}
	}

	// Reparent strategy
	fn build_tree(&self, path: &[usize], radius: f64) -> (RefinmentTree<N>, KdTree<N>) {
		let mut visited = HashSet::new();

		// add first node
		let root_belief_graph_id = self.policy.nodes[*path.first().expect("path shouldn't be empty!")].original_node_id;
		let root_belief_node = &self.belief_graph.nodes[root_belief_graph_id];

		let mut tree = RefinmentTree {
			nodes: vec![],
			belief_state_id: 0,
			leaf: 0
		};
		tree.add_node(root_belief_node.state, None, root_belief_graph_id);
		let mut kdtree =  KdTree::new(root_belief_node.state);
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
			let id = tree.add_node(next_belief_node.state, Some(edge), next_belief_graph_id);
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
			let node_state = tree.nodes[node_id].state;

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

	fn recompose(&self, trees: &[RefinmentTree<N>], skeleton: &[Vec<usize>]) -> Policy<N> {
		let mut policy = Policy {
			nodes: vec![],
			leafs: vec![]
		};

		let mut pieces_start_end: Vec<(Option<usize>, Option<usize>)> = vec![(None, None); skeleton.len()];

		for (i, tree) in trees.iter().enumerate() {
			let mut node = tree.nodes[tree.leaf];
			let belief_node = &self.belief_graph.nodes[node.belief_graph_id];
			let id = policy.add_node(&node.state, &belief_node.belief_state, node.belief_graph_id, false);

			pieces_start_end[i].1 = Some(id); // end of piece inserted first

			while let Some(parent) = node.parent {
				node = tree.nodes[parent.id];
				let belief_node = &self.belief_graph.nodes[node.belief_graph_id];
				let id = policy.add_node(&node.state, &belief_node.belief_state, node.belief_graph_id, false);
				policy.add_edge(id, id - 1);

				pieces_start_end[i].0 = Some(id); // start of piece re-update along the way
			}
		}

		// reconnect to pieces branchings
		for (i, next_pieces) in skeleton.iter().enumerate() {
			let from_end = pieces_start_end[i].1;

			for next_piece in next_pieces {
				let to_start = pieces_start_end[*next_piece].0;

				if let (Some(from_end), Some(to_start)) = (from_end, to_start) {
					policy.add_edge(from_end, to_start);
				}
			}
		}

		// set remaining leafs
		let nodes = policy.nodes.clone();
		for (i, node) in nodes.iter().enumerate() {
			if node.children.is_empty() {
				policy.leafs.push(i);
			}
		}

		policy
	}

	fn is_transition_valid(&self, from: &[f64; N], to: &[f64; N], belief_state_id: usize) -> bool {
		let from_validity = self.fns.state_validity(from);
		let to_validity = self.fns.state_validity(to);
		
		if let (Some(from_validity_id), Some(to_validity_id)) = (from_validity, to_validity) {
			let from_node = PRMNode{
				state: *from,
				validity_id: from_validity_id,
				parents: vec![],
				children: vec![]
			};

			let to_node = PRMNode{
				state: *to,
				validity_id: to_validity_id,
				parents: vec![],
				children: vec![]
			};

			let validity = self.fns.transition_validator(&from_node, &to_node);

			match validity {
				Some(validity_id) => {return self.compatibilities[belief_state_id][validity_id]}, // TODO: should check consistenty with from_validity and to_validity as well?
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
	fn test_plan_on_map2_pomdp_partial_shortcut() {
		let mut m = Map::open("data/map2.pgm", [-1.0, -1.0], [1.0, 1.0]);
		m.add_zones("data/map2_zone_ids.pgm", 0.2);

		let goal = SquareGoal::new(vec![([0.55, 0.9], bitvec![1; 4])], 0.05);

		let mut prm = PRM::new(ContinuousSampler::new([-1.0, -1.0], [1.0, 1.0]),
							DiscreteSampler::new(),
							&m);

		prm.grow_graph(&[0.55, -0.8], &goal, 0.1, 5.0, 2000, 100000).expect("graph not grown up to solution");
		prm.print_summary();
		let policy = prm.plan_belief_space(&vec![0.1, 0.1, 0.1, 0.7]);

		let mut policy_refiner = PRMPolicyRefiner::new(&policy, &m, &prm.belief_graph);
		let (refined_policy, _) = policy_refiner.refine_solution(RefinmentStrategy::PartialShortCut(500));
		
		assert_eq!(policy.leafs.len(), refined_policy.leafs.len());

		let mut m2 = m.clone();
		m2.resize(5);
		m2.draw_full_graph(&prm.graph);
		m2.draw_zones_observability();
		//m2.draw_policy(&policy);
		//m2.draw_refinment_trees(&trees);
		m2.draw_policy(&refined_policy);
		m2.save("results/test_prm_on_map2_pomdp_refined_partial_shortcut");
	}

	#[test]
	fn test_plan_on_map4_pomdp() {
		let mut m = Map::open("data/map4.pgm", [-1.0, -1.0], [1.0, 1.0]);
		m.add_zones("data/map4_zone_ids.pgm", 0.15);
	
		let goal = SquareGoal::new(vec![([-0.55, 0.9], bitvec![1; 16])], 0.05);
		let mut prm = PRM::new(ContinuousSampler::new([-1.0, -1.0], [1.0, 1.0]),
							   DiscreteSampler::new(),
							   &m);
	
		prm.grow_graph(&[0.55, -0.8], &goal, 0.05, 5.0, 10500, 100000).expect("graph not grown up to solution");
		prm.print_summary();
		let policy = prm.plan_belief_space( &vec![1.0/16.0; 16]);
	
		let mut policy_refiner = PRMPolicyRefiner::new(&policy, &m, &prm.belief_graph);
		let (refined_policy, _) = policy_refiner.refine_solution(RefinmentStrategy::PartialShortCut(1500));
		
		assert_eq!(policy.leafs.len(), refined_policy.leafs.len());

		let mut m2 = m.clone();
		m2.resize(5);
		m2.draw_full_graph(&prm.graph);
		m2.draw_zones_observability();
		//m2.draw_policy(&policy);
		//m2.draw_refinment_trees(&trees);
		m2.draw_policy(&refined_policy);
		m2.save("results/test_prm_on_map4_pomdp_refined_partial_shortcut");
	}

	#[test]
	fn test_plan_on_map2_pomdp() {
		let mut m = Map::open("data/map2.pgm", [-1.0, -1.0], [1.0, 1.0]);
		m.add_zones("data/map2_zone_ids.pgm", 0.2);

		let goal = SquareGoal::new(vec![([0.55, 0.9], bitvec![1; 4])], 0.05);

		let mut prm = PRM::new(ContinuousSampler::new([-1.0, -1.0], [1.0, 1.0]),
							DiscreteSampler::new(),
							&m);

		prm.grow_graph(&[0.55, -0.8], &goal, 0.1, 5.0, 2000, 100000).expect("graph not grown up to solution");
		prm.print_summary();
		let policy = prm.plan_belief_space(&vec![0.1, 0.1, 0.1, 0.7]);

		let mut policy_refiner = PRMPolicyRefiner::new(&policy, &m, &prm.belief_graph);
		let (refined_policy, trees) = policy_refiner.refine_solution(RefinmentStrategy::Reparent(0.3));
		
		assert_eq!(policy.leafs.len(), refined_policy.leafs.len());

		let mut m2 = m.clone();
		m2.resize(5);
		m2.draw_full_graph(&prm.graph);
		m2.draw_zones_observability();
		m2.draw_refinment_trees(&trees);
		m2.draw_policy(&refined_policy);
		m2.save("results/test_prm_on_map2_pomdp_refined");
	}

	#[test]
	fn test_plan_on_map1_3_goals() {
		let mut m = MapShelfDomain::open("data/map1_3_goals.pgm", [-1.0, -1.0], [1.0, 1.0]);
		m.add_zones("data/map1_3_goals_zone_ids.pgm", 0.5);

		let goal = SquareGoal::new(vec![([0.65, 0.14], bitvec![1, 0, 0]),
										([0.0, 0.75], bitvec![0, 1, 0]),
										([-0.7, 0.14], bitvec![0, 0, 1])],
										0.05);

		let mut prm = PRM::new(ContinuousSampler::new([-1.0, -1.0], [1.0, 1.0]),
							DiscreteSampler::new(),
							&m);

		prm.grow_graph(&[0.0, -0.8], &goal, 0.05, 5.0, 5000, 100000).expect("graph not grown up to solution");
		prm.print_summary();

		let policy = prm.plan_belief_space(&vec![1.0/3.0, 1.0/3.0, 1.0/3.0]);

		let mut policy_refiner = PRMPolicyRefiner::new(&policy, &m, &prm.belief_graph);
		let (refined_policy, trees) = policy_refiner.refine_solution(RefinmentStrategy::Reparent(0.3));

		assert_eq!(policy.leafs.len(), refined_policy.leafs.len());

		let mut m2 = m.clone();
		m2.resize(5);
		m2.draw_full_graph(&prm.graph);
		m2.draw_zones_observability();
		m2.draw_refinment_trees(&trees);
		m2.draw_policy(&refined_policy);
		m2.save("results/test_plan_on_map1_3_goals_pomdp_refined");
	}
}