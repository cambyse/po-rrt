use itertools::{all, izip, zip};

use crate::common::*;
use crate::nearest_neighbor::*;
use crate::sample_space::*;
use crate::map_shelves_io::*;
use std::{any, cmp::min, collections::BTreeMap, vec::Vec};
use core::cell::RefCell;
use std::rc::{Weak, Rc};
use bitvec::prelude::*;

pub struct RRTNode<const N: usize> {
	pub state: [f64; N],
	pub parent_id: Option<usize>,
	pub dist_from_root: f64,
}

pub struct RRTTree<const N: usize> {
	pub nodes: Vec<RRTNode<N>>,
}

impl<const N: usize> RRTTree<N> {
	fn new(state: [f64; N]) -> Self {
		let root = RRTNode { state, parent_id: None, dist_from_root: 0.0 };
		Self { nodes: vec![root] }
	}

	fn add_node(&mut self, state: [f64; N], parent_id: usize, dist_from_parent: f64) -> usize {
		let id = self.nodes.len();

		let parent = &self.nodes[parent_id];

		let node = RRTNode { state, parent_id: Some(parent_id), dist_from_root: parent.dist_from_root + dist_from_parent};
		self.nodes.push(node);
		id
	}

	fn reparent_node(&mut self, node_id: usize, new_parent_id: usize, dist_from_new_parent: f64) {
		// we do things in two steps to avoid making the borrow checker unhappy
		let parent_dist_from_root = self.nodes[new_parent_id].dist_from_root;
		let node = &mut self.nodes[node_id];
		node.parent_id = Some(new_parent_id);
		node.dist_from_root = parent_dist_from_root + dist_from_new_parent;
	}

	/*
	fn distances_from_common_ancestor(&self, leaf_ids: &Vec<usize>) -> Vec<f64> {
		if leaf_ids.is_empty() {
			return vec![];
		}

		if leaf_ids.len() == 1 {
			return vec![0.0];
		}

		// We get a list of leaves. For each leaf, we go up the ancestor chain,
		// until we hit the root, and compute the distance from the root.
		// It would be more efficient to stop at the common ancestor, but we don't
		// know which one it is.

		fn compute_distance_from_root<const N: usize>(
				tree: &RRTTree<N>,
				distances_from_root: &mut BTreeMap<usize, f64>,
				node_id: usize) -> f64 {
			if node_id == 0 {
				return 0.0;
			}

			if let Some(d) = distances_from_root.get(&node_id) {
				return *d;
			}

			let node = &tree.nodes[node_id];
			let parent_id = node.parent_id.unwrap();

			let d = compute_distance_from_root(tree, distances_from_root, parent_id) + node.dist_from_parent;
			distances_from_root.insert(node_id, d);

			return d;
		}

		let mut distances_from_root = BTreeMap::new();
		leaf_ids.iter()
			.map(|id| compute_distance_from_root(&self, &mut distances_from_root, *id))
			.collect::<Vec<_>>()
	}*/

	fn get_path_to(&self, id: usize) -> Vec<[f64; N]> {
		let mut path = Vec::new();

		let mut node = &self.nodes[id];
		path.push(node.state);

		while let Some(id) = node.parent_id {
			node = &self.nodes[id];
			path.push(node.state);
		}

		path.reverse();
		path
	}
}

pub trait RTTFuncs<const N: usize> {
	fn state_validator(&self, _state: &[f64; N]) -> bool {
		true
	}

	fn transition_validator(&self, _from: &[f64; N], _to: &[f64; N]) -> bool {
		true
	}

	fn cost_evaluator(&self, a: &[f64; N], b: &[f64; N]) -> f64 {
		norm2(a,b)
	}
}

pub struct RRT<F: RTTFuncs<N>, const N: usize> {
	sample_space: ContinuousSampler<N>,
	fns: F,
}

impl<F: RTTFuncs<N>, const N: usize> RRT<F, N> {
	pub fn new(sample_space: ContinuousSampler<N>, fns: F) -> Self {
		Self { sample_space, fns }
	}

	pub fn plan(&mut self, start: [f64; N], goal: &impl GoalFuncs<N>,
				 max_step: f64, search_radius: f64, n_iter_max: usize) -> (Result<Vec<[f64; N]>, &str>, RRTTree<N>) {
		let (rrttree, final_node_ids) = self.grow_tree(start, goal, max_step, search_radius, n_iter_max);

		(self.get_best_solution(&rrttree, &final_node_ids), rrttree)
	}

	fn grow_tree(&mut self, start: [f64; N], goal: &impl GoalFuncs<N>,
				max_step: f64, search_radius: f64, n_iter_max: usize) -> (RRTTree<N>, Vec<usize>) {
		let mut final_node_ids = Vec::<usize>::new();
		let mut rrttree = RRTTree::new(start);
		let mut kdtree = KdTree::new(start);

		for it in 0..n_iter_max {
			let mut new_state = self.sample(goal, it+1);
			let kd_from = kdtree.nearest_neighbor(new_state);

			steer(&kd_from.state, &mut new_state, max_step);

			if self.fns.state_validator(&new_state) {
				// RRT* algorithm
				// Step 1: Find the best parent we can get
				// First, we find the neighbors in a specific radius of new_state.
				let radius = {
					let n = rrttree.nodes.len() as f64;
					let s = search_radius * (n.ln()/n).powf(1.0/(N as f64));
					if s < max_step { s } else { max_step }
				};

				let mut neighbour_ids: Vec<usize> = kdtree.nearest_neighbors(new_state, radius).iter()
					.filter(|node| self.fns.transition_validator(&node.state, &new_state))
					.map(|node| node.id)
					.collect();

				if neighbour_ids.is_empty() {
					neighbour_ids.push(kd_from.id);
				}

				// Evaluate which is the best parent that we can possibly get
				let (distances_from_root, distances_from_parents): (Vec<f64>, Vec<f64>) = neighbour_ids.iter() // vec of pair (dist from root, dist from parent)
					.map(|id| (id, &rrttree.nodes[*id]))
					.map(|(_, parent)| (parent.dist_from_root, self.fns.cost_evaluator(&parent.state, &new_state)))
					.unzip();
			
				//let dist_from_kd = self.fns.cost_evaluator(&rrttree.nodes[kd_from.id].state, &new_state);
				let (best_parent_id, _best_parent_distance_from_root, distance_from_best_parent) = izip!(&neighbour_ids, &distances_from_root, &distances_from_parents)
					.min_by(|(_, d0a, d1a), (_, d0b, d1b)| (*d0a + *d1a).partial_cmp(&(*d0b + *d1b)).unwrap())
					.expect("should have at least the nearest neighbor!");

				// Add the node in the trees
				let new_node_id = rrttree.add_node(new_state, *best_parent_id, *distance_from_best_parent);
				let new_node_distance_from_root = rrttree.nodes[new_node_id].dist_from_root;

				// Step 2: Perhaps we can reparent some of the neighbours to the new node
				for neighbour_id in &neighbour_ids {
					if *neighbour_id == *best_parent_id { continue; }

					let neighbour = &rrttree.nodes[*neighbour_id];
					let distance_from_new_node = self.fns.cost_evaluator(&new_state, &neighbour.state);
					let distance_from_root_via_new_node = new_node_distance_from_root + distance_from_new_node;
					if distance_from_root_via_new_node < neighbour.dist_from_root {
						rrttree.reparent_node(*neighbour_id, new_node_id, distance_from_new_node);
					}
				}

				kdtree.add(new_state, new_node_id);

				if goal.goal(&new_state).is_some() {
					final_node_ids.push(new_node_id);
				}
			}
		}

		//println!("number of final nodes: {}", final_nodes.len());

		(rrttree, final_node_ids)
	}

	fn sample(&mut self, goal: &impl GoalFuncs<N>, iteration: usize) -> [f64; N] {
		let new_state = match iteration % 100 {
			0 => goal.goal_example(0),
			_ => self.sample_space.sample()
		};

		new_state
	}

	fn get_best_solution(&self, rrttree: &RRTTree<N>, final_node_ids: &[usize]) -> Result<Vec<[f64; N]>, &str> {
		final_node_ids.iter()
			.map(|id| {
				let path = rrttree.get_path_to(*id);
				let cost = self.get_path_cost(&path);
				(path, cost)
			})
			.min_by(|(_,a),(_,b)| a.partial_cmp(b).expect("NaN found"))
			.map(|(p, _)| p)
			.ok_or("No solution found")
	}

	fn get_path_cost(&self, path: &[[f64; N]]) -> f64 {
		pairwise_iter(path)
			.map(|(a,b)| self.fns.cost_evaluator(a,b))
			.sum()
	}
}

#[cfg(test)]
mod tests {

use super::*;

#[test]
fn test_plan_empty_space() {
	struct Funcs {}
	impl RTTFuncs<2> for Funcs {}

	let goal = SquareGoal::new(vec![([0.9, 0.9], bitvec![1])], 0.05);

	let mut rrt = RRT::new(ContinuousSampler::new([-1.0, -1.0], [1.0, 1.0]), Funcs{});

	let (path_result, _) = rrt.plan([0.0, 0.0], &goal, 0.1, 1.0, 1000);

	assert!(path_result.clone().expect("No path found!").len() > 2); // why do we need to clone?!
}

#[test]
fn test_plan_on_map7_prefefined_goal() {
	let mut m = MapShelfDomain::open("data/map7.pgm", [-1.0, -1.0], [1.0, 1.0]);
	m.add_zones("data/map7_6_goals_zone_ids.pgm", 0.5);

	struct Funcs<'a>  {
		m: &'a MapShelfDomain,
	}

	impl<'a> RTTFuncs<2> for Funcs<'a> {
		fn state_validator(&self, state: &[f64; 2]) -> bool {
			self.m.is_state_valid(state) == Belief::Free
		}

		fn transition_validator(&self, from: &[f64; 2], to: &[f64; 2]) -> bool {
			self.m.get_traversed_space(from, to) == Belief::Free
		}
	}

	let goal = SquareGoal::new(vec![([0.9, -0.5], bitvec![1])], 0.05);

	let mut rrt = RRT::new(ContinuousSampler::new([-1.0, -1.0], [1.0, 1.0]), Funcs{m:&m});

	let (path_result, rrttree) = rrt.plan([0.0, -0.8], &goal, 0.1, 5.0, 5000);

	assert!(path_result.clone().expect("No path found!").len() > 2); // why do we need to clone?!

	let mut m2 = m.clone();
	m2.resize(5);
	m2.draw_tree(&rrttree);
	m2.draw_zones_observability();
	m2.draw_path(path_result.unwrap().as_slice(), colors::BLACK);
	m2.save("results/map7/test_map7_rrt_to_goal");
}

#[test]
fn test_plan_on_map7_observation_point() {
	let mut m = MapShelfDomain::open("data/map7.pgm", [-1.0, -1.0], [1.0, 1.0]);
	m.add_zones("data/map7_6_goals_zone_ids.pgm", 0.5);

	struct Funcs<'a>  {
		m: &'a MapShelfDomain,
	}

	impl<'a> RTTFuncs<2> for Funcs<'a> {
		fn state_validator(&self, state: &[f64; 2]) -> bool {
			self.m.is_state_valid(state) == Belief::Free
		}

		fn transition_validator(&self, from: &[f64; 2], to: &[f64; 2]) -> bool {
			self.m.get_traversed_space(from, to) == Belief::Free
		}
	}

	pub struct ObservationGoal<'a> {
		m: &'a MapShelfDomain,
		zone_id: usize
	}

	impl<'a> GoalFuncs<2> for ObservationGoal<'a> {
		fn goal(&self, state: &[f64; 2]) -> Option<WorldMask> {	
			if self.m.is_zone_observable(state, self.zone_id) {
				return Some(bitvec![1]);
			}
			None
		}
	
		fn goal_example(&self, _:usize) -> [f64; 2] {
			self.m.get_zone_positions()[self.zone_id]
		}
	}

	let goal = ObservationGoal{m: &m, zone_id: 2};

	let mut rrt = RRT::new(ContinuousSampler::new([-1.0, -1.0], [1.0, 1.0]), Funcs{m:&m});

	let (path_result, rrttree) = rrt.plan([0.0, -0.8], &goal, 0.1, 5.0, 5000);

	assert!(path_result.clone().expect("No path found!").len() > 2); // why do we need to clone?!

	let mut m2 = m.clone();
	m2.resize(5);
	m2.draw_tree(&rrttree);
	m2.draw_zones_observability();
	m2.draw_path(path_result.unwrap().as_slice(), colors::BLACK);
	m2.save("results/map7/test_map7_rrt_to_observation_point");
}
}