use itertools::{all, izip, zip};

use crate::common::*;
use crate::nearest_neighbor::*;
use crate::sample_space::*;
use crate::map_io::*;
use std::{any, cmp::min, collections::BTreeMap, vec::Vec};
use core::cell::RefCell;
use std::rc::{Weak, Rc};

pub struct RRTNode<const N: usize> {
	pub state: [f64; N],
	pub parent_id: Option<usize>,
	pub dist_from_parent: f64,
}

pub struct RRTTree<const N: usize> {
	pub nodes: Vec<RRTNode<N>>,
}

impl<const N: usize> RRTTree<N> {
	fn new(state: [f64; N]) -> Self {
		let root = RRTNode { state, parent_id: None, dist_from_parent: 0.0 };
		Self { nodes: vec![root] }
	}

	fn add_node(&mut self, state: [f64; N], parent_id: usize, dist_from_parent: f64) -> usize {
		let id = self.nodes.len();
		let node = RRTNode { state, parent_id: Some(parent_id), dist_from_parent };
		self.nodes.push(node);
		id
	}

	fn reparent_node(&mut self, node_id: usize, parent_id: usize, dist_from_parent: f64) {
		let node = &mut self.nodes[node_id];
		node.parent_id = Some(parent_id);
		node.dist_from_parent = dist_from_parent;
	}

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
	}

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

pub trait RRTFuncs<const N: usize> {
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

pub struct RRT<'a, F: RRTFuncs<N>, const N: usize> {
	sample_space: SampleSpace<N>,
	fns: &'a F,
}

impl<'a, F: RRTFuncs<N>, const N: usize> RRT<'a, F, N> {
	pub fn new(sample_space: SampleSpace<N>, fns: &'a F) -> Self {
		Self { sample_space, fns }
	}

	pub fn plan(&mut self, start: [f64; N], goal: fn(&[f64; N]) -> bool,
				 max_step: f64, search_radius: f64, n_iter_max: u32) -> (Result<Vec<[f64; N]>, &'static str>, RRTTree<N>) {
		let (rrttree, final_node_ids) = self.grow_tree(start, goal, max_step, search_radius, n_iter_max);

		(self.get_best_solution(&rrttree, &final_node_ids), rrttree)
	}

	fn grow_tree(&self, start: [f64; N], goal: fn(&[f64; N]) -> bool,
				max_step: f64, search_radius: f64, n_iter_max: u32) -> (RRTTree<N>, Vec<usize>) {
		let mut final_node_ids = Vec::<usize>::new();
		let mut rrttree = RRTTree::new(start);
		let mut kdtree = KdTree::new(start);

		for _ in 0..n_iter_max {
			let mut new_state = self.sample_space.sample();
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

				let neighbour_ids = kdtree.nearest_neighbors(new_state, radius).iter()
					.filter(|node| self.fns.transition_validator(&node.state, &new_state))
					.map(|node| node.id)
					.collect();

				// Evaluate which is the best parent that we can possibly get
				let distances = rrttree.distances_from_common_ancestor(&neighbour_ids);
				let (parent_id, parent_distance) = zip(&neighbour_ids, &distances)
					.map(|(id,d)| (*id, *d))
					.min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
					.unwrap_or((kd_from.id, 0.0));

				// Add the node in the trees
				let dist_from_parent = {
					let parent = &rrttree.nodes[parent_id];
					self.fns.cost_evaluator(&parent.state, &new_state)
				};
				let new_node_id = rrttree.add_node(new_state, parent_id, dist_from_parent);
				kdtree.add(new_state, new_node_id);

				// Step 2: Perhaps we can reparent some of the neighbours to the new node
				let new_state_distance = parent_distance + dist_from_parent;
				for (neighbour_id, distance) in zip(&neighbour_ids, &distances) {
					if *neighbour_id == parent_id { continue; }
					let neighbour = &rrttree.nodes[*neighbour_id];
					// XXX We should call self.fns.transition_validator() again if the transition
					// validator is not symetric.
					let new_dist_to_parent = self.fns.cost_evaluator(&new_state, &neighbour.state);
					let new_distance = new_state_distance + new_dist_to_parent;
					if new_distance < *distance {
						rrttree.reparent_node(*neighbour_id, new_node_id, new_dist_to_parent);
					}
				}

				if goal(&new_state) {
					final_node_ids.push(new_node_id);
				}
			}
		}

		(rrttree, final_node_ids)
	}

	fn get_best_solution(&self, rrttree: &RRTTree<N>, final_node_ids: &Vec<usize>) -> Result<Vec<[f64; N]>, &'static str> {
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

	fn get_path_cost(&self, path: &Vec<[f64; N]>) -> f64 {
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
	impl RRTFuncs<2> for Funcs {}

	fn goal(state: &[f64; 2]) -> bool {
		(state[0] - 0.9).abs() < 0.05 && (state[1] - 0.9).abs() < 0.05
	}	

	let mut rrt = RRT::new(SampleSpace{low: [-1.0, -1.0], up: [1.0, 1.0]}, &Funcs{});

	let (path_result, _) = rrt.plan([0.0, 0.0], goal, 0.1, 1.0, 1000);
	assert!(path_result.as_ref().expect("No path found!").len() > 2);
}

#[test]
fn test_plan_on_map() {
	let mut m = Map::open("data/map3.pgm", [-1.0, -1.0], [1.0, 1.0]);

	fn goal(state: &[f64; 2]) -> bool {
		(state[0] - 0.0).abs() < 0.05 && (state[1] - 0.9).abs() < 0.05
	}	

	let mut rrt = RRT::new(SampleSpace{low: [-1.0, -1.0], up: [1.0, 1.0]}, &m);
	let (path_result, rrttree) = rrt.plan([0.0, -0.8], goal, 0.1, 5.0, 5000);

	assert!(path_result.as_ref().expect("No path found!").len() > 2);
	m.draw_tree(&rrttree);
	m.draw_path(path_result.unwrap());
	m.save("results/test_plan_on_map.pgm")
}
}
