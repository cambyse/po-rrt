use crate::common::*;
use crate::nearest_neighbor::*;
use crate::sample_space::*;
use crate::map_io::*;
use std::vec::Vec;
use core::cell::RefCell;
use std::rc::{Weak, Rc};

pub struct RRTNode<const N: usize> {
	pub state: [f64; N],
	pub children_ids: Vec<usize>,
	pub parent_id: Option<usize>,
}

impl<const N: usize> RRTNode<N> {
	pub fn new(state: [f64; N], parent_id: Option<usize>) -> Self {
		Self { state, children_ids: Vec::new(), parent_id }
	}
}

pub struct RRTTree<const N: usize> {
	pub nodes: Vec<RRTNode<N>>,
}

impl<const N: usize> RRTTree<N> {
	fn add_node(&mut self, state: [f64; N], parent_id: Option<usize>) -> usize {
		let id = self.nodes.len();
		let node = RRTNode::new(state, parent_id);
		if let Some(parent_id) = parent_id {
			self.nodes[parent_id].children_ids.push(id);
		}
		self.nodes.push(node);
		id
	}

	fn new(state: [f64; N]) -> Self {
		let mut self_ = Self { nodes: Vec::new() };
		self_.add_node(state, None);
		self_
	}

	fn get_path_to(&self, id: usize) -> Vec<[f64; N]> { // move out of class?
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
	sample_space: SampleSpace<N>,
	fns: F,
}

impl<F: RTTFuncs<N>, const N: usize> RRT<F, N> {
	pub fn new(sample_space: SampleSpace<N>, fns: F) -> Self {
		Self { sample_space, fns }
	}

	pub fn plan(&mut self, start: [f64; N], goal: fn(&[f64; N]) -> bool, max_step: f64, n_iter_max: u32) -> (Result<Vec<[f64; N]>, &str>, RRTTree<N>) {
		let (rrttree, final_node_ids) = self.grow_tree(start, goal, max_step, n_iter_max);

		(self.get_best_solution(&rrttree, &final_node_ids), rrttree)
	}

	fn grow_tree(&self, start: [f64; N], goal: fn(&[f64; N]) -> bool, max_step: f64, n_iter_max: u32) -> (RRTTree<N>, Vec<usize>) {
		let mut final_node_ids = Vec::<usize>::new();
		let mut rrttree = RRTTree::new(start);
		let mut kdtree = KdTree::new(start);

		for _ in 0..n_iter_max {
			let mut new_state = self.sample_space.sample();
			let kd_from = kdtree.nearest_neighbor(new_state);

			new_state = backtrack(&kd_from.state, &mut new_state, max_step);

			if self.fns.state_validator(&new_state) {
				if self.fns.transition_validator(&kd_from.state, &new_state) {
					let new_node_id = rrttree.add_node(new_state, Some(kd_from.id));
					kdtree.add(new_state, new_node_id);

					if goal(&new_state) {
						final_node_ids.push(new_node_id);
					}
				}
			}
		}

		//println!("number of final nodes: {}", final_nodes.len());

		(rrttree, final_node_ids)
	}

	fn get_best_solution(&self, rrttree: &RRTTree<N>, final_node_ids: &Vec<usize>) -> Result<Vec<[f64; N]>, &str> {
		if final_node_ids.len() == 0 {
			return Err("No solution found");
		}

		let mut best_path = rrttree.get_path_to(final_node_ids[0]);
		let mut best_cost = self.get_path_cost(&best_path);

		for final_node_id in &final_node_ids[1..] {
			let path = rrttree.get_path_to(*final_node_id);
			let cost = self.get_path_cost(&path);
			if cost < best_cost {
				best_path = path;
				best_cost = cost;
			}
		}

		Ok(best_path)
	}

	fn get_path_cost(&self, path: &Vec<[f64; N]>) -> f64 {
		let mut cost = 0.0;
		for (prev, next) in pairwise_iter(path) {
			cost += self.fns.cost_evaluator(prev, next)
		}
		cost
	}
}

#[cfg(test)]
mod tests {

use super::*;

#[test]
fn test_plan_empty_space() {
	struct Funcs {}
	impl RTTFuncs<2> for Funcs {}

	fn goal(state: &[f64; 2]) -> bool {
		(state[0] - 0.9).abs() < 0.05 && (state[1] - 0.9).abs() < 0.05
	}	

	let mut rrt = RRT::new(SampleSpace{low: [-1.0, -1.0], up: [1.0, 1.0]}, Funcs{});

	let (path_result, _) = rrt.plan([0.0, 0.0], goal, 0.1, 1000);

	assert!(path_result.clone().expect("No path found!").len() > 2); // why do we need to clone?!
}

#[test]
fn test_plan_on_map() {
	let m = Map::open("data/map3.pgm", [-1.0, -1.0], [1.0, 1.0]);
	let m2 = m.clone();

	struct Funcs {
		m: Map,
	}

	impl RTTFuncs<2> for Funcs {
		fn state_validator(&self, state: &[f64; 2]) -> bool {
			self.m.is_state_valid(state)
		}
	}

	fn goal(state: &[f64; 2]) -> bool {
		(state[0] - 0.0).abs() < 0.05 && (state[1] - 0.9).abs() < 0.05
	}	

	let mut rrt = RRT::new(SampleSpace{low: [-1.0, -1.0], up: [1.0, 1.0]}, Funcs{m});

	let (path_result, rrttree) = rrt.plan([0.0, -0.8], goal, 0.1, 5000);

	assert!(path_result.clone().expect("No path found!").len() > 2); // why do we need to clone?!
	
	let mut m = m2;
	m.draw_tree(&rrttree);
	m.draw_path(path_result.unwrap());
	m.save("results/test_plan_on_map.pgm")
}
}
