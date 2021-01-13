use crate::common::*;
use crate::nearest_neighbor::*;
use crate::sample_space::*;
use crate::map_io::*;
use std::vec::Vec;
use core::cell::RefCell;
use std::rc::{Weak, Rc};


pub struct RRTTree<const N: usize> {
	root: NodeRef<N>,
	nodes: Vec<NodeRef<N>>,
}

impl<const N: usize> RRTTree<N> {
	fn new(state: [f64; N]) -> RRTTree<N> {
		let root = Rc::new(RefCell::new(Node{
				id: 0,
				state,
				parent: Weak::new(),
				children: Vec::<NodeRef<N>>::new()
			}));

		RRTTree {
			root: root.clone(),
			nodes: vec![root.clone()],
		}
	}

	fn get_node(&self, id: usize) -> NodeRef<N> {
		self.nodes[id].clone()
	}

	fn add_node(&mut self, parent: NodeRef<N>, state: [f64; N]) -> NodeRef<N> {
		let node = Rc::new(RefCell::new(Node{
				id: self.nodes.len(),
				state,
				parent: NodeRef::downgrade(&parent),
				children: Vec::<NodeRef<N>>::new()
			}));

		parent.borrow_mut().children.push(node.clone());

		self.nodes.push(node.clone());

		node.clone()
	}
}

pub struct RRT<'a, const N: usize> {
	sample_space: SampleSpace<N>,
	state_validator : &'a dyn Fn(&[f64; N]) -> bool,
	transition_validator : &'a dyn Fn(&[f64; N], &[f64; N]) -> bool,
	cost_evaluator : &'a dyn Fn(&[f64; N], &[f64; N]) -> f64,
}

impl<const N: usize> RRT<'_, N> {
	pub fn plan(&mut self, start: [f64; N], goal: fn(&[f64; N]) -> bool, max_step: f64, n_iter_max: u32) -> (Result<Vec<[f64; N]>, &str>, RRTTree<N>) {
		let (rrttree, final_nodes) = self.grow_tree(start, goal, max_step, n_iter_max);

		(self.get_best_solution(&final_nodes), rrttree)
	}

	fn grow_tree(&self, start: [f64; N], goal: fn(&[f64; N]) -> bool, max_step: f64, n_iter_max: u32) -> (RRTTree<N>, Vec<NodeRef<N>>) {
		let mut final_nodes = Vec::<NodeRef<N>>::new();
		let mut rrttree = RRTTree::new(start);
		let mut kdtree = KdTree::new(start); // why no need to be mutable?

		for _ in 0..n_iter_max {
			let mut new_state = self.sample_space.sample();
			let kd_from = kdtree.nearest_neighbor(new_state);

			new_state = backtrack(&kd_from.state, &mut new_state, max_step);

			if (self.state_validator)(&new_state) {
				if (self.transition_validator)(&kd_from.state, &new_state) {
					let from = rrttree.get_node(kd_from.id);
					let to = rrttree.add_node(from, new_state);
					
					kdtree.add(new_state, to.borrow().id);

					if goal(&new_state) {
						final_nodes.push(to);
					}
				}
			}
		}

		//println!("number of final nodes: {}", final_nodes.len());

		(rrttree, final_nodes)
	}

	fn get_best_solution(&self, final_nodes: &Vec<NodeRef<N>>) -> Result<Vec<[f64; N]>, &str> {
		let mut candidates : Vec<Vec<[f64; N]>> = Vec::with_capacity(final_nodes.len());
		let mut costs : Vec<f64> = Vec::with_capacity(final_nodes.len());

		if final_nodes.len() == 0 {
			return Err("No solution found");
		}

		for final_node in final_nodes.iter() {
			let path = self.get_path_to(final_node);
			let cost = self.get_path_cost(&path);
			candidates.push(path);
			costs.push(cost);
		}

		let mut best_index = 0;
		let mut best_cost = f64::INFINITY;

		for i in 0..costs.len() {
			if costs[i] < best_cost {
				best_index = i;
				best_cost = costs[i];
			}
		}

		Ok(candidates[best_index].clone())
	}

	fn get_path_to(&self, final_node: &NodeRef<N>) -> Vec<[f64; N]> { // move out of class?
		let mut path = Vec::<[f64; N]>::new();
		path.push(final_node.borrow().state);

		let mut parent = final_node.borrow().parent.upgrade().clone();
		
		while !parent.is_none() {
			path.push(parent.clone().unwrap().borrow().state);
			parent = parent.clone().unwrap().borrow().parent.upgrade().clone();
		}

		path.reverse();

		path
	}

	fn get_path_cost(&self, path: &Vec<[f64; N]>) -> f64 {
		let mut cost: f64 = 0.0;

		for i in 1..path.len() {
			cost += (self.cost_evaluator)(&path[i-1], &path[i]);
		}

		cost
	}
}

#[cfg(test)]
mod tests {

use super::*;

#[test]
fn test_plan_empty_space() {
	fn state_validator(_state: &[f64; 2]) -> bool {
		true
	}	

	fn transition_validator(_from: &[f64; 2], _to: &[f64; 2]) -> bool {
		true
	}	

	fn goal(state: &[f64; 2]) -> bool {
		(state[0] - 0.9).abs() < 0.05 && (state[1] - 0.9).abs() < 0.05
	}	

	let mut rrt = RRT{
		sample_space: SampleSpace{low: [-1.0, -1.0], up: [1.0, 1.0]},
		state_validator: &state_validator,
		transition_validator: &transition_validator,
		cost_evaluator: &norm2,
	};

	let (path_result, _) = rrt.plan([0.0, 0.0], goal, 0.1, 1000);

	assert!(path_result.clone().expect("No path found!").len() > 2); // why do we need to clone?!
}

#[test]
fn test_plan_on_map() {
	let m = Map::open("data/map3.pgm", [-1.0, -1.0], [1.0, 1.0]);

	let state_validator = |state: &[f64; 2]| -> bool {
		m.is_state_valid(state)
	};	

	fn transition_validator(_from: &[f64; 2], _to: &[f64; 2]) -> bool {
		true
	}	

	fn goal(state: &[f64; 2]) -> bool {
		(state[0] - 0.0).abs() < 0.05 && (state[1] - 0.9).abs() < 0.05
	}	

	let mut rrt = RRT{
		sample_space: SampleSpace{low: [-1.0, -1.0], up: [1.0, 1.0]},
		state_validator: &state_validator,
		transition_validator: &transition_validator,
		cost_evaluator: &norm2,
	};

	let (path_result, rrttree) = rrt.plan([0.0, -0.8], goal, 0.1, 5000);

	assert!(path_result.clone().expect("No path found!").len() > 2); // why do we need to clone?!
	
	let mut m = m.clone();
	m.draw_tree(rrttree.root);
	m.draw_path(path_result.unwrap());
	m.save("results/test_plan_on_map.pgm")
}
}
