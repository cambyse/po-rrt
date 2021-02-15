use itertools::{all, enumerate, izip, merge, zip};

use crate::{common::*, prm::Reachability};
use crate::nearest_neighbor::*;
use crate::sample_space::*;
use crate::map_io::*;
use std::{cmp::min, collections::{self, HashSet}};
use bitvec::prelude::*;

pub type WorldId = u32;
pub type WorldMask = BitVec;

#[derive(PartialEq, Eq, Hash)]
pub enum Reachable<'a> {
	Always,
	Never,
	Restricted(&'a WorldMask),
}

pub struct ParentLink<'a> {
	pub id: usize,
	pub dist: f64,
	pub world_mask: Reachable<'a>,
}

pub struct RRTNode<'a, const N: usize> {
	pub state: [f64; N],
	pub parents: Vec<ParentLink<'a>>,
}

pub struct RRTTree<'a, const N: usize> {
	pub nodes: Vec<RRTNode<'a, N>>,
}

impl<'a, const N: usize> RRTTree<'a, N> {
	fn new() -> Self {
		Self { nodes: vec![] }
	}

	fn add_node(&mut self, state: [f64; N], parents: Vec<ParentLink<'a>>) -> usize {
		let id = self.nodes.len();
		let node = RRTNode { state, parents };
		self.nodes.push(node);
		id
	}

	fn reparent_node(&mut self, node_id: usize, parents: Vec<ParentLink<'a>>) {
		let node = &mut self.nodes[node_id];
		node.parents = parents;
	}

	fn distances_from_common_ancestor(&self, leaf_ids: &Vec<usize>, world_mask: &Reachable) -> Vec<f64> {
		if leaf_ids.is_empty() {
			return vec![];
		}

		if leaf_ids.len() == 1 {
			return vec![0.0];
		}

		let compute_distance_from_root = |mut node_id: usize| {
			return 0.0;
			/*
			let mut cost = 0.0;
			while node_id != 0 {
				let node = &self.nodes[node_id];
				cost += node.dist_from_parent;
				node_id = node.parent_id.unwrap();
			}
			cost
			*/
		};

		leaf_ids.iter()
			.map(|id| compute_distance_from_root(*id))
			.collect()
	}

	fn get_path_to(&self, id: usize) -> Vec<(WorldMask, Vec<[f64; N]>)> {
		return vec![];
		/* 
		let mut path = Vec::new();

		let mut node = &self.nodes[id];
		path.push(node.state);

		/*
		while let Some(id) = node.parent_id {
			node = &self.nodes[id];
			path.push(node.state);
		}
		*/

		path.reverse();
		path
		*/
	}
}

pub trait RRTFuncs<const N: usize> {
	fn state_validator(&self, _state: &[f64; N]) -> bool {
		true
	}

	fn transition_validator(&self, _from: &[f64; N], _to: &[f64; N]) -> Reachable {
		Reachable::Always
	}

	fn cost_evaluator(&self, a: &[f64; N], b: &[f64; N]) -> f64 {
		norm2(a,b)
	}
}

pub struct RRT<'a, F: RRTFuncs<N>, const N: usize> {
	sampler: ContinuousSampler<N>,
	rrttree: RRTTree<'a, N>,
	fns: &'a F,
}

impl<'a, F: RRTFuncs<N>, const N: usize> RRT<'a, F, N> {
	pub fn new(sampler: ContinuousSampler<N>, fns: &'a F) -> Self {
		let rrttree = RRTTree::new();
		Self { sampler, rrttree, fns }
	}

	pub fn plan(&mut self, start: [f64; N], goal: fn(&[f64; N]) -> bool,
				 max_step: f64, search_radius: f64, n_iter_max: u32) -> Vec<(WorldMask, Vec<[f64; N]>)> {
		let final_node_ids = self.grow_tree(start, goal, max_step, search_radius, n_iter_max);
		self.get_best_solution(&final_node_ids)
	}

	fn grow_tree(&mut self, start: [f64; N], goal: fn(&[f64; N]) -> bool,
				max_step: f64, search_radius: f64, n_iter_max: u32) -> Vec<usize> {
		let mut final_node_ids = Vec::<usize>::new();
		self.rrttree = RRTTree::new();
		let mut kdtree = KdTree::new(start);

		self.rrttree.add_node(start, vec![]); // root node

		for _ in 0..n_iter_max {
			let mut new_state = self.sampler.sample();
			let kd_from = kdtree.nearest_neighbor(new_state);

			steer(&kd_from.state, &mut new_state, max_step);

			if self.fns.state_validator(&new_state) {
				// RRT* algorithm
				// Step 1: Find the best parent we can get
				// First, we find the neighbors in a specific radius of new_state.
				let radius = {
					let n = self.rrttree.nodes.len() as f64;
					let s = search_radius * (n.ln()/n).powf(1.0/(N as f64));
					if s < max_step { s } else { max_step }
				};

				let worldmask_neighbours = {
					let mut neighbours = kdtree.nearest_neighbors(new_state, radius);
					if neighbours.is_empty() { neighbours.push(kd_from); }

					let mut map = collections::HashMap::<Reachable, Vec<usize>>::new();

					for node in neighbours {
						let transition = self.fns.transition_validator(&node.state, &new_state);
						if transition == Reachable::Never { continue; }
						if let Some(nodes) = map.get_mut(&transition) {
							nodes.push(node.id);
						} else {
							map.insert(transition, vec![node.id]);
						}
					}

					map
				};

				let mut parents = vec![];
				for (world_mask, neighbour_ids) in worldmask_neighbours.into_iter() {
					// Evaluate which is the best parent that we can possibly get
					let distances = self.rrttree.distances_from_common_ancestor(&neighbour_ids, &world_mask);
					let (parent_id, parent_distance) = zip(&neighbour_ids, &distances)
						.map(|(id,d)| (*id, *d))
						.min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
						.unwrap();

					let dist_from_parent = {
						let parent = &self.rrttree.nodes[parent_id];
						self.fns.cost_evaluator(&parent.state, &new_state)
					};
					let parent_link = ParentLink { id: parent_id, dist: dist_from_parent, world_mask };
					// TODO reparent

					/*
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
					 */

					parents.push(parent_link);
				}

				let new_node_id = self.rrttree.add_node(new_state, parents);
				kdtree.add(new_state, new_node_id);

				if goal(&new_state) {
					final_node_ids.push(new_node_id);
				}
			}
		}

		final_node_ids
	}

	fn get_best_solution(&self, final_node_ids: &Vec<usize>) -> Vec<(WorldMask, Vec<[f64; N]>)> {
		return vec![];
		/*
		final_node_ids.iter()
			.map(|id| {
				let path = self.rrttree.get_path_to(*id);
				let cost = self.get_path_cost(&path);
				(path, cost)
			})
			.min_by(|(_,a),(_,b)| a.partial_cmp(b).expect("NaN found"))
			.map(|(p, _)| p)
			.ok_or("No solution found")
			*/
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

	let mut rrt = RRT::new(ContinuousSampler::new([-1.0, -1.0], [1.0, 1.0]), &Funcs{});

	let path_result = rrt.plan([0.0, 0.0], goal, 0.1, 1.0, 1000);
	//assert!(path_result.as_ref().expect("No path found!").len() > 2);
}

#[test]
fn test_plan_on_map() {
	let mut m = Map::open("data/map2.pgm", [-1.0, -1.0], [1.0, 1.0]);
	m.add_zones("data/map2_zone_ids.pgm");

	fn goal(state: &[f64; 2]) -> bool {
		(state[0] - 0.0).abs() < 0.05 && (state[1] - 0.9).abs() < 0.05
	}	

	let mut rrt = RRT::new(ContinuousSampler::new([-1.0, -1.0], [1.0, 1.0]), &m);
	let path_result = rrt.plan([0.0, -0.8], goal, 0.1, 5.0, 5000);

	//assert!(path_result.as_ref().expect("No path found!").len() > 2);
	let mut m = m.clone();

	m.draw_tree(&rrt.rrttree);
	//m.draw_path(path_result.unwrap());
	m.save("results/test_rrt_on_map.pgm")
}
}
