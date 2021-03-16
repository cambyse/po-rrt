use itertools::{all, enumerate, izip, merge, zip};

use crate::{common::*};
use crate::nearest_neighbor::*;
use crate::sample_space::*;
use crate::map_io::*;
use std::{cmp::min, collections::{self, HashSet}};

#[derive(PartialEq, Eq, Hash)]
pub enum Reachable<'a> {
	Always,
	Never,
	Restricted(&'a WorldMask),
}

impl<'a> Reachable<'a> {
	#[allow(clippy::style)]
	pub fn is_compatible(&self, belief_state: &BeliefState) -> bool {
		match self {
			Self::Always => true,
			Self::Never => false,
			Self::Restricted(worldmask) => is_compatible(belief_state, worldmask),
		}
	}
}


pub struct ParentLink {
	pub id: usize,
	pub dist: f64,
	//pub world_mask: Reachable<'a>,
}

pub struct RRTNode<const N: usize> {
	pub id: usize,
	pub state: [f64; N],
	pub belief_state_id: usize,
	pub parent: Option<ParentLink>,
}

pub struct RRTTree<const N: usize> {
	pub nodes: Vec<RRTNode<N>>,
	pub belief_states: Vec<BeliefState>
}

impl<'a, const N: usize> RRTTree<N> {
	fn new() -> Self {
		Self { nodes: vec![], belief_states: Default::default() }
	}

	fn add_node(&mut self, state: [f64; N], belief_state_id: usize, parent: Option<ParentLink>) -> usize {
		let id = self.nodes.len();
		let node = RRTNode { id, state, belief_state_id, parent };
		self.nodes.push(node);
		assert!(belief_state_id < self.belief_states.len());
		id
	}

	fn add_belief_state(&mut self, belief_state: BeliefState) -> usize {
		let belief_id = self.belief_states.len();
		self.belief_states.push(belief_state);
		belief_id
	}

	fn reparent_node(&mut self, node_id: usize, parent: ParentLink) {
		let node = &mut self.nodes[node_id];
		node.parent = Some(parent);
	}

	fn distances_from_common_ancestor(&self, leaf_ids: &[usize]) -> Vec<f64> {
		if leaf_ids.is_empty() {
			return vec![];
		}

		if leaf_ids.len() == 1 {
			return vec![0.0];
		}

		let compute_distance_from_root = |mut node_id: usize| {
			let mut cost = 0.0;
			while let Some(parent_link) = &self.nodes[node_id].parent {
				cost += parent_link.dist;
				node_id = parent_link.id;
			}
			cost
		};

		leaf_ids.iter()
			.map(|id| compute_distance_from_root(*id))
			.collect()
	}

	fn get_path_to(&self, _id: usize) -> Vec<(WorldMask, Vec<[f64; N]>)> {
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
	fn state_validator(&self, _state: &[f64; N]) -> Option<WorldMask> {
		None
	}

	fn transition_validator(&self, _from: &[f64; N], _to: &[f64; N]) -> Reachable {
		Reachable::Always
	}

	fn cost_evaluator(&self, a: &[f64; N], b: &[f64; N]) -> f64 {
		norm2(a,b)
	}

	#[allow(clippy::style)]
	fn observe_new_beliefs(&self, _state: &[f64; N], belief_state: &BeliefState) -> Vec<BeliefState> { 
		vec![belief_state.to_owned()]
	}
}

pub struct RRT<'a, F: RRTFuncs<N>, const N: usize> {
	sampler: ContinuousSampler<N>,
	discrete_sampler: DiscreteSampler,
	rrttree: RRTTree<N>,
	fns: &'a F,
}

impl<'a, F: RRTFuncs<N>, const N: usize> RRT<'a, F, N> {
	pub fn new(sampler: ContinuousSampler<N>, discrete_sampler: DiscreteSampler, fns: &'a F) -> Self {
		let rrttree = RRTTree::new();
		Self { sampler, discrete_sampler, rrttree, fns }
	}

	pub fn plan(&mut self, start: [f64; N], belief_state: &BeliefState, goal: fn(&[f64; N]) -> bool,
				 max_step: f64, search_radius: f64, n_iter_max: u32) -> Vec<(WorldMask, Vec<[f64; N]>)> {
		let final_node_ids = self.grow_tree(start, belief_state, goal, max_step, search_radius, n_iter_max);
		self.get_best_solution(&final_node_ids)
	}

	fn grow_tree(&mut self, start: [f64; N], start_belief_state: &BeliefState, goal: fn(&[f64; N]) -> bool,
				max_step: f64, search_radius: f64, n_iter_max: u32) -> Vec<usize> {
		let mut final_node_ids = Vec::<usize>::new();
		self.rrttree = RRTTree::new();
		let mut kdtree = KdTree::new(start);

		{
			let belief_id = self.rrttree.add_belief_state(start_belief_state.clone());
			self.rrttree.add_node(start, belief_id, None); // root node
		}


		for _ in 0..n_iter_max {
			let mut new_state = self.sampler.sample();
			let sampled_belief_id = self.discrete_sampler.sample(self.rrttree.belief_states.len());

			let kd_from = kdtree.nearest_neighbor_filtered(new_state, |id| self.rrttree.nodes[id].belief_state_id == sampled_belief_id ); // log n

			steer(&kd_from.state, &mut new_state, max_step);

			let state_validity = self.fns.state_validator(&new_state);

			if let Some(state_validity) = state_validity {
				let belief_state = &self.rrttree.belief_states[sampled_belief_id];
				if is_compatible(belief_state, &state_validity) {
					// RRT* algorithm
					// Step 1: Find the best parent we can get
					// First, we find the neighbors in a specific radius of new_state.
					let radius = {
						let n = self.rrttree.nodes.len() as f64;
						let s = search_radius * (n.ln()/n).powf(1.0/(N as f64));
						if s < max_step { s } else { max_step }
					};

					// Second, add node and connect to best parent
					let mut neighbour_ids: Vec<usize> = kdtree.nearest_neighbors_filtered(new_state, radius, |id| self.rrttree.nodes[id].belief_state_id == sampled_belief_id ).iter()
						.map(|&kd_node| kd_node.id)
						.collect();

					if neighbour_ids.is_empty() { neighbour_ids.push(kd_from.id); }

					let neighbour_ids: Vec<usize> = neighbour_ids.iter()
						.map(|&id| (id, self.fns.transition_validator(&self.rrttree.nodes[id].state, &new_state) ) )
						.filter(|(_, transition)| transition.is_compatible(belief_state) )
						.map(|(id, _)| id)
						.collect();

					if neighbour_ids.is_empty() { continue; }

					//println!("number of neighbors:{}", neighbour_ids.len());

					// Evaluate which is the best parent that we can possibly get
					let distances = self.rrttree.distances_from_common_ancestor(&neighbour_ids);
					let (parent_id, parent_distance) = zip(&neighbour_ids, &distances)
							.map(|(id,d)| (*id, *d))
							.min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
							.unwrap();
					
					let dist_from_parent = {
						let parent = &self.rrttree.nodes[parent_id];
						self.fns.cost_evaluator(&parent.state, &new_state) 
					};

					let parent_link = ParentLink{id: parent_id, dist: dist_from_parent};
					
					let new_node_id = self.rrttree.add_node(new_state, sampled_belief_id, Some(parent_link));
					kdtree.add(new_state, new_node_id);

					// Reparent					
					let new_state_distance = parent_distance + dist_from_parent;
					for (neighbour_id, distance) in zip(&neighbour_ids, &distances) {
						if *neighbour_id == parent_id { continue; }
						let neighbour = &self.rrttree.nodes[*neighbour_id];
						// XXX We should call self.fns.transition_validator() again if the transition
						// validator is not symetric.
						let new_dist_to_parent = self.fns.cost_evaluator(&new_state, &neighbour.state);
						let new_distance = new_state_distance + new_dist_to_parent;
						if new_distance < *distance {
							self.rrttree.nodes[*neighbour_id].parent = Some(ParentLink{id: new_node_id, dist: new_dist_to_parent});
						}
					}
					
					// Third, transition to other beliefs
					/*let belief_state = &self.rrttree.belief_states[sampled_belief_id];
					let children_belief_states = self.fns.observe_new_beliefs(&new_state, &belief_state);

					for child_belief_state in children_belief_states.iter() {
						/*let children_belief_state_id = self.rrttree.add_belief_state(child_belief_state.clone());
						let parent_link = ParentLink{id: new_node_id, dist: 0.0}; // need to have dist in the edge?

						let new_node_id = self.rrttree.add_node(new_state, children_belief_state_id, Some(parent_link), dist_from_root);
						kdtree.add(new_state, new_node_id);*/
					}*/
					
					if goal(&new_state) {
						final_node_ids.push(new_node_id);
						let belief_state = &self.rrttree.belief_states[sampled_belief_id];

						println!("found leaf for belief:{}, {:?}", sampled_belief_id, &belief_state);
					}
				}
			}
		}

		final_node_ids
	}

	fn get_best_solution(&self, _final_node_ids: &[usize]) -> Vec<(WorldMask, Vec<[f64; N]>)> {
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
	impl RRTFuncs<2> for Funcs {}

	fn goal(state: &[f64; 2]) -> bool {
		(state[0] - 0.9).abs() < 0.05 && (state[1] - 0.9).abs() < 0.05
	}	

	let mut rrt = RRT::new(ContinuousSampler::new([-1.0, -1.0], [1.0, 1.0]),
		DiscreteSampler::new(),
		&Funcs{});

	let _path_result = rrt.plan([0.0, 0.0], &vec![1.0], goal, 0.1, 1.0, 1000);
	//assert!(path_result.as_ref().expect("No path found!").len() > 2);
}

#[test]
fn test_plan_on_map() {
	let mut m = Map::open("data/map2.pgm", [-1.0, -1.0], [1.0, 1.0]);
	m.add_zones("data/map2_zone_ids.pgm", 0.2);

	fn goal(state: &[f64; 2]) -> bool {
		(state[0] - 0.0).abs() < 0.05 && (state[1] - 0.9).abs() < 0.05
	}	

	let mut rrt = RRT::new(ContinuousSampler::new([-1.0, -1.0], [1.0, 1.0]),
		DiscreteSampler::new(),
		&m);
	let _path_result = rrt.plan([0.0, -0.8], &vec![0.25; 4], goal, 0.1, 5.0, 5000);

	//assert!(path_result.as_ref().expect("No path found!").len() > 2);
	let mut m = m.clone();
	m.resize(5);

	m.draw_tree(&rrt.rrttree);
	//m.draw_path(path_result.unwrap());
	m.draw_zones_observability();
	m.save("results/test_rrt_on_map.pgm")
}
}
