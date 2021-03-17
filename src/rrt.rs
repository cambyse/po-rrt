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

#[derive(Clone, Copy)]
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

	/// returns the index of the belief_state.
	/// If not seen, also inserts in known beliefs.
	fn maybe_add_belief_state(&mut self, belief_state: &BeliefState) -> usize {
		self.belief_states
			.iter()
			.position(|bs| bs == belief_state)
			.unwrap_or_else(|| {
				let belief_id = self.belief_states.len();
				self.belief_states.push(belief_state.clone());
				belief_id
			})
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

	fn get_path_to(&self, mut node_id: usize) -> Vec<[f64; N]> {
		let mut path = vec![self.nodes[node_id].state];

		while let Some(parent_link) = &self.nodes[node_id].parent {
			node_id = parent_link.id;
			path.push(self.nodes[node_id].state);
		}

		path.reverse();
		path
	}
}

pub trait RRTFuncs<const N: usize> {
	fn state_validator(&self, _state: &[f64; N]) -> Reachable {
		Reachable::Always
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
	fns: &'a F,
}

impl<'a, F: RRTFuncs<N>, const N: usize> RRT<'a, F, N> {
	pub fn new(sampler: ContinuousSampler<N>, discrete_sampler: DiscreteSampler, fns: &'a F) -> Self {
		Self { sampler, discrete_sampler, fns }
	}

	pub fn plan(&mut self, start: [f64; N], start_belief_state: &BeliefState, goal: fn(&[f64; N]) -> bool,
				 max_step: f64, search_radius: f64, n_iter_max: u32) -> (RRTTree<N>, Vec<(usize, Vec<[f64; N]>)>) {
		let mut final_node_ids = Vec::<usize>::new();
		let mut rrttree = RRTTree::new();
		let mut kdtree = KdTree::new(start);

		{
			let belief_id = rrttree.maybe_add_belief_state(start_belief_state);
			rrttree.add_node(start, belief_id, None); // root node
		}

		for _ in 0..n_iter_max {
			let mut new_state = self.sampler.sample();
			let sampled_belief_id = self.discrete_sampler.sample(rrttree.belief_states.len());

			// XXX nearest_neighbor_filtered can return the root even if the filter closure disagrees.
			let canonical_neighbor = kdtree.nearest_neighbor_filtered(new_state, |id| rrttree.nodes[id].belief_state_id == sampled_belief_id); // log n
			steer(&canonical_neighbor.state, &mut new_state, max_step);

			let belief_state = &rrttree.belief_states[sampled_belief_id];
			if self.fns.state_validator(&new_state).is_compatible(&belief_state) {
				// RRT* algorithm
				// Step 1: Find all the neighbors near of new_state. The radius we use is from papers of RRT*

				// XXX Not sure why adding the 2.0 multiplicative constant is better
				let radius = 2.0 * {
					let n = rrttree.nodes.len() as f64;
					let s = search_radius * (n.ln()/n).powf(1.0/(N as f64));
					if s < max_step { s } else { max_step }
				};

				let mut neighbor_ids: Vec<usize> = kdtree
					.nearest_neighbors_filtered(new_state, radius, |id| rrttree.nodes[id].belief_state_id == sampled_belief_id)
					.iter()
					.map(|&kd_node| kd_node.id)
					.collect();
				if neighbor_ids.is_empty() {
					neighbor_ids.push(canonical_neighbor.id);
				}

				// Step 2: Retain only the neighbors that have valid transitions and are compatible with our belief
				let neighbor_ids: Vec<usize> = neighbor_ids.iter()
					.map(|&id| (id, self.fns.transition_validator(&rrttree.nodes[id].state, &new_state) ) )
					.filter(|(_, transition)| transition.is_compatible(belief_state) )
					.map(|(id, _)| id)
					.collect();

				if neighbor_ids.is_empty() {
					continue;
				}

				// Step 3: Compute useful metrics to pick the best parents.
				// distance between root (or common ancestor) and each neighbor
				let root_to_neighbor_distances = rrttree.distances_from_common_ancestor(&neighbor_ids);
				// distance between each neighbor and the new_state node
				let neighbor_to_new_state_distances = neighbor_ids.iter().cloned()
					.map(|id| self.fns.cost_evaluator(&rrttree.nodes[id].state, &new_state))
					.collect::<Vec<_>>();

				// Step 4: Find the best parent we can get.
				let (parent_id, _, parent_to_new_state_dist, root_to_new_state_distance) =
					izip!(&neighbor_ids, &root_to_neighbor_distances, &neighbor_to_new_state_distances)
						.map(|(&id, &rnd, &nnd)| (id, rnd, nnd, rnd+nnd))
						.min_by(|(_, _, _, a), (_, _, _, b)| a.partial_cmp(b).unwrap())
						.unwrap();

				// Step 5: Add the node new_state in the trees
				let parent_link = ParentLink { id: parent_id, dist: parent_to_new_state_dist };
				let new_node_id = rrttree.add_node(new_state, sampled_belief_id, Some(parent_link));
				kdtree.add(new_state, new_node_id);

				// Step 6: Reparent neighbors that could be better with the new_state node as a parent.
				for (&neighbor_id, &root_to_neighbor_distance, &neighbor_to_new_state_distance) in
						izip!(&neighbor_ids, &root_to_neighbor_distances, &neighbor_to_new_state_distances) {
					if neighbor_id == parent_id { continue; }

					// XXX We should call self.fns.transition_validator() again if the transition validator is not symetric.
					// XXX We also assume that cost from A to B is the same from B to A.

					let root_distance_if_new_state_parent = root_to_new_state_distance + neighbor_to_new_state_distance;
					if root_distance_if_new_state_parent < root_to_neighbor_distance {
						let parent_link = ParentLink { id: new_node_id, dist: neighbor_to_new_state_distance };
						rrttree.nodes[neighbor_id].parent = Some(parent_link);
					}
				}

				let belief_state = &rrttree.belief_states[sampled_belief_id];
				for child_belief_state in self.fns.observe_new_beliefs(&new_state, &belief_state) {
					let children_belief_state_id = rrttree.maybe_add_belief_state(&child_belief_state);
					let new_node_id = rrttree.add_node(new_state, children_belief_state_id, Some(parent_link));
					kdtree.add(new_state, new_node_id);
				}

				if goal(&new_state) {
					final_node_ids.push(new_node_id);

					let belief_state = &rrttree.belief_states[sampled_belief_id];
					println!("found leaf for belief:{}, {:?}", sampled_belief_id, &belief_state);
				}
			}
		}

		let best_goal_ids = {
			final_node_ids.sort_by_key(|&id| rrttree.nodes[id].belief_state_id);
			final_node_ids.group_by(|&id1, &id2| rrttree.nodes[id1].belief_state_id == rrttree.nodes[id2].belief_state_id)
				.map(|ids| {
					*zip(ids, rrttree.distances_from_common_ancestor(ids))
						.min_by(|(_,a),(_,b)| a.partial_cmp(b).unwrap())
						.unwrap()
						.0
				})
				.collect::<Vec<_>>()
		};

		let best_paths = best_goal_ids.iter()
			.map(|&id| (rrttree.nodes[id].belief_state_id, rrttree.get_path_to(id)))
			.collect();

		(rrttree, best_paths)
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

	let (_rrttree, paths) = rrt.plan([0.0, 0.0], &vec![1.0], goal, 0.1, 1.0, 1000);
	assert!(!paths.is_empty(), "No path found!");
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
	let (rrttree, paths) = rrt.plan([0.0, -0.8], &vec![0.25; 4], goal, 0.05, 5.0, 15000);
	assert!(!paths.is_empty(), "No path found!");

	let mut m = m.clone();
	m.resize(5);

	m.draw_tree(&rrttree);
	for (belief_id, path) in &paths {
		m.draw_path(path, crate::map_io::colors::color_map(*belief_id));
	}
	m.draw_zones_observability();
	m.save("results/test_rrt_on_map.pgm")
}
}
