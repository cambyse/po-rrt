use itertools::{all, enumerate, izip, merge, zip};

use crate::{common::*};
use crate::nearest_neighbor::*;
use crate::sample_space::*;
use crate::map_io::*;
use crate::belief_graph::*;
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

pub struct RRTDefaultSamplers<const N: usize> {
	pub state_sampler: ContinuousSampler<N>,
	pub belief_state_sampler: DiscreteSampler,
}

/*
impl Default for RRTDefaultSamplers<2> {
	fn default() -> Self {
        todo!()
    }
}*/

impl<const N: usize> SampleFuncs<N> for RRTDefaultSamplers<N> {
    fn sample_state(&mut self) -> [f64; N] {
        self.state_sampler.sample()
    }

    fn sample_discrete(&mut self, n: usize) -> usize {
        self.belief_state_sampler.sample(n)
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
	pub node_type: BeliefNodeType,
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

	fn add_node(&mut self, state: [f64; N], belief_state_id: usize, node_type: BeliefNodeType, parent: Option<ParentLink>) -> usize {
		let id = self.nodes.len();
		let node = RRTNode { id, state, belief_state_id, node_type, parent };
		self.nodes.push(node);
		assert!(belief_state_id < self.belief_states.len());
		id
	}

	/// returns the index of the belief_state.
	/// If not seen, also inserts in known beliefs.
	#[allow(clippy::style)]
	fn maybe_add_belief_state(&mut self, belief_state: &BeliefState) -> usize {
		self.belief_states
			.iter()
			.position(|bs| bs == belief_state)
			.unwrap_or_else(|| {
				let belief_id = self.belief_states.len();
				println!("new belief!:{}, {:?}", belief_id, belief_state);
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

pub trait SampleFuncs<const N: usize> {
	fn sample_state(&mut self) -> [f64; N];
	fn sample_discrete(&mut self, n: usize) -> usize;
} 	


pub struct RRT<'a, FS: SampleFuncs<N>,  F: RRTFuncs<N>, const N: usize> {
	samplers: &'a mut FS,
	fns: &'a F,
}

impl<'a, FS: SampleFuncs<N>,  F: RRTFuncs<N>, const N: usize> RRT<'a, FS, F, N> {
	pub fn new(samplers: &'a mut FS, fns: &'a F) -> Self {
		Self { samplers, fns }
	}

	#[allow(clippy::style, clippy::type_complexity)]
	pub fn plan(&mut self, start: [f64; N], start_belief_state: &BeliefState, goal: fn(&[f64; N]) -> bool,
				 max_step: f64, search_radius: f64, n_iter_max: u32) -> (RRTTree<N>, Policy<N>, Vec<(usize, Vec<[f64; N]>)>) {
		let mut final_node_ids = Vec::<usize>::new();
		let mut rrttree = RRTTree::new();
		let mut kdtree = KdTree::new(start);

		{
			let belief_id = rrttree.maybe_add_belief_state(start_belief_state);
			rrttree.add_node(start, belief_id, BeliefNodeType::Action, None); // root node
		}

		for _ in 0..n_iter_max {
			let mut new_state = self.samplers.sample_state();
			let sampled_belief_id = self.samplers.sample_discrete(rrttree.belief_states.len());

			// XXX nearest_neighbor_filtered can return the root even if the filter closure disagrees.
			let canonical_neighbor = kdtree.nearest_neighbor_filtered(new_state, |id| rrttree.nodes[id].belief_state_id == sampled_belief_id &&  rrttree.nodes[id].node_type != BeliefNodeType::Observation); // n log n
			steer(&canonical_neighbor.state, &mut new_state, max_step);

			//
			//assert!(rrttree.nodes[canonical_neighbor.id].node_type != BeliefNodeType::Observation);
			//

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
					.nearest_neighbors_filtered(new_state, radius, |id| rrttree.nodes[id].belief_state_id == sampled_belief_id && rrttree.nodes[id].node_type != BeliefNodeType::Observation)
					.iter()
					.map(|&kd_node| kd_node.id)
					.collect();
				if neighbor_ids.is_empty() {
					neighbor_ids.push(canonical_neighbor.id);
				}

				//
				//for &id in &neighbor_ids {
				//	assert!(rrttree.nodes[id].node_type != BeliefNodeType::Observation);
				//}
				//

				// Step 2: Retain only the neighbors that have valid transitions and are compatible with our belief
				let neighbor_ids: Vec<usize> = neighbor_ids.iter()
					.map(|&id| (id, self.fns.transition_validator(&rrttree.nodes[id].state, &new_state) ) )
					.filter(|(id, transition)| transition.is_compatible(belief_state) && rrttree.nodes[*id].node_type != BeliefNodeType::Observation)
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

				//
				//assert!(rrttree.nodes[parent_id].node_type != BeliefNodeType::Observation);
				//

				// Step 5: Add the node new_state in the trees
				let belief_state = &rrttree.belief_states[sampled_belief_id];
				let children_belief_states = self.fns.observe_new_beliefs(&new_state, &belief_state);
				let new_node_type =  if children_belief_states.len() > 1 { BeliefNodeType::Observation } else { BeliefNodeType::Action };
				let parent_link = ParentLink { id: parent_id, dist: parent_to_new_state_dist };
				let new_node_id = rrttree.add_node(new_state, sampled_belief_id, new_node_type, Some(parent_link));
				kdtree.add(new_state, new_node_id);

				// Step 6: Reparent neighbors that could be better with the new_state node as a parent.
				if rrttree.nodes[new_node_id].node_type == BeliefNodeType::Action {
					if true {
						for (&neighbor_id, &root_to_neighbor_distance, &neighbor_to_new_state_distance) in
								izip!(&neighbor_ids, &root_to_neighbor_distances, &neighbor_to_new_state_distances) {
							if neighbor_id == parent_id { continue; }

							// XXX We should call self.fns.transition_validator() again if the transition validator is not symetric.
							// XXX We also assume that cost from A to B is the same from B to A.

							let root_distance_if_new_state_parent = root_to_new_state_distance + neighbor_to_new_state_distance;
							if root_distance_if_new_state_parent < root_to_neighbor_distance {
								// reparent
								if let Some(previous_parent) =  rrttree.nodes[neighbor_id].parent {
									let previous_parent_id = previous_parent.id;
									let previous_parent = &rrttree.nodes[previous_parent_id];
									
									if let Some(grand_parent) = previous_parent.parent {
										let grand_parent = &rrttree.nodes[grand_parent.id];

										if grand_parent.belief_state_id != sampled_belief_id {
											continue;
										}
									}

									if previous_parent.belief_state_id != sampled_belief_id {
										assert!(previous_parent.node_type == BeliefNodeType::Observation);
										continue;
									}

									if previous_parent.node_type == BeliefNodeType::Observation {
										// no reparent belief roots ??
										continue;
									}
								} 
								
								let parent_link = ParentLink { id: new_node_id, dist: neighbor_to_new_state_distance };
								rrttree.nodes[neighbor_id].parent = Some(parent_link);
							}
						}
					}
				}
				else { //if rrttree.nodes[new_node_id].node_type == BeliefNodeType::Observation {
					// Step 7: Create sibling in new belief in case of belief transition
					for child_belief_state in children_belief_states {
						let children_belief_state_id = rrttree.maybe_add_belief_state(&child_belief_state);
						let parent_link = ParentLink { id: new_node_id, dist: 0.0 };
						let new_node_id = rrttree.add_node(new_state, children_belief_state_id, BeliefNodeType::Action, Some(parent_link));
						kdtree.add(new_state, new_node_id);
					}
				}

				if goal(&new_state) {
					final_node_ids.push(new_node_id);

					let belief_state = &rrttree.belief_states[sampled_belief_id];
					println!("found leaf for belief:{}, {:?}", sampled_belief_id, &belief_state);
				}
			}
		}

		// paths to leafs
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
		//

		let belief_graph = BeliefGraph::from(&rrttree);
		let expected_costs_to_goal = conditional_dijkstra(&belief_graph, &final_node_ids, |a: &[f64; N], b: &[f64;N]| self.fns.cost_evaluator(a, b));
		let policy = extract_policy(&belief_graph, &expected_costs_to_goal);

		(rrttree, policy, best_paths)
	}
}

impl <const N: usize> From<&RRTTree<N>> for BeliefGraph<N> {
	#[allow(clippy::style)]
    fn from(rrttree: &RRTTree<N>) -> Self {
		let reachable_belief_states = vec![];

		let mut nodes: Vec<_> = rrttree.nodes.iter()
			.map(|n| BeliefNode{
				state: n.state,
				belief_state: rrttree.belief_states[n.belief_state_id].clone(),
				belief_id: n.belief_state_id,
				parents: n.parent.map(|p| vec![p.id] ).unwrap_or_else(|| vec![]),
				children: vec![],
				node_type: n.node_type,
			}).collect();

		for (id, n) in rrttree.nodes.iter().enumerate() {
			if let Some(parent) = n.parent {
				nodes[parent.id].children.push(id);
			}
		}

        Self { nodes, reachable_belief_states }
    }
}

#[cfg(test)]
mod tests {

use super::*;

#[test]
fn test_plan_on_map1() {
	let mut m = Map::open("data/map1.pgm", [-1.0, -1.0], [1.0, 1.0]);
	m.add_zones("data/map1_zone_ids.pgm", 0.2);

	fn goal(state: &[f64; 2]) -> bool {
		(state[0] - 0.5).abs() < 0.05 && (state[1] - 0.35).abs() < 0.05
	}	

	struct FakeSampler {
		index: usize,
	}

	/*
	impl SampleFuncs<2> for FakeSampler{
		fn sample_state(&mut self) -> [f64; 2] {
			const SAMPLES: &[[f64; 2]] = &[[0.0, 0.0]];
			SAMPLES[self.index]
		}
	
		fn sample_discrete(&mut self, _n: usize) -> usize {
			const SAMPLES: &[usize] = &[0];
			let sample = SAMPLES[self.index];
			self.index += 1;
			sample
		}
	}
	let mut samplers = FakeSampler{index: 0};
	*/

	let mut samplers = RRTDefaultSamplers {
		state_sampler: ContinuousSampler::new([-1.0, -1.0], [1.0, 1.0]),
		belief_state_sampler: DiscreteSampler::new(),
	};

	let mut rrt = RRT::new(
		&mut samplers,
		&m);
	let (rrttree, policy, _paths) = rrt.plan([0.5, -0.8], &vec![0.2, 0.8], goal, 0.05, 5.0, 30000);
	//assert!(!paths.is_empty(), "No path found!");

	for belief_id in 0..rrttree.belief_states.len() {
		let mut m = m.clone();
		m.resize(5);
		m.draw_tree(&rrttree, Some(belief_id));
		m.draw_policy(&policy);
		
		//for (belief_id, path) in &paths {
		//	m.draw_path(path, crate::map_io::colors::color_map(*belief_id));
		//}

		m.draw_zones_observability();
		m.save(&format!("results/test_rrt_on_map1_{}", belief_id));
	}
}
#[test]
fn test_plan_on_map() {
	let mut m = Map::open("data/map2.pgm", [-1.0, -1.0], [1.0, 1.0]);
	m.add_zones("data/map2_zone_ids.pgm", 0.2);

	fn goal(state: &[f64; 2]) -> bool {
		(state[0] - 0.5).abs() < 0.05 && (state[1] - 0.7).abs() < 0.05
	}	

	let mut samplers = RRTDefaultSamplers {
		state_sampler: ContinuousSampler::new([-1.0, -1.0], [1.0, 1.0]),
		belief_state_sampler: DiscreteSampler::new(),
	};

	let mut rrt = RRT::new(
		&mut samplers,
		&m);
	let (rrttree, policy, _paths) = rrt.plan([0.5, -0.8], &vec![0.1, 0.1, 0.1, 0.7], goal, 0.05, 5.0, 20000);
	//assert!(!paths.is_empty(), "No path found!");

	for belief_id in 0..rrttree.belief_states.len() {
		let mut m = m.clone();
		m.resize(5);
		m.draw_hit_zone(goal);
		m.draw_tree(&rrttree, Some(belief_id));
		m.draw_policy(&policy);
		m.draw_zones_observability();
		m.save(&format!("results/test_rrt_on_map2_{}", belief_id));
	}
	//for (belief_id, path) in &paths {
	//	m.draw_path(path, crate::map_io::colors::color_map(*belief_id));
	//}
	m.draw_policy(&policy);
	m.draw_zones_observability();
	m.save("results/test_rrt_on_map")
}

#[test]
fn test_plan_empty_space() {
	struct Funcs {}
	impl RRTFuncs<2> for Funcs {}

	fn goal(state: &[f64; 2]) -> bool {
		(state[0] - 0.9).abs() < 0.05 && (state[1] - 0.9).abs() < 0.05
	}	

	let mut samplers = RRTDefaultSamplers {
		state_sampler: ContinuousSampler::new([-1.0, -1.0], [1.0, 1.0]),
		belief_state_sampler: DiscreteSampler::new(),
	};

	let mut rrt = RRT::new(
		&mut samplers,
		&Funcs{});

	let (_rrttree, _policy, _best_paths) = rrt.plan([0.0, 0.0], &vec![1.0], goal, 0.1, 1.0, 1000);
	//assert!(!paths.is_empty(), "No path found!");
}

}
