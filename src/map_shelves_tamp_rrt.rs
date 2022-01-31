use itertools::{all, enumerate, izip, merge, zip};

use crate::common::*;
use crate::nearest_neighbor::*;
use crate::belief_graph::*;
use crate::sample_space::*;
use crate::map_shelves_io::*;
use crate::rrt::*;
use bitvec::prelude::*;
use priority_queue::PriorityQueue;
use std::collections::BTreeMap;
use ordered_float::*;

fn normalize_belief(unnormalized_belief_state: &BeliefState) -> BeliefState {
	let sum = unnormalized_belief_state.iter().fold(0.0, |sum, p| sum + p);
	unnormalized_belief_state.iter().map(|p| p / sum).collect()
}

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

#[derive(Clone)]
pub struct SearchNode {
	pub id: usize,
    pub target_zone_id: Option<usize>,
	pub parent: Option<usize>,
    pub children: Vec<usize>,
	pub remaining_zones: Vec<usize>,
	// start and states
	pub start_state: [f64; 2],
	pub observation_state: [f64; 2],
	pub pickup_state: [f64; 2],
	// resulting paths
	pub path_to_observation: Vec<[f64;2]>,
	pub path_to_pickup: Vec<[f64;2]>,
	// costs
	pub path_to_start_cost: f64,
	pub path_to_observation_cost: f64,
	pub path_to_pickup_cost: f64,
	// probabilities
	pub reaching_probability: f64,
	pub belief_state: BeliefState,
	pub expected_cost: f64
}

pub struct SearchTree {
    pub nodes: Vec<SearchNode>,
}

impl SearchTree {
	pub fn add_node(&mut self, parent_id: usize, target_zone_id: usize, remaining_zones: &[usize],
					start_state: [f64; 2], observation_state: [f64; 2], pickup_state: [f64; 2],
				    path_to_observation: Vec<[f64;2]>, path_to_pickup: Vec<[f64;2]>,
					path_to_start_cost: f64, path_to_observation_cost:f64, path_to_pickup_cost: f64,
					reaching_probability: f64,
					belief_state: &BeliefState,
					expected_cost: f64) -> usize {
		let id = self.nodes.len();
		let v = SearchNode{
			id,
			target_zone_id: Some(target_zone_id),
			parent: Some(parent_id),
			children: Vec::new(),
			remaining_zones: remaining_zones.to_owned(),
			start_state,
			observation_state,
			pickup_state,
			path_to_observation,
			path_to_pickup,
			path_to_start_cost,
			path_to_observation_cost,
			path_to_pickup_cost,
			reaching_probability,
			belief_state: belief_state.clone(),
			expected_cost
		};

		self.nodes.push(v);
		id
	}
}

pub struct MapShelfDomainTampRRT<'a> {
	continuous_sampler: ContinuousSampler<2>,
	pub map_shelves_domain: &'a MapShelfDomain,
	pub kdtree: KdTree<2>,
	pub n_worlds: usize,
	n_it: usize,
}

impl<'a> MapShelfDomainTampRRT<'a> {
	pub fn new(continuous_sampler: ContinuousSampler<2>, map_shelves_domain: &'a MapShelfDomain) -> Self {
		Self { continuous_sampler,
			   map_shelves_domain, 
			   kdtree: KdTree::new([0.0; 2]),
			   n_worlds: map_shelves_domain.n_zones(), 
			   n_it: 0}
	}

	pub fn plan(&mut self, &start: &[f64; 2], initial_belief_state: &BeliefState, max_step: f64, search_radius: f64, n_iter: usize) -> Result<Vec<Vec<[f64; 2]>>, &'static str> {
		let mut solution_nodes = BTreeMap::new();
		let mut q = PriorityQueue::new();

		let root_node = SearchNode{
			id: 0,
			target_zone_id: None,
			parent: None,
			children: Vec::new(),
			remaining_zones: (0..self.map_shelves_domain.n_zones()).collect(),
			start_state: start,
			observation_state: start,
			pickup_state: start,
			path_to_observation: vec![],
			path_to_pickup: vec![],
			path_to_start_cost: 0.0,
			path_to_observation_cost: 0.0,
			path_to_pickup_cost: 0.0,
			reaching_probability: 1.0,
			belief_state: initial_belief_state.clone(),
			expected_cost: 0.0
		};

		q.push(root_node.id, Priority{prio: root_node.path_to_start_cost});

		let mut search_tree = SearchTree{
			nodes: vec![root_node]
		};

		// rrt 
		let mut rrt = RRT::new(ContinuousSampler::new([-1.0, -1.0], [1.0, 1.0]), Funcs{m:&self.map_shelves_domain});

		let mut best_expected_cost = std::f64::INFINITY;
		let mut it = 0;
		while !q.is_empty() {
			it+=1;

			println!("iteration:{}", it);

			let (u_id, _) = q.pop().unwrap();
			let u = search_tree.nodes[u_id].clone();

			for target_zone_id in &u.remaining_zones {
				let mut remaining_zones = u.remaining_zones.clone();
				remaining_zones.retain(|zone_id| zone_id != target_zone_id);
				
				// compute belief state and reaching probability
				let mut v_belief_state = u.belief_state.clone();
				if let Some(u_target_zone_id) = u.target_zone_id {
					v_belief_state[u_target_zone_id] = 0.0; // we are at thisstage on theskeleton if the object was not there
				}
				v_belief_state = normalize_belief(&v_belief_state);
				let reaching_probability = u.reaching_probability * transition_probability(&u.belief_state, &v_belief_state);

				/// Query motion planner
				// piece 1: go to target observe zone
				let observation_goal = ObservationGoal{m: &self.map_shelves_domain, zone_id: *target_zone_id};
				
				let (observation_planning_result, _) = rrt.plan(u.observation_state, &observation_goal, max_step, search_radius, n_iter);
				let (observation_path, observation_path_cost) = observation_planning_result.expect("no observation path found!");
				let v_observation_state = observation_path.last().unwrap().clone();

				// piece 2: object is here: plan to reach goal corresponding to 
				let zone_position = self.map_shelves_domain.get_zone_positions()[*target_zone_id];
				let pickup_goal = SquareGoal::new(vec![(zone_position, bitvec![1])], 0.05);

				let (pickup_planning_result, _) = rrt.plan(v_observation_state, &pickup_goal, max_step, search_radius, n_iter);
				let (pickup_path, pickup_path_cost) = pickup_planning_result.expect("no pickup path found!");
				let v_pickup_state = pickup_path.last().unwrap().clone();

				// compute expected cost to start
				let pickup_probability = v_belief_state[*target_zone_id];
				let expected_cost = u.expected_cost + reaching_probability * (observation_path_cost + pickup_probability * pickup_path_cost);

				// create next node
				let v_id = search_tree.add_node(
					u.id,
					*target_zone_id,
					&remaining_zones,
					// states
					u.observation_state,
					v_observation_state,
					v_pickup_state,
					// paths
					observation_path,
					pickup_path,
					// costs
					u.path_to_observation_cost,
					observation_path_cost,
					pickup_path_cost,
					// probabilities
					reaching_probability,
					&v_belief_state,
					expected_cost
				);

				let v = &search_tree.nodes[v_id];

				// addonly if not too bad
				if v.expected_cost < best_expected_cost {
					q.push(v_id, Priority{prio: expected_cost});
				}
				else {
					println!("prune!");
				}
			}

			// save leaf node
			if u.remaining_zones.is_empty() {
				if u.expected_cost < best_expected_cost {
					best_expected_cost = u.expected_cost;
				}
				solution_nodes.insert(OrderedFloat(u.expected_cost), u);
			}
		}

		println!("it:{}, nodes:{}", it, search_tree.nodes.len());
		println!("n leafs:{}", search_tree.nodes.iter().filter(|&n| n.remaining_zones.is_empty() ).count());

		let (best_expected_cost, best_final_node) = solution_nodes.last_key_value().expect("No solution found!");
		let path_tree = self.reconstruct_path_tree(&best_final_node, &search_tree);

		println!("expected costs:{}", best_expected_cost);
		
		Ok(path_tree)
	}

	pub fn reconstruct_path_tree(&self, leaf: &SearchNode, tree: &SearchTree) -> Vec<Vec<[f64; 2]>> {
		assert!(leaf.remaining_zones.is_empty());

		// get node path to last leaf
		let mut node_path_to_last_leaf: Vec<SearchNode> = vec![];

		node_path_to_last_leaf.push(leaf.clone());

		let mut current = leaf;
		while let Some(parent_id) = current.parent {
			let parent = &tree.nodes[parent_id];
			node_path_to_last_leaf.push(parent.clone());

			current = parent;
		}
		node_path_to_last_leaf = node_path_to_last_leaf.into_iter().rev().collect();

		println!("number of nodes:{}", node_path_to_last_leaf.len());

		// gather path tree
		let mut path_tree: Vec<Vec<[f64; 2]>> = vec![];

		path_tree.push(vec![]);
		for current in &node_path_to_last_leaf {
			for p in &current.path_to_observation {
				path_tree[0].push(p.clone());
			}

			path_tree.push(current.path_to_pickup.clone());
		}

		path_tree
	}
}

#[cfg(test)]
mod tests {

use super::*;

#[test]
fn test_plan_on_map2_pomdp() {
	let mut m = MapShelfDomain::open("data/map1_2_goals.pgm", [-1.0, -1.0], [1.0, 1.0]);
	m.add_zones("data/map1_2_goals_zone_ids.pgm", 0.5);

	let mut tamp_rrt = MapShelfDomainTampRRT::new(ContinuousSampler::new([-1.0, -1.0], [1.0, 1.0]), &m);		
	
	let initial_belief_state = vec![1.0/2.0; 2];
	let path_tree = tamp_rrt.plan(&[0.0, -0.8], &initial_belief_state, 0.1, 2.0, 2500);
	let paths = path_tree.expect("nopath tree found!");

	let mut m2 = m.clone();
	m2.resize(5);
	m2.draw_zones_observability();
	for path in paths {
		m2.draw_path(path.as_slice(), colors::BLACK);
	}
	m2.save("results/test_map1_2_goals_tamp_rrt");
}

#[test]
fn test_plan_on_map7() {
	let mut m = MapShelfDomain::open("data/map7.pgm", [-1.0, -1.0], [1.0, 1.0]);
	m.add_zones("data/map7_6_goals_zone_ids.pgm", 0.5);

	let mut tamp_rrt = MapShelfDomainTampRRT::new(ContinuousSampler::new([-1.0, -1.0], [1.0, 1.0]), &m);	
	
	let initial_belief_state = vec![1.0/6.0; 6];
	let path_tree = tamp_rrt.plan(&[0.0, -0.8], &initial_belief_state, 0.1, 2.0, 2500);
	let paths = path_tree.expect("nopath tree found!");

	let mut m2 = m.clone();
	m2.resize(5);
	m2.draw_zones_observability();
	for path in paths {
		m2.draw_path(path.as_slice(), colors::BLACK);
	}
	m2.save("results/map7/test_map7_tamp_rrt");
}

}
