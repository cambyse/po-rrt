use bitvec::vec;
use itertools::{all, enumerate, izip, merge, zip};

use crate::common::*;
use crate::nearest_neighbor::*;
use crate::belief_graph::*;
use crate::sample_space::*;
use crate::map_shelves_io::*;
use crate::prm::*;
use crate::pto_graph::PTOFuncs;
use bitvec::prelude::*;
use priority_queue::PriorityQueue;
use std::collections::BTreeMap;
use std::f64::consts::PI;
use ordered_float::*;
use std::collections::HashMap;
//use std::collections::HashSet;


fn normalize_belief(unnormalized_belief_state: &BeliefState) -> BeliefState {
	let sum = unnormalized_belief_state.iter().fold(0.0, |sum, p| sum + p);
	unnormalized_belief_state.iter().map(|p| p / sum).collect()
}

pub fn shuffled(vector: &Vec<usize>, sampler: &mut DiscreteSampler) -> Vec<usize> {
	let mut to_shuffle = vector.clone();
	let mut shuffled = vec![];
	shuffled.reserve(vector.len());

	while !to_shuffle.is_empty() {
		let index = sampler.sample(to_shuffle.len());
		shuffled.push(to_shuffle[index]);
		to_shuffle.swap_remove(index);
	}

	shuffled
}


struct Funcs<'a>  {
	m: &'a MapShelfDomain,
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

//#[derive(Clone)]
pub struct Mode<'a> {
	pub id: usize,
	pub incoming_transitions: Vec<usize>,
    pub outcoming_transitions: Vec<usize>,
	pub remaining_zones: Vec<usize>,
	// probabilities
	pub reaching_probability: f64,
	pub belief_state: BeliefState,
	// prm
	pub prm: PRM<'a, MapShelfDomain, 2>,
	// convenience hashes
	object_there_transition_hash_map: HashMap<usize, usize>,
	object_not_there_transition_hash_map: HashMap<usize, usize>
}

impl<'a> Mode<'a>{

	pub fn add_outcoming_transition(&mut self, target_zone: usize, transition_id: usize, observation: bool) {
		match observation {
			true => { self.object_there_transition_hash_map.insert(target_zone, transition_id) },
			_ =>  { self.object_not_there_transition_hash_map.insert(target_zone, transition_id) },
		};

		self.outcoming_transitions.push(transition_id);
	}

	pub fn add_incoming_transition(&mut self, transition_id: usize){
		self.incoming_transitions.push(transition_id);
	}

	pub fn get_outcoming_transition(&self, target_zone: usize, observation: bool) -> Option<usize> {
		match observation {
			true => {if self.object_there_transition_hash_map.contains_key(&target_zone) { Some(self.object_there_transition_hash_map[&target_zone]) } else {None}},
			_ =>  {if self.object_not_there_transition_hash_map.contains_key(&target_zone) { Some(self.object_not_there_transition_hash_map[&target_zone]) } else {None}},
		}
	}
}

pub struct ModeTransition {
	pub id: usize,
	pub observed_zone_id: usize,
	pub from_mode_id: usize,
	pub to_mode_id: usize,
	pub observation_transitions: Vec<[usize;2]>,
	pub observation: bool
}

pub struct ModeTree<'a> {
    pub modes: Vec<Mode<'a>>,
	pub mode_transitions: Vec<ModeTransition>,
	pub belief_states: Vec<BeliefState>,
	pub belief_hash_map: HashMap<usize, usize>, // belief hash, to belief index
	pub mode_hash_map: HashMap<usize, usize>,  // belief hash to mode index
}

impl<'a> ModeTree<'a> {
	pub fn new() -> Self {
		Self{
			modes: Vec::new(),
			mode_transitions: Vec::new(),
			belief_states: Vec::new(),
			belief_hash_map: HashMap::new(),
			mode_hash_map: HashMap::new()
		}
	}

	pub fn add_mode(&mut self,
					remaining_zones: &[usize],
					reaching_probability: f64,
					belief_state: &BeliefState,
					continuous_sampler: ContinuousSampler<2>,
					map_shelves_domain: &'a MapShelfDomain) -> usize {
		let id = self.modes.len();
		
		let v = Mode{
			id,
			incoming_transitions: Vec::new(),
			outcoming_transitions: Vec::new(),
			remaining_zones: remaining_zones.to_owned(),
			reaching_probability,
			belief_state: belief_state.clone(),
			prm: PRM::new(continuous_sampler, map_shelves_domain),
			object_there_transition_hash_map: HashMap::new(), // zone to transition index
			object_not_there_transition_hash_map: HashMap::new() // zone to transition index
		};

		let belief_hash = hash(&belief_state);
		self.mode_hash_map.insert(belief_hash, v.id);

		self.modes.push(v);
		id
	}

	pub fn add_transition(&mut self, observed_zone_id: usize, from_mode_id: usize, to_mode_id: usize, observation: bool) -> usize {
		let id = self.mode_transitions.len();

		let transition = ModeTransition{
			id,
			observed_zone_id,
			from_mode_id,
			to_mode_id,
			observation_transitions: Vec::new(),
			observation
		};

		self.mode_transitions.push(transition);		

		id
	}

	fn get_transitions(&mut self, mode_id: usize, target_zone_id: usize, continuous_sampler: &ContinuousSampler<2>, map_shelves_domain: &'a MapShelfDomain) ->Vec<usize> {
		let mode = &mut self.modes[mode_id];

		let mut successor_modes = Vec::new();

		match mode.get_outcoming_transition(target_zone_id, true) {
			Some(id) => successor_modes.push(id),
			_ => {
				// create new mode 
				let mut succ_belief_state = vec![0.0; mode.belief_state.len()];
				succ_belief_state[target_zone_id] = 1.0; 
				succ_belief_state = normalize_belief(&succ_belief_state);

				let succ_belief_hash = hash(&succ_belief_state);
				let successor_mode_id = match self.mode_hash_map.get(&succ_belief_hash) {
					Some(id) => *id,
					None => {
						let succ_reaching_probability = mode.reaching_probability * transition_probability(&mode.belief_state, &succ_belief_state);
						let mut remaining_zones = mode.remaining_zones.clone();
						remaining_zones.retain(|zone_id| *zone_id != target_zone_id);
						self.add_mode(&remaining_zones, succ_reaching_probability, &succ_belief_state, continuous_sampler.clone(), map_shelves_domain)
					} 
				};
				
				// create new transition
				let transition_index = self.add_transition(target_zone_id, mode_id, successor_mode_id, true);
				
				// update the mode
				self.modes[mode_id].add_outcoming_transition(target_zone_id, transition_index, true);

				successor_modes.push(transition_index);
			}
		};

		let mode = &self.modes[mode_id];

		// case object not there
		match mode.get_outcoming_transition(target_zone_id, false) {
			Some(id) => successor_modes.push(id),
			_ => {
				// create new mode 
				let mut succ_belief_state = mode.belief_state.clone();
				succ_belief_state[target_zone_id] = 0.0; 

				if succ_belief_state.iter().sum::<f64>() > 0.0 {
				
					succ_belief_state = normalize_belief(&succ_belief_state);

					let mut remaining_zones = mode.remaining_zones.clone();
					remaining_zones.retain(|zone_id| *zone_id != target_zone_id);

					let succ_belief_hash = hash(&succ_belief_state);
					let successor_mode_id = match self.mode_hash_map.get(&succ_belief_hash) {
						Some(id) => *id,
						None => {
							let succ_reaching_probability = mode.reaching_probability * transition_probability(&mode.belief_state, &succ_belief_state);
							let mut remaining_zones = mode.remaining_zones.clone();
							remaining_zones.retain(|zone_id| *zone_id != target_zone_id);
							self.add_mode(&remaining_zones, succ_reaching_probability, &succ_belief_state, continuous_sampler.clone(), map_shelves_domain)
						} 
					};

					// create new transition
					let transition_index = self.add_transition(target_zone_id, mode_id, successor_mode_id, true);
					
					// update the mode
					self.modes[mode_id].add_outcoming_transition(target_zone_id, transition_index, false);

					successor_modes.push(transition_index);
				}
			}
		};

		successor_modes
	}
}

pub struct MapShelfDomainTampPRM<'a> {
	continuous_sampler: ContinuousSampler<2>,
	discrete_sampler: DiscreteSampler,
	pub map_shelves_domain: &'a MapShelfDomain,
	pub zone_sampler: ContinuousSampler<2>,
	pub n_worlds: usize,
	pub goal_radius: f64,
	n_it: usize,
	// mode tree
	pub search_tree: ModeTree<'a>,
	pub belief_graph: BeliefGraph<2>
}

impl<'a> MapShelfDomainTampPRM<'a> {
	pub fn new(continuous_sampler: ContinuousSampler<2>, discrete_sampler: DiscreteSampler, map_shelves_domain: &'a MapShelfDomain, goal_radius: f64) -> Self {
		Self { continuous_sampler,
			   discrete_sampler,
			   map_shelves_domain, 
			   zone_sampler: ContinuousSampler::new([0.0, 0.0], [map_shelves_domain.visibility_distance, 2.0 * PI  as f64 ]),
			   n_worlds: map_shelves_domain.n_zones(), 
			   goal_radius,
			   n_it: 0,
			   search_tree: ModeTree::new(),
			   belief_graph: BeliefGraph::new(Vec::new(), Vec::new())}
	}

	pub fn plan(&mut self, start: &[f64; 2], initial_belief_state: &BeliefState, max_step: f64, search_radius: f64, n_iter_max: usize) -> Result<Policy<2>, &'static str> {
		self.grow_mm_prm(start, initial_belief_state, max_step, search_radius, n_iter_max);

		//self.build_belief_graph();

		Ok(Policy{
			expected_costs: 0.0,
			leafs: vec![],
			nodes: vec![]
		})
	}	

	fn grow_mm_prm(&mut self, &_start: &[f64; 2], initial_belief_state: &BeliefState, max_step: f64, search_radius: f64, n_iter_max: usize) {
		check_belief_state(initial_belief_state);

		let belief_states = self.map_shelves_domain.reachable_belief_states(initial_belief_state);
		let belief_hash_map = create_belief_states_hash_map(&belief_states);
		
		self.search_tree.belief_states = belief_states;
		self.search_tree.belief_hash_map = belief_hash_map;


		let remaining_zones: Vec<usize> = (0..self.map_shelves_domain.n_zones()).collect();
		self.search_tree.add_mode(&remaining_zones, 1.0,&initial_belief_state.clone(), self.continuous_sampler.clone(), self.map_shelves_domain);

		let mut i = 0;
		while i < n_iter_max {
			i +=1;

			let mode_id = self.discrete_sampler.sample(self.search_tree.modes.len());
			let mode = &mut self.search_tree.modes[mode_id];

			println!("sampled mode:{}, of belief:{:?}", mode_id, &mode.belief_state);

			// grow the prm
			mode.prm.grow_graph(max_step, search_radius, 200);

			// sample transitions nodes
			for _j in 0..10 {
				let mode = &self.search_tree.modes[mode_id];

				if !mode.remaining_zones.is_empty() {
					let target_zone_index = self.discrete_sampler.sample(mode.remaining_zones.len());
					let target_zone_id = self.search_tree.modes[mode_id].remaining_zones[target_zone_index];
	
					let transition_ids = self.search_tree.get_transitions(mode_id, target_zone_id, &self.continuous_sampler, &self.map_shelves_domain);
					
					// sample transition
					let transition_sample = self.sample_observation_of_zone(target_zone_id);

					// add sample to the respective prms
					let mode = &mut self.search_tree.modes[mode_id];
					let observation_node_id = mode.prm.add_sample(transition_sample, max_step, search_radius);
					for transition_id in transition_ids {
						let transition = &mut self.search_tree.mode_transitions[transition_id];
						let destination_mode = &mut self.search_tree.modes[transition.to_mode_id];
						let destination_observation_node_id = destination_mode.prm.add_sample(transition_sample, max_step, search_radius);

						transition.observation_transitions.push([observation_node_id, destination_observation_node_id]);
					}
				}
			}
		}
	}

	fn sample_observation_of_zone(&mut self, target_zone_id: usize) -> [f64; 2] {
		let zone_position = self.map_shelves_domain.get_zone_positions()[target_zone_id];

		let [radius, angle] = self.zone_sampler.sample();
		//let radius = 0.5;
		let x = (zone_position[0] + radius * angle.cos()).clamp(self.continuous_sampler.low[0], self.continuous_sampler.up[0]-0.0001);
		let y = (zone_position[1] + radius * angle.sin()).clamp(self.continuous_sampler.low[1], self.continuous_sampler.up[1]-0.0001);

		[x, y]
	}

	/*
	fn shortcut(&self, path: &Vec<[f64; 2]>) -> Vec<[f64; 2]> {
		let mut short_cut_path = path.clone();

		if short_cut_path.len() <= 2 {
			return short_cut_path;
		}

		let mut sampler = DiscreteSampler::new();

		fn interpolate(a: f64, b: f64, lambda: f64) -> f64 {
			a * (1.0 - lambda) + b * lambda
		}

		let checker = Funcs{m:&self.map_shelves_domain};

		let joint_dim = 2;
		for _i in 0..100 {
			let joint = sampler.sample(joint_dim);
			let interval_start = sampler.sample(short_cut_path.len() - 2);
			let interval_end = interval_start + 2 + sampler.sample(short_cut_path.len() - interval_start - 2);

			assert!(interval_end < short_cut_path.len());
			assert!(interval_end - interval_start >= 2);

			let interval_start_state = &short_cut_path[interval_start];
			let interval_end_state = &short_cut_path[interval_end];

			// create shortcut states (interpolated on a particular joint)
			let mut shortcut_states = vec![];
			shortcut_states.reserve(interval_end - interval_start);
			for j in interval_start..interval_end {
				let lambda = (j - interval_start) as f64 / (interval_end - interval_start) as f64;
				let mut shortcut_state = short_cut_path[j];
				shortcut_state[joint] = interpolate(interval_start_state[joint], interval_end_state[joint], lambda);
				shortcut_states.push(shortcut_state);
			}

			// check validities
			let mut should_commit = true;
			for (from, to) in pairwise_iter(&shortcut_states) {
				should_commit = should_commit && checker.transition_validator(from, to); // TODO: can be optimized to avoid rechecking 2 times the nodes
			}

			// commit if valid
			if should_commit {
				for j in interval_start..interval_end {
					short_cut_path[j] = shortcut_states[j - interval_start];
				}
			}
		}

		short_cut_path
	}

	fn build_policy(&self, leaf: &SearchNode, tree: &SearchTree) -> Policy<2> {
		let node_path_to_last_leaf = self.get_search_nodes_to_last_leaf(leaf, tree);

		let mut policy = Policy {
			leafs: vec![],
			nodes: vec![],
			expected_costs: 0.0
		};

		let mut last_observation_node_id = 0;
		for search_node in &node_path_to_last_leaf {
			let mut previous_node_id = last_observation_node_id;

			// observation path
			let path_to_observation = self.shortcut(&search_node.path_to_observation);
			for state in &path_to_observation {
				let node_id = policy.add_node(state, &search_node.belief_state, 0, false);
				if node_id != previous_node_id {
					policy.add_edge(previous_node_id, node_id);
				}
				previous_node_id = node_id;
			}
			last_observation_node_id = previous_node_id;

			// pick-up path
			let path_to_pickup = self.shortcut(&search_node.path_to_pickup);
			for (i, state) in path_to_pickup.iter().enumerate() {
				let is_leaf = i == search_node.path_to_pickup.len() - 1;
				let mut belief_state = search_node.belief_state.clone();
				for (i, p) in belief_state.iter_mut().enumerate() {
					if i != search_node.target_zone_id.unwrap() {
						*p = 0.0;
					}
				}
				belief_state = normalize_belief(&belief_state);
				let node_id = policy.add_node(state, &belief_state, 0, is_leaf);
				if node_id != previous_node_id {
					policy.add_edge(previous_node_id, node_id);
				}
				previous_node_id = node_id;
			}
		}

		policy.compute_expected_costs_to_goals(&|a: &[f64;2], b: &[f64;2]| Funcs{m:&self.map_shelves_domain}.cost_evaluator(a, b));

		policy
	}

	fn reconstruct_path_tree(&self, leaf: &SearchNode, tree: &SearchTree) -> Vec<Vec<[f64; 2]>> {
		assert!(leaf.remaining_zones.is_empty());

		let node_path_to_last_leaf = self.get_search_nodes_to_last_leaf(leaf, tree);

		println!("number of nodes:{}", node_path_to_last_leaf.len());

		// gather path tree
		let mut path_tree: Vec<Vec<[f64; 2]>> = vec![];

		path_tree.push(vec![]);
		for current in &node_path_to_last_leaf {
			for p in &current.path_to_observation {
				path_tree[0].push(*p);
			}

			path_tree.push(current.path_to_pickup.clone());
		}

		path_tree
	}

	fn get_search_nodes_to_last_leaf(&self, leaf: &SearchNode, tree: &SearchTree) ->  Vec<SearchNode> {
		let mut node_path_to_last_leaf: Vec<SearchNode> = vec![];

		node_path_to_last_leaf.push(leaf.clone());

		let mut current = leaf;
		while let Some(parent_id) = current.parent {
			let parent = &tree.nodes[parent_id];
			node_path_to_last_leaf.push(parent.clone());

			current = parent;
		}
		node_path_to_last_leaf = node_path_to_last_leaf.into_iter().rev().collect();

		node_path_to_last_leaf
	}*/
}

#[cfg(test)]
mod tests {

use super::*;


#[test]
fn test_plan_on_map2_prm() {
	let mut m = MapShelfDomain::open("data/map1_2_goals.pgm", [-1.0, -1.0], [1.0, 1.0]);
	m.add_zones("data/map1_2_goals_zone_ids.pgm", 0.5);

	let mut tamp_prm = MapShelfDomainTampPRM::new(ContinuousSampler::new([-1.0, -1.0], [1.0, 1.0]), DiscreteSampler::new(), &m, 0.05);		
	
	let initial_belief_state = vec![1.0/2.0; 2];
	let policy = tamp_prm.plan(&[-0.9, 0.0], &initial_belief_state, 0.1, 2.0, 10);
	let policy = policy.expect("nopath tree found!");

	let mut m2 = m.clone();
	m2.resize(5);
	m2.draw_zones_observability();
	
	for mode in &tamp_prm.search_tree.modes {
		m2.draw_full_graph(&mode.prm.graph);
		break;
	}
	
	m2.draw_policy(&policy);
	m2.save("results/test_map1_2_goals_tamp_prm");
}

#[test]
fn test_plan_on_map7_plan_prm() {
	let mut m = MapShelfDomain::open("data/map7.pgm", [-1.0, -1.0], [1.0, 1.0]);
	m.add_zones("data/map7_6_goals_zone_ids.pgm", 0.5);

	let mut tamp_prm = MapShelfDomainTampPRM::new(ContinuousSampler::new([-1.0, -1.0], [1.0, 1.0]), DiscreteSampler::new(), &m, 0.05);	
	
	let initial_belief_state = vec![1.0/6.0; 6];
	let policy = tamp_prm.plan(&[0.0, -0.8], &initial_belief_state, 0.1, 2.0, 150);
	let policy = policy.expect("nopath tree found!");

	let mut m2 = m.clone();
	m2.resize(5);
	m2.draw_zones_observability();
	m2.draw_policy(&policy);
	m2.save("results/map7/test_map7_tamp_prm");
}

#[test]
fn test_shuffle() {
	shuffled(&vec![0, 1, 2, 3, 4], &mut DiscreteSampler::new());
}

}