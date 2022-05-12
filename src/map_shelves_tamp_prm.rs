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
use std::borrow::BorrowMut;
use std::collections::BTreeMap;
use std::f64::consts::PI;
use ordered_float::*;
use std::collections::HashMap;
//use std::collections::HashSet;

fn is_final(belief_state: &BeliefState) -> bool {
	belief_state.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap() >= &0.999
}

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
	pub final_node_ids: Vec<usize>,
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
	pub observation: bool,
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
			final_node_ids: vec![],
			object_there_transition_hash_map: HashMap::new(),    // zone to transition index
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

		if is_final(&mode.belief_state) {
			return Vec::new();
		}

		// case object not there
		match mode.get_outcoming_transition(target_zone_id, true) {
			Some(id) => successor_modes.push(id),
			_ => {
				// create new mode corresponding to object found
				let mut succ_belief_state = vec![0.0; mode.belief_state.len()];
				succ_belief_state[target_zone_id] = 1.0; 
				succ_belief_state = normalize_belief(&succ_belief_state);
				let succ_reaching_probability = mode.reaching_probability * transition_probability(&mode.belief_state, &succ_belief_state);
				//
				//println!("transition p when target found: {}, bs {:?} -> {:?}", succ_reaching_probability, &mode.belief_state, &succ_belief_state);
				//
				let succ_belief_hash = hash(&succ_belief_state);
				let successor_mode_id = match self.mode_hash_map.get(&succ_belief_hash) {
					Some(id) => *id,
					None => {
						let mut remaining_zones = mode.remaining_zones.clone();
						remaining_zones.retain(|zone_id| *zone_id != target_zone_id);
						//let sampler = ContinuousSampler::new_with_seed(continuous_sampler.low, continuous_sampler.up, (self.modes.len() + 1) as u64);
						let mode_id = self.add_mode(&remaining_zones, succ_reaching_probability, &succ_belief_state, continuous_sampler.clone(), map_shelves_domain);

						// add initial goal state
						let mode = &mut self.modes[mode_id];
						let goal_id = mode.prm.add_sample(map_shelves_domain.get_zone_positions()[target_zone_id], 0.0, 0.0);
						mode.final_node_ids.push(goal_id);

						mode_id
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
				let succ_reaching_probability = mode.reaching_probability * transition_probability(&mode.belief_state, &succ_belief_state);

				assert!(succ_belief_state.iter().sum::<f64>() > 0.0);

				//println!("transition p when target not found:{}", succ_reaching_probability);

				succ_belief_state = normalize_belief(&succ_belief_state);

				let mut remaining_zones = mode.remaining_zones.clone();
				remaining_zones.retain(|zone_id| *zone_id != target_zone_id);

				let succ_belief_hash = hash(&succ_belief_state);
				let successor_mode_id = match self.mode_hash_map.get(&succ_belief_hash) {
					Some(id) => *id,
					None => {
						let mut remaining_zones = mode.remaining_zones.clone();
						remaining_zones.retain(|zone_id| *zone_id != target_zone_id);
						//let sampler = ContinuousSampler::new_with_seed(continuous_sampler.low, continuous_sampler.up, (self.modes.len() + 1) as u64);
						let mode_id = self.add_mode(&remaining_zones, succ_reaching_probability, &succ_belief_state, continuous_sampler.clone(), map_shelves_domain);

						if let Some(goal_zone_id) = succ_belief_state.iter().position(|&p| p == 1.0) {
							// add initial goal state
							let mode = &mut self.modes[mode_id];
							let goal_id = mode.prm.add_sample(map_shelves_domain.get_zone_positions()[goal_zone_id], 0.0, 0.0);
							mode.final_node_ids.push(goal_id);
						}

						mode_id
					} 
				};

				// create new transition
				let transition_index = self.add_transition(target_zone_id, mode_id, successor_mode_id, true);
				
				// update the mode
				self.modes[mode_id].add_outcoming_transition(target_zone_id, transition_index, false);

				successor_modes.push(transition_index);
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
	pub belief_graph: BeliefGraph<2>,
	expected_costs_to_goals: Vec<f64>,
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
			   belief_graph: BeliefGraph::new(Vec::new(), Vec::new()),
			   expected_costs_to_goals: Vec::new()
			}
	}

	pub fn plan(&mut self, start: &[f64; 2], initial_belief_state: &BeliefState, max_step: f64, search_radius: f64, n_iter_per_belief: usize) -> Result<Policy<2>, &'static str> {
		self.grow_mm_prm(start, initial_belief_state, max_step, search_radius, n_iter_per_belief);

		println!("build belief graph..");

		let final_belief_node_ids = self.build_belief_graph();

		println!("dynamic programming..");

		self.compute_expected_costs_to_goals(&final_belief_node_ids);

		println!("extract policy..");

		let policy = self.extract_policy();

		Ok(policy)
	}	

	fn grow_mm_prm(&mut self, &start: &[f64; 2], initial_belief_state: &BeliefState, max_step: f64, search_radius: f64, n_iter_per_belief: usize) {
		check_belief_state(initial_belief_state);

		let belief_states = self.map_shelves_domain.reachable_belief_states(initial_belief_state);
		let belief_hash_map = create_belief_states_hash_map(&belief_states);
		
		self.search_tree.belief_states = belief_states;
		self.search_tree.belief_hash_map = belief_hash_map;

		let remaining_zones: Vec<usize> = (0..self.map_shelves_domain.n_zones()).collect();
		self.search_tree.add_mode(&remaining_zones, 1.0, &initial_belief_state.clone(), self.continuous_sampler.clone(), self.map_shelves_domain);
		self.search_tree.modes[0].prm.add_sample(start, 0.0, 0.0); // planning start!

		let total_number_of_samples = n_iter_per_belief * self.search_tree.belief_states.len();
		let batch_size = 200;
		let transition_sample_per_batch = 10;
		let n_outer_loop = (total_number_of_samples as f64 / batch_size as f64) as usize;
		let n_sample_within_mode = batch_size - transition_sample_per_batch;

		let mut i = 0;
		while i < n_outer_loop {
			i +=1;

			let mode_id = self.discrete_sampler.sample(self.search_tree.modes.len());
			let mode = &mut self.search_tree.modes[mode_id];

			//println!("sampled mode:{}, of belief:{:?}", mode_id, &mode.belief_state);

			// grow the prm
			mode.prm.grow_graph(max_step, search_radius, n_sample_within_mode);

			// sample transitions nodes
			for _j in 0..transition_sample_per_batch {
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

						// debug
						//let bs_from = &self.search_tree.modes[transition.from_mode_id].belief_state;
						//let bs_to = &self.search_tree.modes[transition.to_mode_id].belief_state;
						//assert_ne!(transition.to_mode_id, transition.from_mode_id);
						//assert!(transition_probability(bs_from, bs_to) > 0.00001);
						// 

						let destination_mode = &mut self.search_tree.modes[transition.to_mode_id];
						let destination_observation_node_id = destination_mode.prm.add_sample(transition_sample, max_step, search_radius);

						transition.observation_transitions.push([observation_node_id, destination_observation_node_id]);
					}
				}
			}
		}
	}

	fn build_belief_graph(&mut self) -> Vec<usize> {
		self.belief_graph = BeliefGraph::new(Vec::new(), self.search_tree.belief_states.clone());
		
		let mut final_belief_node_ids = Vec::new();
		let mut node_mapping_per_mode = Vec::<HashMap<usize, usize>>::new();
		node_mapping_per_mode.reserve(self.search_tree.belief_states.len()); // prm node -> belief node id
		let mut belief_node_to_mode_and_prm_node = HashMap::<usize, (usize, usize)>::new();


		// add nodes
		for mode in &self.search_tree.modes {
			let belief_id = self.belief_graph.belief_states_to_id[&hash(&mode.belief_state)];

			let mut node_to_belief_node_id = HashMap::new();

			// add nodes
			for (node_id, node) in mode.prm.graph.nodes.iter().enumerate() {
				let belief_node_id = self.belief_graph.add_node(node.state, mode.belief_state.clone(), belief_id, BeliefNodeType::Action); // set action before overriding it
				node_to_belief_node_id.insert(node_id, belief_node_id);
				belief_node_to_mode_and_prm_node.insert(belief_node_id, (mode.id, node_id));
			}

			// convert final belief nodes
			for final_node_id in &mode.final_node_ids {
				final_belief_node_ids.push(node_to_belief_node_id[&final_node_id]);
			}

			node_mapping_per_mode.push(node_to_belief_node_id);
		}

		// add observation edges
		for transition in &self.search_tree.mode_transitions {
			for transition_edge in &transition.observation_transitions {
				let from_node_id = transition_edge[0];
				let to_node_id = transition_edge[1];

				let from_belief_node_id = node_mapping_per_mode[transition.from_mode_id][&from_node_id];
				let to_belief_node_id = node_mapping_per_mode[transition.to_mode_id][&to_node_id];

				// debug
				//let bs_from = &self.belief_graph.nodes[from_belief_node_id].belief_state;
				//let bs_to = &self.belief_graph.nodes[to_belief_node_id].belief_state;
				//assert!(transition_probability(bs_from, bs_to) > 0.0);
				// debug

				self.belief_graph.add_edge(from_belief_node_id, to_belief_node_id);
				self.belief_graph.nodes[from_belief_node_id].node_type = BeliefNodeType::Observation;
			}
		}

		// add action edges
		for mode in &self.search_tree.modes {
			// add action edges
			for (node_id, node) in mode.prm.graph.nodes.iter().enumerate() {
				let belief_node_id = node_mapping_per_mode[mode.id][&node_id];
				
				if self.belief_graph.nodes[belief_node_id].node_type == BeliefNodeType::Observation {
					continue;
				}	

				for children in &node.children {
					let children_belief_node_id = node_mapping_per_mode[mode.id][&children.id];

					// debug
					//assert_ne!(belief_node_id, children_belief_node_id);
					//let from_state = &self.belief_graph.nodes[belief_node_id].state;
					//let to_state = &self.belief_graph.nodes[children_belief_node_id].state;
					//assert!(norm2(from_state, to_state) > 0.000001);
					//assert_eq!(&self.belief_graph.nodes[belief_node_id].belief_id, &self.belief_graph.nodes[children_belief_node_id].belief_id);
					// debug

					self.belief_graph.add_edge(belief_node_id, children_belief_node_id);
				}	
			}
		}
		

		final_belief_node_ids
	}

	pub fn compute_expected_costs_to_goals(&mut self, final_belief_node_ids: &Vec<usize>) {
		self.expected_costs_to_goals = conditional_dijkstra(&self.belief_graph, &final_belief_node_ids, &|a: &[f64; 2], b: &[f64;2]| self.map_shelves_domain.cost_evaluator(a, b));
	}

	pub fn extract_policy(&self) -> Policy<2> {
		//let mut policy = extract_policy(&self.belief_graph, &self.expected_costs_to_goals, &|a: &[f64; N], b: &[f64;N]| self.fns.cost_evaluator(a, b) );
		//policy.compute_expected_costs_to_goals(&|a: &[f64; N], b: &[f64;N]| self.fns.cost_evaluator(a, b));
		//println!("COST:{}", policy.expected_costs);

		extract_policy(&self.belief_graph, &self.expected_costs_to_goals, &|a: &[f64; 2], b: &[f64; 2]| self.map_shelves_domain.cost_evaluator(a, b) )
	}

	fn sample_observation_of_zone(&mut self, target_zone_id: usize) -> [f64; 2] {
		let zone_position = self.map_shelves_domain.get_zone_positions()[target_zone_id];

		let [radius, angle] = self.zone_sampler.sample();
		let radius = radius * (radius / self.map_shelves_domain.visibility_distance).sqrt();
		let x = (zone_position[0] + radius * angle.cos()).clamp(self.continuous_sampler.low[0], self.continuous_sampler.up[0]-0.0001);
		let y = (zone_position[1] + radius * angle.sin()).clamp(self.continuous_sampler.low[1], self.continuous_sampler.up[1]-0.0001);

		[x, y]
	}
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
	let policy = tamp_prm.plan(&[-0.9, 0.0], &initial_belief_state, 0.1, 2.0, 2500);

	let policy = policy.expect("nopath tree found!");

	let mut m2 = m.clone();
	m2.resize(5);
	m2.draw_zones_observability();
	//m2.draw_full_graph(&tamp_prm.search_tree.modes[2].prm.graph);
	m2.draw_policy(&policy);
	m2.save("results/test_map1_2_goals_tamp_prm");
}

#[test]
fn test_plan_on_map4_prm() {
	let mut m = MapShelfDomain::open("data/map_benchmark.pgm", [-1.0, -1.0], [1.0, 1.0]);
	m.add_zones("data/map_benchmark_4_goals_zone_ids.pgm", 0.5);

	let mut tamp_prm = MapShelfDomainTampPRM::new(ContinuousSampler::new([-1.0, -1.0], [1.0, 1.0]), DiscreteSampler::new(), &m, 0.05);		

	let initial_belief_state = vec![1.0/4.0; 4];
	let policy = tamp_prm.plan(&[0.0, -1.0], &initial_belief_state, 0.1, 2.0, 2500).expect("graph not grown up to solution");
	
	let mut m2 = m.clone();
	m2.resize(5);
	m2.draw_zones_observability();
	//m2.draw_full_graph(&tamp_prm.search_tree.modes[2].prm.graph);
	m2.draw_policy(&policy);
	m2.save("results/test_map1_4_goals_tamp_prm");
}

#[test]
fn test_plan_on_map7_plan_prm() {
	let mut m = MapShelfDomain::open("data/map7.pgm", [-1.0, -1.0], [1.0, 1.0]);
	m.add_zones("data/map7_6_goals_zone_ids.pgm", 0.5);

	let mut tamp_prm = MapShelfDomainTampPRM::new(ContinuousSampler::new([-1.0, -1.0], [1.0, 1.0]), DiscreteSampler::new(), &m, 0.05);	
	
	let initial_belief_state = vec![1.0/6.0; 6];
	let policy = tamp_prm.plan(&[0.0, -0.8], &initial_belief_state, 0.1, 2.0, 2500);
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