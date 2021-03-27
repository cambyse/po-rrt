use itertools::{all, enumerate, izip, merge, zip};

use crate::common::*;
use crate::nearest_neighbor::*;
use crate::sample_space::*;
use crate::map_io::*; // tests only
use crate::prm_graph::*;
use crate::prm_reachability::*;
use crate::belief_graph::*;
use bitvec::prelude::*;
use std::{collections::HashMap, ops::Index};

pub struct PRM<'a, F: PRMFuncs<N>, const N: usize> {
	continuous_sampler: ContinuousSampler<N>,
	discrete_sampler: DiscreteSampler,
	fns: &'a F,
	kdtree: KdTree<N>,
	// graph growth
	n_worlds: usize,
	n_it: usize,
	pub graph: PRMGraph<N>,
	final_node_ids: Vec<usize>,
	// grow graph rrg
	conservative_reachability: Reachability,
	// qmdp
	cost_to_goals: Vec<Vec<f64>>,
	// pomdp
	node_to_belief_nodes: Vec<Vec<Option<usize>>>,
	belief_graph: BeliefGraph<N>,
	expected_costs_to_goals: Vec<f64>
}

impl<'a, F: PRMFuncs<N>, const N: usize> PRM<'a, F, N> {
	pub fn new(continuous_sampler: ContinuousSampler<N>, discrete_sampler: DiscreteSampler, fns: &'a F) -> Self {
		Self { continuous_sampler,
			   discrete_sampler,
			   fns, 
			   kdtree: KdTree::new([0.0; N]),
			   n_worlds: 0, 
			   n_it: 0,
			   graph: PRMGraph{nodes: vec![]},
			   final_node_ids: Vec::new(),
			   conservative_reachability: Reachability::new(), 
			   cost_to_goals: Vec::new(),
			   node_to_belief_nodes: Vec::new(),
		       belief_graph: BeliefGraph{nodes: Vec::new(), reachable_belief_states: Vec::new()},
			   expected_costs_to_goals: Vec::new() }
	}

	pub fn grow_graph(&mut self, &start: &[f64; N], goal: fn(&[f64; N]) -> WorldMask,
				max_step: f64, search_radius: f64, n_iter_min: usize, n_iter_max: usize) -> Result<(), &'static str> {

		println!("grow graph..");

		let root_validity = self.fns.state_validity(&start).expect("Start from a valid state!");
		self.n_worlds = root_validity.len();
		self.graph.add_node(start, root_validity.clone());
		self.conservative_reachability.set_root(root_validity);
		self.kdtree.reset(start);

		let mut i = 0;
		while i < n_iter_min || !self.conservative_reachability.is_final_set_complete() && i < n_iter_max {
			i+=1;
	
			// First sample state and world
			let mut new_state = self.continuous_sampler.sample();
			let world = self.discrete_sampler.sample(self.n_worlds);

			// Second, retrieve closest node for sampled world and steer from there
			let kd_from = self.kdtree.nearest_neighbor_filtered(new_state, |id|{self.conservative_reachability.reachability(id)[world]}); // log n
			steer(&kd_from.state, &mut new_state, max_step); 

			if let Some(state_validity) = self.fns.state_validity(&new_state) {
				// Third, add node
				let new_node_id = self.graph.add_node(new_state, state_validity.clone());
				let new_node = &self.graph.nodes[new_node_id];
				self.conservative_reachability.add_node(state_validity.clone());

				// Fourth, we find the neighbors in a specific radius of new_state.
				let radius = {
					let n = self.graph.nodes.len() as f64;
					let s = search_radius * (n.ln()/n).powf(1.0/(N as f64));
					if s < max_step { s } else { max_step }
				};

				// Fifth we connect to neighbors 
				let mut neighbour_ids: Vec<usize> = self.kdtree.nearest_neighbors(new_state, radius).iter()
				.map(|&kd_node| kd_node.id)
				.collect();

				if neighbour_ids.is_empty() { neighbour_ids.push(kd_from.id); }

				// Idea: sample which ones we rewire to?
				let fwd_ids: Vec<usize> = neighbour_ids.iter()
					.map(|&id| (id, &self.graph.nodes[id]))
					.filter(|(_, node)| self.fns.transition_validator(node, new_node))
					.map(|(id, _)| id)
					.collect();

				let bwd_ids: Vec<usize> = neighbour_ids.iter()
					.map(|&id| (id, &self.graph.nodes[id]))
					.filter(|(_, node)| self.fns.transition_validator(new_node, node))
					.map(|(id, _)| id)
					.collect();
							
				for &id in &fwd_ids {
					self.graph.add_edge(id, new_node_id);
					self.conservative_reachability.add_edge(id, new_node_id);
				}
				
				for &id in &bwd_ids {
					self.graph.add_edge(new_node_id, id);
					self.conservative_reachability.add_edge(new_node_id, id);
				}
					
				let finality = goal(&new_state);
				let is_final = finality.iter().any(|w|{*w});
				if is_final {
					self.conservative_reachability.add_final_node(new_node_id, finality);
				}

				self.kdtree.add(new_state, new_node_id);
			}
		}

		self.n_it += i;

		match self.conservative_reachability.is_final_set_complete() {
			true => {
				self.final_node_ids = self.conservative_reachability.final_node_ids();
				Ok(())
			},
			_ => Err(&"final nodes are not reached for each world")
		}
	}

	#[allow(clippy::style)]
	pub fn plan_belief_state(&mut self, start_belief_state: &BeliefState) -> Policy<N> {
		assert_belief_state_validity(start_belief_state);
		
		println!("build belief graph..");

		self.build_belief_graph(start_belief_state);

		println!("compute expected costs to goal..");

		self.compute_expected_costs_to_goals();
		
		println!("extract policy..");

		let policy = self.extract_policy();

		println!("success!");

		policy
	}

	#[allow(clippy::style)]
	pub fn build_belief_graph(&mut self, start_belief_state: &BeliefState) {
		// build belief state graph
		let reachable_belief_states = self.fns.reachable_belief_states(start_belief_state);
		let mut belief_space_graph: BeliefGraph<N> = BeliefGraph{nodes: Vec::new(), reachable_belief_states: reachable_belief_states.clone()};
		let mut node_to_belief_nodes: Vec<Vec<Option<usize>>> = vec![vec![None; reachable_belief_states.len()]; self.graph.n_nodes()];
		
		// build nodes
		for (id, node) in self.graph.nodes.iter().enumerate() {
			for (belief_id, belief_state) in reachable_belief_states.iter().enumerate() {
				let belief_node_id = belief_space_graph.add_node(node.state, belief_state.clone(), belief_id, BeliefNodeType::Unknown);

				if is_compatible(belief_state, &node.validity) {
					node_to_belief_nodes[id][belief_id] = Some(belief_node_id);
				}
			}
		}

		// build transitions due to observations (observation edges)
		for (id, node) in self.graph.nodes.iter().enumerate() {
			for (belief_id, belief_state) in reachable_belief_states.iter().enumerate() {
				let children_belief_states = self.fns.observe(&node.state, &belief_state);
				let parent_belief_node_id = node_to_belief_nodes[id][belief_id];

				for child_belief_state in &children_belief_states {
					if belief_state != child_belief_state {
						// debug
						//let p = transition_probability(&belief_state, &child_belief_state);
						//assert!(p > 0.0);
						//

						let child_belief_state_id = belief_space_graph.belief_id(&child_belief_state);
						let child_belief_node_id = node_to_belief_nodes[id][child_belief_state_id];

						if let (Some(parent_id), Some(child_id)) = (parent_belief_node_id, child_belief_node_id) {
							belief_space_graph.nodes[parent_id].node_type = BeliefNodeType::Observation;
							belief_space_graph.add_edge(parent_id, child_id);
						}
					}
				}
			}
		}

		// build possible geometric edges (action edges)
		for (id, node) in self.graph.nodes.iter().enumerate() {
			for (belief_id, _) in reachable_belief_states.iter().enumerate() {
				let parent_belief_node_id = node_to_belief_nodes[id][belief_id];

				if parent_belief_node_id.is_some() && belief_space_graph.nodes[parent_belief_node_id.unwrap()].node_type == BeliefNodeType::Observation {
					continue;
				}

				for &child_id in &node.children {
					let child_belief_node_id = node_to_belief_nodes[child_id][belief_id];

					if let (Some(parent_id), Some(child_id)) = (parent_belief_node_id, child_belief_node_id) {
						belief_space_graph.nodes[parent_id].node_type = BeliefNodeType::Action;
						belief_space_graph.add_edge(parent_id, child_id);
					}
				}
			}
		}

		self.node_to_belief_nodes = node_to_belief_nodes;
		self.belief_graph = belief_space_graph;
	}

	pub fn compute_expected_costs_to_goals(&mut self) {
		//let mut final_belief_state_node_ids = final_node_ids.iter().fold(Vec::new(), |finals, final_id| { finals.extend(node_to_belief_nodes[final_id]); finals } );
		let mut final_belief_state_node_ids: Vec<usize> = Vec::new();
		for &final_id in &self.final_node_ids {
			for belief_node_id in &self.node_to_belief_nodes[final_id] {
				if belief_node_id.is_some() {
					final_belief_state_node_ids.push(belief_node_id.unwrap());
				}
			}
		}

		// DP in belief state
		self.expected_costs_to_goals = conditional_dijkstra(&self.belief_graph, &final_belief_state_node_ids, |a: &[f64; N], b: &[f64;N]| self.fns.cost_evaluator(a, b));
	}

	pub fn extract_policy(&self) -> Policy<N> {
		extract_policy(&self.belief_graph, &self.expected_costs_to_goals)
	}

	pub fn plan_qmdp(&mut self) -> Result<(), &'static str> {
		// compute the cost to goals
		self.cost_to_goals = vec![Vec::new(); self.n_worlds];
		for world in 0..self.n_worlds {
			let final_nodes = self.conservative_reachability.final_nodes_for_world(world);
			if final_nodes.is_empty() {
				return Err(&"We should have final node ids for each world")
			}
			self.cost_to_goals[world] = dijkstra(&PRMGraphWorldView{graph: &self.graph, world}, &final_nodes, self.fns);
		}

		Ok(())
	}

	#[allow(clippy::style)]
	pub fn react_qmdp(&mut self, start: &[f64; N], belief_state: &BeliefState, common_horizon: f64) -> Result<Vec<Vec<[f64; N]>>, &'static str> {
		let kd_start = self.kdtree.nearest_neighbor(*start);

		let (common_path, id) = self.get_common_path(kd_start.id, belief_state, common_horizon).unwrap();
		let mut paths : Vec<Vec<[f64; N]>> = vec![Vec::new(); self.n_worlds];
		for world in 0..self.n_worlds {
			paths[world] = common_path.clone();
			paths[world].extend(self.get_path(id, world));
		}

		Ok(paths)
	}

	fn get_policy_graph(&self) -> Result<PRMGraph<N>, &'static str> {
		get_policy_graph(&self.graph, &self.cost_to_goals)
	}

	#[allow(clippy::style)]
	fn get_common_path(&self, start_id:usize, belief_state: &BeliefState, common_horizon: f64) -> Result<(Vec<[f64; N]>, usize), &'static str> {
		if belief_state.len() != self.n_worlds {
			return Err("belief state size should match the number of worlds")
		}

		let mut path: Vec<[f64; N]> = Vec::new();

		let mut id = start_id;
		let mut smallest_expected_cost = std::f64::INFINITY;
		let mut accumulated_horizon = 0.0;

		while accumulated_horizon < common_horizon && smallest_expected_cost > 0.0 {
			path.push(self.graph.nodes[id].state);

			let id_cost = self.get_best_expected_child(id, belief_state);
			accumulated_horizon += norm2(&self.graph.nodes[id].state, &self.graph.nodes[id_cost.0].state); // TODO: replace by injected function?

			id = id_cost.0;
			smallest_expected_cost = id_cost.1;
		}
		
		Ok((path, id))
	}

	fn get_path(&self, start_id:usize, world: usize) -> Vec<[f64; N]> {
		let mut path: Vec<[f64; N]> = Vec::new();

		let mut id = start_id;
		while self.cost_to_goals[world][id] > 0.0 {
			path.push(self.graph.nodes[id].state);

			id = self.get_best_child(id, world);
		}

		path
	}

	pub fn print_summary(&self) {
		println!("number of iterations:{}", self.n_it);
		self.graph.print_summary();
	}

	#[allow(clippy::style)]
	fn get_best_expected_child(&self, node_id: usize, belief_state: &BeliefState) -> (usize, f64) {
		let node = &self.graph.nodes[node_id]; 
		let mut best_child_id = 0;
		let mut smallest_expected_cost = std::f64::INFINITY;

		for child_id in &node.children {
			let mut child_expected_cost = 0.0;

			for world in 0..self.n_worlds {
				child_expected_cost += self.cost_to_goals[world][*child_id] * belief_state[world];
			}

			if child_expected_cost < smallest_expected_cost {
				best_child_id = *child_id;
				smallest_expected_cost = child_expected_cost;
			}
		}
		(best_child_id, smallest_expected_cost)
	}

	fn get_best_child(&self, node_id: usize, world: usize) -> usize {
		let node = &self.graph.nodes[node_id]; 
		let mut best_child_id = 0;
		let mut smaller_cost = std::f64::INFINITY;

		for child_id in &node.children {
			if self.cost_to_goals[world][*child_id] < smaller_cost {
				smaller_cost = self.cost_to_goals[world][*child_id];
				best_child_id = *child_id;
			}
		}

		best_child_id
	}
}

#[cfg(test)]
mod tests {

use crate::belief_graph;

    use super::*;

#[test]
fn test_plan_on_map2_pomdp() {
	let mut m = Map::open("data/map2.pgm", [-1.0, -1.0], [1.0, 1.0]);
	m.add_zones("data/map2_zone_ids.pgm", 0.2);

	fn goal(state: &[f64; 2]) -> WorldMask {
		bitvec![if (state[0] - 0.55).abs() < 0.05 && (state[1] - 0.9).abs() < 0.05 { 1 } else { 0 }; 4]
	}

	let mut prm = PRM::new(ContinuousSampler::new([-1.0, -1.0], [1.0, 1.0]),
						   DiscreteSampler::new(),
						   &m);

	prm.grow_graph(&[0.55, -0.8], goal, 0.05, 5.0, 5000, 100000).expect("graph not grown up to solution");
	prm.print_summary();
	let policy = prm.plan_belief_state(&vec![0.1, 0.1, 0.1, 0.7]);

	let mut m2 = m.clone();
	m2.resize(5);
	m2.draw_full_graph(&prm.graph);
	m2.draw_zones_observability();
	m2.draw_policy(&policy);
	m2.save("results/test_prm_on_map2_pomdp");
}

#[test]
fn test_plan_on_map4_pomdp() {
	let mut m = Map::open("data/map4.pgm", [-1.0, -1.0], [1.0, 1.0]);
	m.add_zones("data/map4_zone_ids.pgm", 0.15);

	fn goal(state: &[f64; 2]) -> WorldMask {
		bitvec![if (state[0] + 0.55).abs() < 0.05 && (state[1] - 0.9).abs() < 0.05 { 1 } else { 0 }; 16]
	}

	let mut prm = PRM::new(ContinuousSampler::new([-1.0, -1.0], [1.0, 1.0]),
						   DiscreteSampler::new(),
						   &m);

	prm.grow_graph(&[0.55, -0.8], goal, 0.05, 5.0, 1000, 100000).expect("graph not grown up to solution");
	prm.print_summary();
	let policy = prm.plan_belief_state( &vec![1.0/16.0; 16]);

	let mut m2 = m.clone();
	m2.resize(5);
	m2.draw_full_graph(&prm.graph);
	m2.draw_zones_observability();
	m2.draw_policy(&policy);
	m2.save("results/test_prm_on_map4_pomdp");
}

#[test]
fn test_plan_on_map1_fov_pomdp() {
	let mut m = Map::open("data/map1_fov.pgm", [-1.0, -1.0], [1.0, 1.0]);
	m.add_zones("data/map1_fov_zone_ids.pgm", 1.5);

	fn goal(state: &[f64; 2]) -> WorldMask {
		bitvec![if (state[0] - 0.85).abs() < 0.05 && (state[1] - 0.37).abs() < 0.05 { 1 } else { 0 }; 2]
	}

	let mut prm = PRM::new(ContinuousSampler::new([-1.0, -1.0], [1.0, 1.0]),
						   DiscreteSampler::new(),
						   &m);

	prm.grow_graph(&[-0.37, 0.37], goal, 0.05, 5.0, 5000, 100000).expect("graph not grown up to solution");
	prm.print_summary();
	let policy = prm.plan_belief_state(&vec![0.5, 0.5]);

	let mut m2 = m.clone();
	m2.resize(5);
	m2.draw_full_graph(&prm.graph);
	m2.draw_zones_observability();
	m2.draw_policy(&policy);
	m2.save("results/test_prm_on_map1_fov_pomdp");
}

#[test]
fn test_plan_on_map2_fov_pomdp() {
	let mut m = Map::open("data/map2_fov.pgm", [-1.0, -1.0], [1.0, 1.0]);
	m.add_zones("data/map2_fov_zone_ids.pgm", 1.1);

	fn goal(state: &[f64; 2]) -> WorldMask {
		bitvec![if (state[0] - 0.775).abs() < 0.05 && (state[1] - 0.3).abs() < 0.05 { 1 } else { 0 }; 4]
	}

	let mut prm = PRM::new(ContinuousSampler::new([-1.0, -1.0], [1.0, 1.0]),
						   DiscreteSampler::new(),
						   &m);

	prm.grow_graph(&[0.35, -0.125], goal, 0.05, 5.0, 5000, 100000).expect("graph not grown up to solution");
	prm.print_summary();
	let policy = prm.plan_belief_state(&vec![0.25, 0.25, 0.25, 0.25]);

	let mut m2 = m.clone();
	m2.resize(5);
	m2.draw_full_graph(&prm.graph);
	m2.draw_zones_observability();
	m2.draw_policy(&policy);
	m2.save("results/test_prm_on_map2_fov_pomdp");
}

#[test]
fn test_plan_on_map2_qmdp() {
	let mut m = Map::open("data/map2.pgm", [-1.0, -1.0], [1.0, 1.0]);
	m.add_zones("data/map2_zone_ids.pgm", 0.1);

	fn goal(state: &[f64; 2]) -> WorldMask {
		bitvec![if (state[0] - 0.55).abs() < 0.05 && (state[1] - 0.9).abs() < 0.05 { 1 } else { 0 }; 4]
	}

	let mut prm = PRM::new(ContinuousSampler::new([-1.0, -1.0], [1.0, 1.0]),
						   DiscreteSampler::new(),
						   &m);

	prm.grow_graph(&[0.55, -0.8], goal, 0.05, 5.0, 5000, 100000).expect("graph not grown up to solution");
	prm.plan_qmdp().expect("general solution couldn't be found");
	let paths = prm.react_qmdp(&[0.55, -0.6], &vec![1.0/4.0; 4], 0.2).expect("impossible to extract policy");
	prm.print_summary();

	let mut full = m.clone();
	full.resize(5);
	full.draw_full_graph(&prm.graph);
//	full.draw_graph_from_root(&prm.get_policy_graph().unwrap());
//	full.draw_graph_for_world(&prm.graph, 0);

	for (i, path) in enumerate(&paths) {
		full.draw_path(path, crate::map_io::colors::color_map(i));
	}
	full.save("results/test_prm_on_map2_qmdp");
}

#[test]
fn test_plan_on_map1_2_goals() {
	let mut m = Map::open("data/map1_2_goals.pgm", [-1.0, -1.0], [1.0, 1.0]);
	m.add_zones("data/map1_2_goals_zone_ids.pgm", 0.1);

	fn goal(state: &[f64; 2]) -> WorldMask {
		let mut finality = bitvec![0;2];
		finality.set(0, (state[0] - 0.68).abs() < 0.05 && (state[1] + 0.45).abs() < 0.05);
		finality.set(1, (state[0] - 0.68).abs() < 0.05 && (state[1] - 0.38).abs() < 0.05);
		finality
	}

	let mut prm = PRM::new(ContinuousSampler::new([-1.0, -1.0], [1.0, 1.0]),
						   DiscreteSampler::new(),
						   &m);

	prm.grow_graph(&[-0.8, -0.8], goal, 0.05, 5.0, 5000, 100000).expect("graph not grown up to solution");
	prm.plan_qmdp().expect("general solution couldn't be found");
	let paths = prm.react_qmdp(&[-0.8, -0.8], &vec![0.5, 0.5], 0.2).expect("impossible to extract policy");
	prm.print_summary();

	let mut full = m.clone();
	full.resize(5);
	full.draw_full_graph(&prm.graph);
//	full.draw_graph_from_root(&prm.get_policy_graph().unwrap());
//	full.draw_graph_for_world(&prm.graph, 0);

	for (i, path) in enumerate(paths) {
		full.draw_path(&path, crate::map_io::colors::color_map(i));
	}

	full.save("results/test_prm_on_map1_2_goals");
}

#[test]
fn test_build_belief_graph() {
	let mut m = Map::open("data/map1.pgm", [-1.0, -1.0], [1.0, 1.0]);
	m.add_zones("data/map1_zone_ids.pgm", 0.1);

	fn goal(state: &[f64; 2]) -> WorldMask {
		bitvec![if (state[0] - 0.55).abs() < 0.05 && (state[1] - 0.9).abs() < 0.05 { 1 } else { 0 }; 4]
	}

	let mut prm = PRM::new(ContinuousSampler::new([-1.0, -1.0], [1.0, 1.0]),
						   DiscreteSampler::new(),
						   &m);
	// mock graph growth
	prm.n_worlds = 2;

	prm.graph.add_node([0.55, -0.8], bitvec![1, 1]); // 0
	prm.graph.add_node([-0.42, -0.38], bitvec![1, 1]); // 1
	prm.graph.add_node([0.54, 0.0], bitvec![1, 1]);   // 2
	prm.graph.add_node([0.54, 0.1], bitvec![0, 1]);   // 3
	prm.graph.add_node([-0.97, 0.65], bitvec![1, 1]); // 4
	prm.graph.add_node([0.55, 0.9], bitvec![1, 1]);   // 5

	prm.graph.add_edge(0, 1); 	prm.graph.add_edge(1, 0);
	prm.graph.add_edge(1, 2);   prm.graph.add_edge(2, 1);
	prm.graph.add_edge(2, 3);   prm.graph.add_edge(3, 2);
	prm.graph.add_edge(3, 5);   prm.graph.add_edge(5, 3);

	prm.graph.add_edge(1, 4);   prm.graph.add_edge(4, 1);
	prm.graph.add_edge(4, 5);   prm.graph.add_edge(5, 4);

	prm.final_node_ids.push(5);
	//

	let _policy = prm.plan_belief_state(&vec![0.5, 0.5]);	
	assert_eq!(prm.belief_graph.nodes[6].children, vec![7, 8]); // observation transitions
	assert!(!prm.belief_graph.nodes[7].children.contains(&6)); // observation is irreversible
	assert!(!prm.belief_graph.nodes[8].children.contains(&6)); // observation is irreversible

	for (id, node) in prm.belief_graph.nodes.iter().enumerate() { // belief jump only at observation points
		if id == 6 {
			continue;
		}
		for &child_id in &node.children {
			assert_eq!(node.belief_id, prm.belief_graph.nodes[child_id].belief_id);
		}
	}

	// draw
	//let mut full = m.clone();
	//full.resize(5);
	//full.draw_full_graph(&prm.graph);
	//full.draw_policy(&policy);
	//full.save("results/test_build_belief_graph.pgm");
}


#[test]
#[should_panic]
fn test_when_grow_graph_doesnt_reach_goal() {
	let mut m = Map::open("data/map2.pgm", [-1.0, -1.0], [1.0, 1.0]);
	m.add_zones("data/map2_zone_ids.pgm", 0.1);

	fn goal(state: &[f64; 2]) -> WorldMask {
		bitvec![if (state[0] - 0.55).abs() < 0.05 && (state[1] - 0.9).abs() < 0.05 { 1 } else { 0 }; 4]
	}

	let mut prm = PRM::new(ContinuousSampler::new([-1.0, -1.0], [1.0, 1.0]),
						   DiscreteSampler::new(),
						   &m);

	assert_ne!(Ok(()), prm.grow_graph(&[0.55, -0.8], goal, 0.05, 5.0, 300, 1000));

	prm.plan_qmdp().unwrap(); // panics
}
}

// Compresser pour avoir N mondes même pour des domaines où le nombre de mondes explose

// Done:
// - random seed + time saving
// - dijkstra
// - more efficient tree growing
// - reachability
// - serializaion
// - error flow
// - add transition check
// - resize map
// - extract common path
// - plan from random point
// - field of view observation model
// TODO:
// - avoid copies
// - optimize nearest neighbor (avoid sqrt)
// - actual PRM roadmap
// - transition on
