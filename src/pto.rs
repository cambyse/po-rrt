use itertools::{all, enumerate, izip, merge, zip};

use crate::common::*;
use crate::nearest_neighbor::*;
use crate::sample_space::*;
use crate::map_io::*; // tests only
use crate::map_shelves_io::*; // tests only
use crate::pto_graph::*;
use crate::pto_reachability::*;
use crate::belief_graph::*;
use std::time::Instant;
use bitvec::prelude::*;


pub struct PTO<'a, F: PTOFuncs<N>, const N: usize> {
	continuous_sampler: ContinuousSampler<N>,
	discrete_sampler: DiscreteSampler,
	pub fns: &'a F,
	pub kdtree: KdTree<N>,
	// graph growth
	pub n_worlds: usize,
	pub graph: PTOGraph<N>,
	// grow graph rrg
	pub conservative_reachability: Reachability,
	// pomdp
	node_to_belief_nodes: Vec<Vec<Option<usize>>>,
	pub belief_graph: BeliefGraph<N>,
	expected_costs_to_goals: Vec<f64>,
	// debug / benchmark
	pub n_it: usize,
	pub graph_growth_s: f64,
	pub belief_space_expansion_s: f64,
	pub dynamic_programming_s: f64
}

impl<'a, F: PTOFuncs<N>, const N: usize> PTO<'a, F, N> {
	pub fn new(continuous_sampler: ContinuousSampler<N>, discrete_sampler: DiscreteSampler, fns: &'a F) -> Self {
		Self { continuous_sampler,
			   discrete_sampler,
			   fns, 
			   kdtree: KdTree::new([0.0; N]),
			   n_worlds: fns.n_worlds(), 
			   graph: PTOGraph{nodes: vec![], validities: fns.world_validities()},
			   conservative_reachability: Reachability::new(), 
			   node_to_belief_nodes: Vec::new(),
		       belief_graph: BeliefGraph::new(Vec::new(), Vec::new()),
			   expected_costs_to_goals: Vec::new(),
			   n_it: 0,
			   graph_growth_s: 0.0,
			   belief_space_expansion_s: 0.0,
			   dynamic_programming_s: 0.0
			}
	}

	pub fn grow_graph(&mut self, &start: &[f64; N], goal: &impl GoalFuncs<N>,
				max_step: f64, search_radius: f64, n_iter_min: usize, n_iter_max: usize) -> Result<(), &'static str> {

		println!("grow graph..");
		let start_time = Instant::now();

		let root_validity_id = self.fns.state_validity(&start).expect("Start from a valid state!");
		self.graph.add_node(start, root_validity_id);
		self.conservative_reachability.set_root(self.graph.validities[root_validity_id].clone());
		self.kdtree.reset(start);

		let mut i = 0;
		while i < n_iter_min || !self.conservative_reachability.is_final_set_complete() && i < n_iter_max { // order important! avoid calling the is_final_set_complete as long as the number of iterations is not reached!
			i+=1;
	
			// First sample state and world
			let (world, mut new_state) = self.sample(goal, i); 
			
			// Second, retrieve closest node for sampled world and steer from there
			let kd_from = self.kdtree.nearest_neighbor_filtered(new_state, |id|{
				let r = self.conservative_reachability.reachability(id);
				r[world]
			}); // log n

			steer(&kd_from.state, &mut new_state, max_step); 

			if let Some(state_validity_id) = self.fns.state_validity(&new_state) {
				// Third, add node
				let new_node_id = self.graph.add_node(new_state, state_validity_id);
				let new_node = &self.graph.nodes[new_node_id];
				self.conservative_reachability.add_node(self.graph.validities[state_validity_id].clone());

				// Fourth, we find the neighbors in a specific radius of new_state.
				let radius = heuristic_radius(self.graph.nodes.len(), max_step, search_radius, N);

				//if i%100 == 0{
				//	println!("radius:{}", radius);
				//}

				// Fifth we connect to neighbors 
				let mut neighbour_ids: Vec<usize> = self.kdtree.nearest_neighbors(new_state, radius).iter()
				.map(|&kd_node| kd_node.id)
				.collect();

				if neighbour_ids.is_empty() { neighbour_ids.push(kd_from.id); }

				// No need to check differently the backward edges, assumption is pure path planning, and that paths are reversible
				// Idea: sample which ones we rewire to?
				let edges: Vec<(usize, usize)> = neighbour_ids.iter()
					.map(|&id| (id, &self.graph.nodes[id]))
					.map(|(id, node)| (id, self.fns.transition_validator(node, new_node)))
					.filter(|(_, validity_id)| validity_id.is_some())
					.map(|(id, validity_id)| (id, validity_id.unwrap()))
					.collect();
							
				// connect neighbors to new node
				for (id, validity_id) in &edges {
					self.conservative_reachability.add_edge(*id, new_node_id, &self.graph.validities[*validity_id]);
					self.graph.add_edge(*id, new_node_id, *validity_id);
				}

				// connect new node to neighbor
				for (id, validity_id) in &edges {
					self.conservative_reachability.add_edge(new_node_id, *id, &self.graph.validities[*validity_id]);
					self.graph.add_edge(new_node_id, *id, *validity_id);
				}

				if let Some(finality) = goal.goal(&new_state) {
					self.conservative_reachability.add_final_node(new_node_id, finality);
				}

				self.kdtree.add(new_state, new_node_id);
			}
		}

		self.n_it = i;
		self.graph_growth_s = start_time.elapsed().as_secs_f64();

		match self.conservative_reachability.is_final_set_complete() {
			true => {
				Ok(())
			},
			_ => Err(&"final nodes are not reached for each world")
		}
	}

	pub fn sample(&mut self, goal: &impl GoalFuncs<N>, iteration: usize) -> (usize, [f64; N]) {
		let world = self.discrete_sampler.sample(self.n_worlds);
		let new_state = match iteration % 100 {
			0 => goal.goal_example(world),
			_ => self.continuous_sampler.sample()
		};

		(world, new_state)
	}

	#[allow(clippy::style)]
	pub fn plan_belief_space(&mut self, start_belief_state: &BeliefState) -> Policy<N> {
		assert_belief_state_validity(start_belief_state);
		
		//
		println!("build belief graph..");
		let start_time = Instant::now();
		//

		self.build_belief_graph(start_belief_state);

		//
		self.belief_space_expansion_s = start_time.elapsed().as_secs_f64();
		println!("compute expected costs to goal..");
		let start_time = Instant::now();
		//

		self.compute_expected_costs_to_goals();
		
		//
		println!("extract policy..");
		//

		let policy = self.extract_policy();

		//
		self.dynamic_programming_s = start_time.elapsed().as_secs_f64();
		println!("success!");
		//

		policy
	}

	#[allow(clippy::style)]
	pub fn build_belief_graph(&mut self, start_belief_state: &BeliefState) {
		// build belief state graph
		let reachable_belief_states = self.fns.reachable_belief_states(start_belief_state);
		let world_validities = self.fns.world_validities();
		let compatibilities = compute_compatibility(&reachable_belief_states, &world_validities);

		let mut belief_space_graph = BeliefGraph::new(Vec::new(), reachable_belief_states.clone());

		//let mut belief_space_graph: BeliefGraph<N> = BeliefGraph{nodes: Vec::new(), reachable_belief_states: reachable_belief_states.clone()};
		let mut node_to_belief_nodes: Vec<Vec<Option<usize>>> = vec![vec![None; reachable_belief_states.len()]; self.graph.n_nodes()];
		
		// build nodes
		for (id, node) in self.graph.nodes.iter().enumerate() {
			for (belief_id, belief_state) in reachable_belief_states.iter().enumerate() {
				let belief_node_id = belief_space_graph.add_node(node.state, belief_state.clone(), belief_id, BeliefNodeType::Unknown);

				if compatibilities[belief_id][node.validity_id] {
					node_to_belief_nodes[id][belief_id] = Some(belief_node_id);
					belief_space_graph.nodes[belief_node_id].children.reserve(node.children.len());
					belief_space_graph.nodes[belief_node_id].parents.reserve(node.parents.len());
				}
			}
		}

		// build transitions due to observations (observation edges)
		for (id, node) in self.graph.nodes.iter().enumerate() {
			for (belief_id, belief_state) in reachable_belief_states.iter().enumerate() {
				let children_belief_states = self.fns.observe(&node.state, &belief_state);
				let parent_belief_node_id = node_to_belief_nodes[id][belief_id];

				for child_belief_state in &children_belief_states {
					if hash(belief_state) != hash(child_belief_state) {
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
				if let Some(parent_id) = node_to_belief_nodes[id][belief_id] {
					if belief_space_graph.nodes[parent_id].node_type == BeliefNodeType::Observation {
						continue;
					}

					for child_edge in &node.children {
						if let Some(child_id) = node_to_belief_nodes[child_edge.id][belief_id] {

							let parent_belief_id = belief_space_graph.nodes[parent_id].belief_id;
							let child_validity_id = child_edge.validity_id;
							if compatibilities[parent_belief_id][child_validity_id] {
								belief_space_graph.nodes[parent_id].node_type = BeliefNodeType::Action;
								belief_space_graph.add_edge(parent_id, child_id);
							}
						}
					}
				}
			}
		}

		self.node_to_belief_nodes = node_to_belief_nodes;
		self.belief_graph = belief_space_graph;
	}

	pub fn compute_expected_costs_to_goals(&mut self) {
		// create belief node ids in belief space
		let mut final_belief_state_node_ids: Vec<usize> = Vec::new();
		for (&final_id, validity) in self.conservative_reachability.final_nodes_with_validities() {
			for belief_node_id in self.node_to_belief_nodes[final_id].iter().flatten() {
				let belief_state = &self.belief_graph.nodes[*belief_node_id].belief_state;
				if is_compatible(belief_state, validity) {
					final_belief_state_node_ids.push(*belief_node_id);
				}
			}
		}

		// DP in belief state
		self.expected_costs_to_goals = conditional_dijkstra(&self.belief_graph, &final_belief_state_node_ids, &|a: &[f64; N], b: &[f64;N]| self.fns.cost_evaluator(a, b));
	}

	pub fn extract_policy(&self) -> Policy<N> {
		//let mut policy = extract_policy(&self.belief_graph, &self.expected_costs_to_goals, &|a: &[f64; N], b: &[f64;N]| self.fns.cost_evaluator(a, b) );
		//policy.compute_expected_costs_to_goals(&|a: &[f64; N], b: &[f64;N]| self.fns.cost_evaluator(a, b));
		//println!("COST:{}", policy.expected_costs);

		extract_policy(&self.belief_graph, &self.expected_costs_to_goals, &|a: &[f64; N], b: &[f64;N]| self.fns.cost_evaluator(a, b) )
	}

	pub fn print_summary(&self) {
		println!("number of iterations:{}", self.n_it);
		self.graph.print_summary();
	}
}

#[cfg(test)]
mod tests {

use crate::belief_graph;

use super::*;

#[test]
fn test_plan_on_map0_pomdp() {
	let mut m = Map::open("data/map0.pgm", [-1.0, -1.0], [1.0, 1.0]);
	m.init_without_zones();

	let goal = SquareGoal::new(vec![([0.0, 0.9], bitvec![1; 1])], 0.05);
	let mut pto = PTO::new(ContinuousSampler::new([-1.0, -1.0], [1.0, 1.0]),
						   DiscreteSampler::new(),
						   &m);

	pto.grow_graph(&[0.0, 0.0], &goal, 0.1, 5.0, 100, 100000).expect("graph not grown up to solution");
	pto.print_summary();
	let policy = pto.plan_belief_space(&vec![1.0]);

	let mut m2 = m.clone();
	m2.resize(5);
	m2.draw_full_graph(&pto.graph);
	m2.draw_zones_observability();
	m2.draw_policy(&policy);
	m2.save("results/test_pto_on_map0_pomdp");
}

#[test]
fn test_plan_on_map2_pomdp() {
	let mut m = Map::open("data/map2.pgm", [-1.0, -1.0], [1.0, 1.0]);
	m.add_zones("data/map2_zone_ids.pgm", 0.2);

	let goal = SquareGoal::new(vec![([0.55, 0.9], bitvec![1; 4])], 0.05);
	let mut pto = PTO::new(ContinuousSampler::new([-1.0, -1.0], [1.0, 1.0]),
						   DiscreteSampler::new(),
						   &m);

	pto.grow_graph(&[0.55, -0.8], &goal, 0.1, 5.0, 2000, 100000).expect("graph not grown up to solution");
	pto.print_summary();
	let policy = pto.plan_belief_space(&vec![0.1, 0.1, 0.1, 0.7]);

	let mut m2 = m.clone();
	m2.resize(5);
	m2.draw_full_graph(&pto.graph);
	m2.draw_zones_observability();
	m2.draw_policy(&policy);
	m2.save("results/test_pto_on_map2_pomdp");
}

#[test]
fn test_plan_on_map4_pomdp() {
	let mut m = Map::open("data/map4.pgm", [-1.0, -1.0], [1.0, 1.0]);
	m.add_zones("data/map4_zone_ids.pgm", 0.15);

	let goal = SquareGoal::new(vec![([-0.55, 0.9], bitvec![1; 16])], 0.05);
	let mut pto = PTO::new(ContinuousSampler::new([-1.0, -1.0], [1.0, 1.0]),
						   DiscreteSampler::new(),
						   &m);

	pto.grow_graph(&[0.55, -0.8], &goal, 0.05, 5.0, 10500, 100000).expect("graph not grown up to solution");
	pto.print_summary();
	let policy = pto.plan_belief_space( &vec![1.0/16.0; 16]);

	let mut m2 = m.clone();
	m2.resize(5);
	m2.draw_full_graph(&pto.graph);
	m2.draw_zones_observability();
	m2.draw_policy(&policy);
	m2.save("results/test_pto_on_map4_pomdp");
}

#[test]
fn test_plan_on_map1_fov_pomdp() {
	let mut m = Map::open("data/map1_fov.pgm", [-1.0, -1.0], [1.0, 1.0]);
	m.add_zones("data/map1_fov_zone_ids.pgm", 1.5);

	let goal = SquareGoal::new(vec![([0.85, 0.37], bitvec![1; 2])], 0.05);
	let mut pto = PTO::new(ContinuousSampler::new([-1.0, -1.0], [1.0, 1.0]),
						   DiscreteSampler::new(),
						   &m);

	pto.grow_graph(&[-0.37, 0.37], &goal, 0.05, 5.0, 5000, 100000).expect("graph not grown up to solution");
	pto.print_summary();
	let policy = pto.plan_belief_space(&vec![0.5, 0.5]);

	let mut m2 = m.clone();
	m2.resize(5);
	m2.draw_full_graph(&pto.graph);
	m2.draw_zones_observability();
	m2.draw_policy(&policy);
	m2.save("results/test_pto_on_map1_fov_pomdp");
}

#[test]
fn test_plan_on_map2_fov_pomdp() {
	let mut m = Map::open("data/map2_fov.pgm", [-1.0, -1.0], [1.0, 1.0]);
	m.add_zones("data/map2_fov_zone_ids.pgm", 1.1);

	//fn goal(state: &[f64; 2]) -> WorldMask {
	//	bitvec![if (state[0] - 0.775).abs() < 0.05 && (state[1] - 0.3).abs() < 0.05 { 1 } else { 0 }; 4]
	//}

	let goal = SquareGoal::new(vec![([0.775, 0.4], bitvec![1; 4])], 0.05);


	let mut pto = PTO::new(ContinuousSampler::new([-1.0, -1.0], [1.0, 1.0]),
						   DiscreteSampler::new(),
						   &m);

	pto.grow_graph(&[0.35, -0.125], &goal, 0.05, 5.0, 5000, 100000).expect("graph not grown up to solution");
	pto.print_summary();
	let policy = pto.plan_belief_space(&vec![0.25, 0.25, 0.25, 0.25]);

	let mut m2 = m.clone();
	m2.resize(5);
	m2.draw_full_graph(&pto.graph);
	m2.draw_zones_observability();
	m2.draw_policy(&policy);
	m2.save("results/test_pto_on_map2_fov_pomdp");
}

#[test]
fn test_plan_on_map2_fov_bounds_restricted_pomdp() { // same example as before but with restricted sampling bounds
	let mut m = Map::open("data/map2_fov.pgm", [-1.0, -1.0], [1.0, 1.0]);
	m.add_zones("data/map2_fov_zone_ids.pgm", 1.1);

	let goal = SquareGoal::new(vec![([0.775, 0.4], bitvec![1; 4])], 0.05);


	let mut pto = PTO::new(ContinuousSampler::new([-1.0, -0.5], [1.0, 0.5]),
						   DiscreteSampler::new(),
						   &m);

	pto.grow_graph(&[0.35, -0.125], &goal, 0.05, 5.0, 5000, 100000).expect("graph not grown up to solution");
	pto.print_summary();
	let policy = pto.plan_belief_space(&vec![0.25, 0.25, 0.25, 0.25]);

	let mut m2 = m.clone();
	m2.resize(5);
	m2.draw_full_graph(&pto.graph);
	m2.draw_zones_observability();
	m2.draw_policy(&policy);
	m2.save("results/test_pto_on_map2_fov_restricted_pomdp");
}


#[test]
fn test_plan_on_map0() {
	let mut m = Map::open("data/map0.pgm", [-1.0, -1.0], [1.0, 1.0]);
	m.init_without_zones();

	//fn goal(state: &[f64; 2]) -> WorldMask {
	//	bitvec![ if (state[0] - 0.5).abs() < 0.05 && (state[1] - 0.35).abs() < 0.05 { 1 } else { 0 }; 1]
	//}	

	let goal = SquareGoal::new(vec![([0.5, 0.35], bitvec![1; 1])], 0.05);


	let mut pto = PTO::new(ContinuousSampler::new([-1.0, -1.0], [1.0, 1.0]),
						   DiscreteSampler::new(),
						   &m);

	pto.grow_graph(&[0.55, -0.8], &goal, 0.1, 5.0, 500, 10000).expect("graph not grown up to solution");
	pto.print_summary();
	let policy = pto.plan_belief_space(&vec![1.0]);

	let mut m2 = m.clone();
	m2.resize(5);
	m2.draw_full_graph(&pto.graph);
	m2.draw_zones_observability();
	m2.draw_policy(&policy);
	m2.save("results/test_pto_on_map0");
}

#[test]
fn test_plan_on_map1_2_goals() {
	let mut m = MapShelfDomain::open("data/map1_2_goals.pgm", [-1.0, -1.0], [1.0, 1.0]);
	m.add_zones("data/map1_2_goals_zone_ids.pgm", 0.5);

	let goal = SquareGoal::new(vec![([0.68, -0.45], bitvec![1, 0]),
									([0.68, 0.38], bitvec![0, 1])], 0.05);

	let mut pto = PTO::new(ContinuousSampler::new([-1.0, -1.0], [1.0, 1.0]),
						DiscreteSampler::new(),
						&m);

	pto.grow_graph(&[-0.8, -0.8], &goal, 0.05, 5.0, 2000, 100000).expect("graph not grown up to solution");
	pto.print_summary();

	let policy = pto.plan_belief_space(&vec![0.2, 0.8]);

	let mut m2 = m.clone();
	m2.resize(5);
	m2.draw_full_graph(&pto.graph);
	m2.draw_zones_observability();
	m2.draw_policy(&policy);
	m2.save("results/test_plan_on_map1_2_goals_pomdp");
}

#[test]
fn test_plan_on_map1_3_goals() {
	let mut m = MapShelfDomain::open("data/map1_3_goals.pgm", [-1.0, -1.0], [1.0, 1.0]);
	m.add_zones("data/map1_3_goals_zone_ids.pgm", 0.5);

	let goal = SquareGoal::new(vec![([0.65, 0.14], bitvec![1, 0, 0]),
									([0.0, 0.75], bitvec![0, 1, 0]),
									([-0.7, 0.14], bitvec![0, 0, 1])],
									 0.05);

	let mut pto = PTO::new(ContinuousSampler::new([-1.0, -1.0], [1.0, 1.0]),
						DiscreteSampler::new(),
						&m);

	pto.grow_graph(&[0.0, -0.8], &goal, 0.05, 5.0, 5000, 100000).expect("graph not grown up to solution");
	pto.print_summary();

	//let policy = pto.plan_belief_space(&vec![1.0/3.0, 1.0/3.0, 1.0/3.0]);
	let policy = pto.plan_belief_space(&vec![0.5/3.0, 0.5/3.0, 2.0/3.0]);

	let mut m2 = m.clone();
	m2.resize(5);
	m2.draw_full_graph(&pto.graph);
	m2.draw_zones_observability();
	m2.draw_policy(&policy);
	m2.save("results/test_plan_on_map1_3_goals_pomdp");
}

#[test]
fn test_plan_on_map5_4_goals() {
	let mut m = MapShelfDomain::open("data/map5.pgm", [-1.0, -1.0], [1.0, 1.0]);
	m.add_zones("data/map5_4_goals_zone_ids.pgm", 0.4);

	let goal = SquareGoal::new(vec![([-0.75, 0.75], bitvec![1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
									([-0.25, 0.75], bitvec![0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
									([ 0.25, 0.75], bitvec![0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
									([ 0.75, 0.75], bitvec![0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0])],
									 0.05);

	let mut pto = PTO::new(ContinuousSampler::new([-1.0, -1.0], [1.0, 1.0]),
						DiscreteSampler::new(),
						&m);

	pto.grow_graph(&[0.0, -0.8], &goal, 0.05, 5.0, 5000, 100000).expect("graph not grown up to solution");
	pto.print_summary();

	let policy = pto.plan_belief_space(&vec![1.0/4.0, 1.0/4.0, 1.0/4.0, 1.0/4.0]);

	let mut m2 = m.clone();
	m2.resize(5);
	m2.draw_full_graph(&pto.graph);
	m2.draw_zones_observability();
	m2.draw_policy(&policy);
	m2.save("results/map5/test_map5_4_goals_pomdp");
}

#[test]
fn test_build_belief_graph() {
	let mut m = Map::open("data/map1.pgm", [-1.0, -1.0], [1.0, 1.0]);
	m.add_zones("data/map1_zone_ids.pgm", 0.1);

	let mut pto = PTO::new(ContinuousSampler::new([-1.0, -1.0], [1.0, 1.0]),
						   DiscreteSampler::new(),
						   &m);
	// mock graph growth
	pto.n_worlds = 2;
	pto.graph.validities = vec![bitvec![0, 1], bitvec![1, 1]];

	pto.graph.add_node([0.55, -0.8], 1); // 0 ~ [0, 1, 2]
	pto.graph.add_node([-0.42, -0.38], 1); // 1 ~ [3, 4, 5]
	pto.graph.add_node([0.54, 0.0], 1);   // 2 ~ [6, 7, 8]
	pto.graph.add_node([0.54, 0.1], 0);   // 3 ~ [10, 11, 12]
	pto.graph.add_node([-0.97, 0.65], 1); // 4 ~ [9, 10, 11]
	pto.graph.add_node([0.55, 0.9], 1);   // 5 ~ [15, 16, 17]

	pto.graph.add_bi_edge(0, 1, 1);
	pto.graph.add_bi_edge(1, 2, 1);
	pto.graph.add_bi_edge(2, 3, 0);
	pto.graph.add_bi_edge(3, 5, 0);

	pto.graph.add_bi_edge(1, 4, 1);
	pto.graph.add_bi_edge(4, 5, 1);

	pto.conservative_reachability.add_final_node(5, bitvec![1, 1]);
	//

	let _policy = pto.plan_belief_space(&vec![0.5, 0.5]);	
	assert_eq!(pto.belief_graph.nodes[6].children, vec![7, 8]); // observation transitions
	assert!(!pto.belief_graph.nodes[7].children.contains(&6)); // observation is irreversible
	assert!(!pto.belief_graph.nodes[8].children.contains(&6)); // observation is irreversible

	for (id, node) in pto.belief_graph.nodes.iter().enumerate() { // belief jump only at observation points
		if id == 6 {
			continue;
		}
		for &child_id in &node.children {
			assert_eq!(node.belief_id, pto.belief_graph.nodes[child_id].belief_id);
		}
	}

	// draw
	//let mut full = m.clone();
	//full.resize(5);
	//full.draw_full_graph(&pto.graph);
	//full.draw_policy(&policy);
	//full.save("results/test_build_belief_graph.pgm");
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
// - remove qmdp
// - tentative propagation
// - transition onm
// TODO:
// - avoid copies
// - optimize nearest neighbor (avoid sqrt)
// - actual PTO roadmap

// tentative propagation code