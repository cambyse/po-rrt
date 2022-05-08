use itertools::{all, enumerate, izip, merge, zip};

use crate::common::*;
use crate::nearest_neighbor::*;
use crate::sample_space::*;
use crate::map_io::*; // tests only
use crate::map_shelves_io::*; // tests only
use crate::map_shelves_io::colors; // tests only
use crate::pto_graph::*;
use std::time::Instant;


pub struct PRM<'a, F: PTOFuncs<N>, const N: usize> {
	continuous_sampler: ContinuousSampler<N>,
	pub fns: &'a F,
	pub kdtree: KdTree<N>,
	// prm graph growth
	pub graph: PTOGraph<N>,
	// debug
	pub n_it: usize
}

impl<'a, F: PTOFuncs<N>, const N: usize> PRM<'a, F, N> {
	pub fn new(continuous_sampler: ContinuousSampler<N>, fns: &'a F) -> Self {
		Self { continuous_sampler,
			   fns, 
			   kdtree: KdTree::new([0.0; N]),
			   graph: PTOGraph{nodes: vec![], validities: fns.world_validities()},
			   n_it: 0
			}
	}

	pub fn init(&mut self, &start: &[f64; N]) {
		self.graph.add_node(start, 0);
		self.kdtree.reset(start);
	}

	pub fn grow_graph(&mut self, max_step: f64, search_radius: f64, n_iter: usize) {

		println!("grow graph..");
		//let start_time = Instant::now();
		
		let mut i = 0;
		while i < n_iter { // order important! avoid calling the is_final_set_complete as long as the number of iterations is not reached!
			i+=1;
	
			// First sample state and world
			let new_state = self.continuous_sampler.sample();
			
			// Second, retrieve closest node for sampled world and steer from there
			//let kd_from = self.kdtree.nearest_neighbor(new_state); // log n

			// Third, add node
			let new_node_id = self.graph.add_node(new_state, 0);
			let new_node = &self.graph.nodes[new_node_id];

			// Fourth, we find the neighbors in a specific radius of new_state.
			let radius = heuristic_radius(self.graph.nodes.len(), max_step, search_radius, N);

			//if i%100 == 0{
			//	println!("radius:{}", radius);
			//}

			// Fifth we connect to neighbors 
			let neighbour_ids: Vec<usize> = self.kdtree.nearest_neighbors(new_state, radius).iter()
			.map(|&kd_node| kd_node.id)
			.collect();

			self.kdtree.add(new_state, new_node_id);

			if neighbour_ids.is_empty() { 
				continue;
			}

			// No need to check differently the backward edges, assumption is pure path planning, and that paths are reversible
			// Idea: sample which ones we rewire to?
			let edges: Vec<(usize, usize)> = neighbour_ids.iter()
				.map(|&id| (id, &self.graph.nodes[id]))
				.map(|(id, node)| (id, self.fns.transition_validator(node, new_node)))
				.filter(|(_, validity_id)| validity_id.is_some())
				.map(|(id, validity_id)| (id, validity_id.unwrap()))
				.collect();
						
			// connect neighbors to new node
			for (id, _) in &edges {
				self.graph.add_edge(*id, new_node_id, 0);
			}

			// connect new node to neighbor
			for (id, _) in &edges {
				self.graph.add_edge(new_node_id, *id, 0);
			}

			self.n_it += 1;
		}
	}

	pub fn plan_path(&mut self, start: &[f64; N], goal: &[f64; N]) -> Vec<[f64; N]>  {
		let kd_start = self.kdtree.nearest_neighbor(*start);
		let kd_goal = self.kdtree.nearest_neighbor(*goal);

		let cost_to_goal = dijkstra(&self.graph, &vec![kd_goal.id], self.fns);
		extract_path(&self.graph, kd_start.id, &cost_to_goal, self.fns)
	}

	/*
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
	}*/

	pub fn print_summary(&self) {
		//println!("number of iterations:{}", self.n_it);
		self.graph.print_summary();
	}
}

#[cfg(test)]
mod tests {

use super::*;

#[test]
fn test_plan_on_map0_prm() {
	let mut m = Map::open("data/map0.pgm", [-1.0, -1.0], [1.0, 1.0]);
	m.init_without_zones();

	let mut prm = PRM::new(ContinuousSampler::new([-1.0, -1.0], [1.0, 1.0]),
						   &m);

	prm.init(&[0.0, 0.0]);					   
	prm.grow_graph(0.1, 5.0, 2500);
	prm.print_summary();
	let path = prm.plan_path(&[0.0, 0.0], &[0.0, 0.9]);

	let mut m2 = m.clone();
	m2.resize(5);
	m2.draw_full_graph(&prm.graph);
	m2.draw_zones_observability();
	m2.draw_path(&path, colors::BLACK);
	m2.save("results/test_plan_on_map0_prm");
}
}