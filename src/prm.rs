use itertools::{all, enumerate, izip, merge, zip};

use crate::common::*;
use crate::nearest_neighbor::*;
use crate::sample_space::*;
use crate::map_io::*; // tests only
use crate::prm_graph::*;
use crate::prm_reachability::*;
use bitvec::prelude::*;

pub struct PRM<'a, F: PRMFuncs<N>, const N: usize> {
	continuous_sampler: ContinuousSampler<N>,
	discrete_sampler: DiscreteSampler,
	fns: &'a F,
	kdtree: KdTree<N>,
	graph: PRMGraph<N>,
	conservative_reachability: Reachability,
	cost_to_goals: Vec<Vec<f64>>,
	n_worlds: usize,
	n_it: usize
}

impl<'a, F: PRMFuncs<N>, const N: usize> PRM<'a, F, N> {
	pub fn new(continuous_sampler: ContinuousSampler<N>, discrete_sampler: DiscreteSampler, fns: &'a F) -> Self {
		Self { continuous_sampler,
			   discrete_sampler,
			   fns, 
			   kdtree: KdTree::new([0.0; N]),
			   graph: PRMGraph{nodes: vec![]},
			   conservative_reachability: Reachability::new(), 
			   cost_to_goals: Vec::new(),
			   n_worlds: 0, 
			   n_it: 0 }
	}

	pub fn grow_graph(&mut self, &start: &[f64; N], goal: fn(&[f64; N]) -> WorldMask,
				max_step: f64, search_radius: f64, n_iter_min: usize, n_iter_max: usize) -> Result<(), &'static str> {

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
			let kd_from = self.kdtree.nearest_neighbor_filtered(new_state, &|id|{self.conservative_reachability.reachability(id)[world]}); // log n
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
				.map(|(id, _)| id.clone())
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
				if finality.iter().any(|w|{*w}) {
					self.conservative_reachability.add_final_node(new_node_id, finality);
				}

				self.kdtree.add(new_state, new_node_id);
			}
		}

		self.n_it += i;

		match self.conservative_reachability.is_final_set_complete() {
			true => Ok(()),
			_ => Err(&"final nodes are not reached for each world")
		}
	}

	pub fn plan_belief_state(&mut self, _belief_state: &Vec<f64>) {
		// belief state graph
	}

	pub fn plan_qmdp(&mut self) -> Result<(), &'static str> {
		// compute the cost to goals
		self.cost_to_goals = vec![Vec::new(); self.n_worlds];
		for world in 0..self.n_worlds {
			let final_nodes = self.conservative_reachability.final_nodes_for_world(world);
			if final_nodes.is_empty() {
				return Err(&"We should have final node ids for each world")
			}
			self.cost_to_goals[world] = dijkstra(&self.graph, &final_nodes, world, self.fns);
		}

		Ok(())
	}

	pub fn react_qmdp(&mut self, start: &[f64; N], belief_state: &Vec<f64>, common_horizon: f64) -> Result<Vec<Vec<[f64; N]>>, &'static str> {
		let kd_start = self.kdtree.nearest_neighbor(*start);

		let (common_path, id) = self.get_common_path(kd_start.id, belief_state, common_horizon).unwrap();
		let mut paths : Vec<Vec<[f64; N]>> = vec![Vec::new(); self.n_worlds];
		for world in 0..self.n_worlds {
			paths[world] = common_path.clone();
			paths[world].extend(self.get_path(id, world).expect("path should be succesfully extracted at this stage, since each world has final nodes"));
		}

		Ok(paths)
	}

	fn get_policy_graph(&self) -> Result<PRMGraph<N>, &'static str> {
		get_policy_graph(&self.graph, &self.cost_to_goals)
	}

	fn get_common_path(&self, start_id:usize, belief_state: &Vec<f64>, common_horizon: f64) -> Result<(Vec<[f64; N]>, usize), &'static str> {
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

	fn get_path(&self, start_id:usize, world: usize) -> Result<Vec<[f64; N]>, &'static str> {
		let mut path: Vec<[f64; N]> = Vec::new();

		let mut id = start_id;
		while self.cost_to_goals[world][id] > 0.0 {
			path.push(self.graph.nodes[id].state);

			id = self.get_best_child(id, world);
		}

		Ok(path)
	}

	pub fn print_summary(&self) {
		println!("number of iterations:{}", self.n_it);
		self.graph.print_summary();
	}

	fn get_best_expected_child(&self, node_id: usize, belief_state: &Vec<f64>) -> (usize, f64) {
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

use super::*;

#[test]
fn test_plan_on_map2() {
	let mut m = Map::open("data/map2.pgm", [-1.0, -1.0], [1.0, 1.0]);
	m.add_zones("data/map2_zone_ids.pgm");

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

	for path in paths {
		full.draw_path(path);
	}
	full.save("results/test_plan_on_map2.pgm");
}

#[test]
fn test_plan_on_map4() {
	let mut m = Map::open("data/map4.pgm", [-1.0, -1.0], [1.0, 1.0]);
	m.add_zones("data/map4_zone_ids.pgm");

	fn goal(state: &[f64; 2]) -> WorldMask {
		bitvec![if (state[0] - 0.55).abs() < 0.05 && (state[1] - 0.9).abs() < 0.05 { 1 } else { 0 }; 16]
	}

	let mut prm = PRM::new(ContinuousSampler::new([-1.0, -1.0], [1.0, 1.0]),
						   DiscreteSampler::new(),
						   &m);

	prm.grow_graph(&[0.55, -0.8], goal, 0.05, 5.0, 5000, 100000).expect("graph not grown up to solution");
	prm.plan_qmdp().expect("general solution couldn't be found");
	let paths = prm.react_qmdp(&[0.55, -0.6], &vec![1.0/16.0; 16], 0.2).expect("impossible to extract policy");
	prm.print_summary();

	let mut full = m.clone();
	full.resize(5);
	full.draw_full_graph(&prm.graph);
//	full.draw_graph_from_root(&prm.get_policy_graph().unwrap());
//	full.draw_graph_for_world(&prm.graph, 0);

	for path in paths {
		full.draw_path(path);
	}
	full.save("results/test_plan_on_map4.pgm");
}

#[test]
fn test_plan_on_map1_2_goals() {
	let mut m = Map::open("data/map1_2_goals.pgm", [-1.0, -1.0], [1.0, 1.0]);
	m.add_zones("data/map1_2_goals_zone_ids.pgm");

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

	for path in paths {
		full.draw_path(path);
	}
	full.save("results/test_plan_on_map1_2_goals.pgm");
}

#[test]
#[should_panic]
fn test_when_grow_graph_doesnt_reach_goal() {
	let mut m = Map::open("data/map2.pgm", [-1.0, -1.0], [1.0, 1.0]);
	m.add_zones("data/map2_zone_ids.pgm");

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

/* N = nombre de mondes
* PLAN()
* Option 1/ Pour chaque monde, calculer la distance à l'objectif de chaque node (Value iteration)
* 	-> avantage : valeurs valables partout
*   -> long calcul offline - scalability?
*   d(n) = min( d(m, n) + d(m) for m in children(n) )
*
* Option 2/ Au runtime, calculer meilleur chemin pour chacun des mondes (e.g. A*) et prendre decision en pondérant chacunes des options 
*	-> avantage : étape offline moins longue
*   -> possible à l'exécution ? (assez rapide?)
*
* Option 3/ Monte Carlo -> N
* 
* Option 4/ N mondes: RRT start N fois
* 
*
* EXECUTION()
* - Robot a un belief state : probabilité d'être dans chaque monde
* - Deduit le chemin a suivre en cherchant en calculant l'espérance de la distance à l'objectif de chacun de ces enfants
*
* Belief state : [0.25, 0.25, 0.25, 0.25] 
* 
*
* QUESTIONS:
* - Quand arreter de faire croitre le graph et quand même avoir une solution our chaque monde??
*/

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
// TODO:
// - avoid copies
// - optimize nearest neighbor (avoid sqrt)
// - multithreaded dijkstra
