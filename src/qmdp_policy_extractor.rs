use itertools::{all, enumerate, izip, zip};

use crate::common::*;
use crate::nearest_neighbor::*;
use crate::sample_space::*;
use crate::map_io::*; // tests only
use crate::map_shelves_io::*; // tests only
use crate::prm::*;
use crate::prm_graph::*;
use crate::prm_reachability::*;
use bitvec::prelude::*;

struct QMdpPolicyExtractor <'a, F: PRMFuncs<N>, const N: usize> {
	graph: &'a PRMGraph<N>,
	kdtree: &'a KdTree<N>,
	n_worlds: &'a usize,
	conservative_reachability: &'a Reachability,
	fns: &'a F,
	cost_to_goals: Vec<Vec<f64>>,
}

impl <'a, F: PRMFuncs<N>, const N: usize> QMdpPolicyExtractor<'a, F, N> {
	pub fn plan_qmdp(&mut self) -> Result<(), &'static str> {
		// compute the cost to goals
		self.cost_to_goals = vec![Vec::new(); *self.n_worlds];
		for world in 0..*self.n_worlds {
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
		let mut paths : Vec<Vec<[f64; N]>> = vec![Vec::new(); *self.n_worlds];
		for world in 0..*self.n_worlds {
			paths[world] = common_path.clone();
			paths[world].extend(self.get_path(id, world));
		}

		Ok(paths)
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

	#[allow(clippy::style)]
	fn get_common_path(&self, start_id:usize, belief_state: &BeliefState, common_horizon: f64) -> Result<(Vec<[f64; N]>, usize), &'static str> {
		if belief_state.len() != *self.n_worlds {
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

	#[allow(clippy::style)]
	fn get_best_expected_child(&self, node_id: usize, belief_state: &BeliefState) -> (usize, f64) {
		let node = &self.graph.nodes[node_id]; 
		let mut best_child_id = 0;
		let mut smallest_expected_cost = std::f64::INFINITY;

		for child_edge in &node.children {
			let mut child_expected_cost = 0.0;

			for world in 0..*self.n_worlds {
				child_expected_cost += self.cost_to_goals[world][child_edge.id] * belief_state[world];
			}

			if child_expected_cost < smallest_expected_cost {
				best_child_id = child_edge.id;
				smallest_expected_cost = child_expected_cost;
			}
		}
		(best_child_id, smallest_expected_cost)
	}

	fn get_best_child(&self, node_id: usize, world: usize) -> usize {
		let node = &self.graph.nodes[node_id]; 
		let mut best_child_id = 0;
		let mut smaller_cost = std::f64::INFINITY;

		for child_edge in &node.children {
			if self.cost_to_goals[world][child_edge.id] < smaller_cost {
				smaller_cost = self.cost_to_goals[world][child_edge.id];
				best_child_id = child_edge.id;
			}
		}

		best_child_id
	}

	fn get_policy_graph(&self) -> Result<PRMGraph<N>, &'static str> {
		get_policy_graph(&self.graph, &self.cost_to_goals)
	}
}

#[cfg(test)]
mod tests {

use crate::belief_graph;

    use super::*;

#[test]
fn test_plan_on_map2_qmdp() {
	let mut m = Map::open("data/map2.pgm", [-1.0, -1.0], [1.0, 1.0]);
	m.add_zones("data/map2_zone_ids.pgm", 0.1);

	let goal = SquareGoal::new(vec![([0.55, 0.9], bitvec![1; 4])], 0.05);

	let mut prm = PRM::new(ContinuousSampler::new([-1.0, -1.0], [1.0, 1.0]),
						   DiscreteSampler::new(),
						   &m);

	prm.grow_graph(&[0.55, -0.8], &goal, 0.05, 5.0, 5000, 100000).expect("graph not grown up to solution");
	prm.print_summary();
	
	let mut qmdp = QMdpPolicyExtractor{
		graph: &prm.graph,
		kdtree: &prm.kdtree,
		n_worlds: &prm.n_worlds,
		conservative_reachability: &prm.conservative_reachability,
		fns: prm.fns,
		cost_to_goals: vec![]
	};
	qmdp.plan_qmdp().expect("general solution couldn't be found");
	let paths = qmdp.react_qmdp(&[0.55, -0.6], &vec![1.0/4.0; 4], 0.2).expect("impossible to extract policy");

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
	let mut m = MapShelfDomain::open("data/map1_2_goals.pgm", [-1.0, -1.0], [1.0, 1.0]);
	m.add_zones("data/map1_2_goals_zone_ids.pgm", 0.5);

	let goal = SquareGoal::new(vec![([0.68, -0.45], bitvec![1, 0]),
									([0.68, 0.38], bitvec![0, 1])], 0.05);

	let mut prm = PRM::new(ContinuousSampler::new([-1.0, -1.0], [1.0, 1.0]),
						DiscreteSampler::new(),
						&m);

	prm.grow_graph(&[-0.8, -0.8], &goal, 0.05, 5.0, 2000, 100000).expect("graph not grown up to solution");
	prm.print_summary();

	let mut qmdp = QMdpPolicyExtractor{
		graph: &prm.graph,
		kdtree: &prm.kdtree,
		n_worlds: &prm.n_worlds,
		conservative_reachability: &prm.conservative_reachability,
		fns: prm.fns,
		cost_to_goals: vec![]
	};

	qmdp.plan_qmdp().expect("general solution couldn't be found");
	let paths = qmdp.react_qmdp(&[-0.8, -0.8], &vec![0.5, 0.5], 0.2).expect("impossible to extract policy");

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
#[should_panic]
fn test_when_grow_graph_doesnt_reach_goal() {
	let mut m = Map::open("data/map2.pgm", [-1.0, -1.0], [1.0, 1.0]);
	m.add_zones("data/map2_zone_ids.pgm", 0.1);

	let goal = SquareGoal::new(vec![([0.55, 0.9], bitvec![1; 4])], 0.05);

	let mut prm = PRM::new(ContinuousSampler::new([-1.0, -1.0], [1.0, 1.0]),
						   DiscreteSampler::new(),
						   &m);

	assert_ne!(Ok(()), prm.grow_graph(&[0.55, -0.8], &goal, 0.05, 5.0, 300, 1000));

	let mut qmdp = QMdpPolicyExtractor{
		graph: &prm.graph,
		kdtree: &prm.kdtree,
		n_worlds: &prm.n_worlds,
		conservative_reachability: &prm.conservative_reachability,
		fns: prm.fns,
		cost_to_goals: vec![]
	};

	qmdp.plan_qmdp().unwrap(); // panics
}
}
