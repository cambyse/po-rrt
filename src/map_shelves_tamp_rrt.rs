use itertools::{all, enumerate, izip, merge, zip};

use crate::common::*;
use crate::nearest_neighbor::*;
use crate::sample_space::*;
use crate::map_shelves_io::*;
use crate::rrt::*;
use bitvec::prelude::*;
use priority_queue::PriorityQueue;

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
	pub path_to_pickup_cost: f64
}

pub struct SearchTree {
    pub nodes: Vec<SearchNode>,
}

impl SearchTree {
	pub fn add_node(&mut self, parent_id: usize, target_zone_id: usize, remaining_zones: &[usize],
					start_state: [f64; 2], observation_state: [f64; 2], pickup_state: [f64; 2],
				    path_to_observation: Vec<[f64;2]>, path_to_pickup: Vec<[f64;2]>,
					path_to_start_cost: f64, path_to_observation_cost:f64, path_to_pickup_cost: f64) -> usize {
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
			path_to_pickup_cost
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

	pub fn plan(&mut self, &start: &[f64; 2], _goal: &impl GoalFuncs<2>, _max_step: f64, _search_radius: f64, _n_iter_min: usize, _n_iter_max: usize) -> Result<Vec<Vec<[f64; 2]>>, &'static str> {
		let mut solution_nodes: Vec<SearchNode> = vec![];
		let mut q = PriorityQueue::new();

		let root_node = SearchNode{
			id: 0,
			target_zone_id: Some(0),
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
			path_to_pickup_cost: 0.0
		};

		q.push(root_node.id, Priority{prio: root_node.path_to_start_cost});

		let mut search_tree = SearchTree{
			nodes: vec![root_node]
		};

		// rrt 
		let mut rrt = RRT::new(ContinuousSampler::new([-1.0, -1.0], [1.0, 1.0]), Funcs{m:&self.map_shelves_domain});

		let mut it = 0;
		while !q.is_empty() {
			it+=1;
			let (u_id, _) = q.pop().unwrap();
			let u = search_tree.nodes[u_id].clone();

			for target_zone_id in &u.remaining_zones {
				let mut remaining_zones = u.remaining_zones.clone();
				remaining_zones.retain(|zone_id| zone_id != target_zone_id);
				
				// query motion planner
				// piece 1: go to target observe zone
				let observation_goal = ObservationGoal{m: &self.map_shelves_domain, zone_id: *target_zone_id};
				
				let (observation_planning_result, _) = rrt.plan(u.observation_state, &observation_goal, 0.1, 5.0, 5000);
				let (observation_path, observation_path_cost) = observation_planning_result.expect("no observation path found!");
				let v_observation_state = observation_path.last().unwrap().clone();

				// piece 2: object is here: plan to reach goal corresponding to 
				let zone_position = self.map_shelves_domain.get_zone_positions()[*target_zone_id];
				let pickup_goal = SquareGoal::new(vec![(zone_position, bitvec![1])], 0.05);

				let (pickup_planning_result, _) = rrt.plan(v_observation_state, &pickup_goal, 0.1, 5.0, 5000);
				let (pickup_path, pickup_path_cost) = pickup_planning_result.expect("no pickup path found!");
				let v_pickup_state = pickup_path.last().unwrap().clone();
				
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
					pickup_path_cost
				);

				let v = &search_tree.nodes[v_id];

				// addonly if not too bad
				q.push(v_id, Priority{prio: v.path_to_start_cost});
			}

			// save leaf node
			if u.remaining_zones.is_empty() {
				solution_nodes.push(u);
				break;
			}
		}

		println!("it:{}, nodes:{}", it, search_tree.nodes.len());
		println!("n leafs:{}", search_tree.nodes.iter().filter(|&n| n.remaining_zones.is_empty() ).count());

		let path_tree = self.reconstruct_path_tree(solution_nodes.last().unwrap(), &search_tree);

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

	let goal = SquareGoal::new(vec![([0.55, 0.9], bitvec![1; 4])], 0.05);

	let mut tamp_rrt = MapShelfDomainTampRRT::new(ContinuousSampler::new([-1.0, -1.0], [1.0, 1.0]), &m);			
	let path_tree = tamp_rrt.plan(&[0.0, -0.8], &goal, 0.05, 5.0, 5000, 100000);
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

	let goal = SquareGoal::new(vec![([-0.9,-0.5], bitvec![1, 0, 0, 0, 0, 0]),
									([-0.9, 0.5], bitvec![0, 1, 0, 0, 0, 0]),
									([-0.5, 0.9], bitvec![0, 0, 1, 0, 0, 0]),
									([ 0.5, 0.9], bitvec![0, 0, 0, 1, 0, 0]),
									([ 0.9, 0.5], bitvec![0, 0, 0, 0, 1, 0]),
									([ 0.9,-0.5], bitvec![0, 0, 0, 0, 0, 1])],
										0.05);

	let mut tamp_rrt = MapShelfDomainTampRRT::new(ContinuousSampler::new([-1.0, -1.0], [1.0, 1.0]), &m);			
	let path_tree = tamp_rrt.plan(&[0.0, -0.8], &goal, 0.05, 5.0, 5000, 100000);
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
