use bitvec::vec;
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

pub enum TampSearch {
	AStar,
	BranchAndBound,
	BranchAndBoundMultipleViewPoints
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
	discrete_sampler: DiscreteSampler,
	pub map_shelves_domain: &'a MapShelfDomain,
	pub kdtree: KdTree<2>,
	pub n_worlds: usize,
	pub goal_radius: f64,
	n_it: usize,
}

impl<'a> MapShelfDomainTampRRT<'a> {
	pub fn new(continuous_sampler: ContinuousSampler<2>, discrete_sampler: DiscreteSampler, map_shelves_domain: &'a MapShelfDomain, goal_radius: f64) -> Self {
		Self { continuous_sampler,
			   discrete_sampler,
			   map_shelves_domain, 
			   kdtree: KdTree::new([0.0; 2]),
			   n_worlds: map_shelves_domain.n_zones(), 
			   goal_radius,
			   n_it: 0}
	}

	pub fn plan(&mut self, start: &[f64; 2], initial_belief_state: &BeliefState, max_step: f64, search_radius: f64, n_iter_min: usize, n_iter_max: usize, search_method: TampSearch) -> Result<Policy<2>, &'static str> {
		match search_method {
			TampSearch::AStar => self.plan_astar(start, initial_belief_state, max_step, search_radius, n_iter_min, n_iter_max),
			TampSearch::BranchAndBound => self.plan_branch_bound(start, initial_belief_state, max_step, search_radius, n_iter_min, n_iter_max),
			TampSearch::BranchAndBoundMultipleViewPoints => self.plan_branch_bound_multiple_viewpoints(start, initial_belief_state, max_step, search_radius, n_iter_min, n_iter_max),
		}
	}	

	fn plan_branch_bound(&mut self, &start: &[f64; 2], initial_belief_state: &BeliefState, max_step: f64, search_radius: f64, n_iter_min: usize, n_iter_max: usize) -> Result<Policy<2>, &'static str> {
		let mut solution_nodes = BTreeMap::new();
		let mut q = vec![];

		check_belief_state(initial_belief_state);

		let root_node = SearchNode{
			id: 0,
			target_zone_id: None,
			parent: None,
			children: Vec::new(),
			remaining_zones: shuffled(&(0..self.map_shelves_domain.n_zones()).collect(), &mut self.discrete_sampler),
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

		q.push(root_node.id);

		let mut search_tree = SearchTree{
			nodes: vec![root_node]
		};

		// rrt 
		let checker = Funcs{m:&self.map_shelves_domain};
		let mut rrt = RRT::new(self.continuous_sampler.clone(), &checker);

		let mut best_expected_cost = std::f64::INFINITY;
		let mut it = 0;
		while !q.is_empty() {
			it+=1;

			println!("iteration:{}", it);

			let u_id = q.pop().unwrap();
			let u = search_tree.nodes[u_id].clone();

			for target_zone_id in &u.remaining_zones {
				let mut remaining_zones = shuffled(&u.remaining_zones, &mut self.discrete_sampler);
				remaining_zones.retain(|zone_id| zone_id != target_zone_id);

				// compute belief state and reaching probability
				let mut v_belief_state = u.belief_state.clone();
				if let Some(u_target_zone_id) = u.target_zone_id {
					v_belief_state[u_target_zone_id] = 0.0; // we are at thisstage on theskeleton if the object was not there
				}
				v_belief_state = normalize_belief(&v_belief_state);
				let reaching_probability = u.reaching_probability * transition_probability(&u.belief_state, &v_belief_state);

				// Query motion planner
				// piece 1: go to target observe zone
				let observation_goal = ObservationGoal{m: &self.map_shelves_domain, zone_id: *target_zone_id};
				
				let (observation_planning_result, _) = rrt.plan(u.observation_state, &observation_goal, max_step, search_radius, n_iter_min, n_iter_max);
				let (observation_path, observation_path_cost) = observation_planning_result.expect("no observation path found!");
				let v_observation_state = *observation_path.last().unwrap();

				// piece 2: object is here: plan to reach goal corresponding to 
				let zone_position = self.map_shelves_domain.get_zone_positions()[*target_zone_id];
				let pickup_goal = SquareGoal::new(vec![(zone_position, bitvec![1])], self.goal_radius);

				let (pickup_planning_result, _) = rrt.plan(v_observation_state, &pickup_goal, max_step, search_radius, n_iter_min, n_iter_max);
				let (pickup_path, pickup_path_cost) = pickup_planning_result.expect("no pickup path found!");
				let v_pickup_state = *pickup_path.last().unwrap();

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
				
				// addonly prune if lower bound of costs greater than solution already found
				if expected_cost < best_expected_cost {
					q.push(v_id);
				}
				else {
					println!("prune!");
				}
			}

			// save leaf node
			if u.remaining_zones.is_empty() {
				if u.expected_cost < best_expected_cost {
					println!("found improving solution! with cost:{}", u.expected_cost);
					best_expected_cost = u.expected_cost;
				}
				solution_nodes.insert(OrderedFloat(u.expected_cost), u);
			}
		}

		println!("it:{}, nodes:{}", it, search_tree.nodes.len());
		println!("n leafs:{}", search_tree.nodes.iter().filter(|&n| n.remaining_zones.is_empty() ).count());

		let (best_expected_cost, best_final_node) = solution_nodes.first_key_value().expect("No solution found!");
		let policy = self.build_policy(&best_final_node, &search_tree);

		println!("best expected costs before shortcut:{}", best_expected_cost);
		println!("policy expected cost after shortcut:{}", policy.expected_costs);

		Ok(policy)
	}

	fn plan_branch_bound_multiple_viewpoints(&mut self, &start: &[f64; 2], initial_belief_state: &BeliefState, max_step: f64, search_radius: f64, n_iter_min: usize, n_iter_max: usize) -> Result<Policy<2>, &'static str> {
		let mut solution_nodes = BTreeMap::new();
		let mut q = vec![];

		check_belief_state(initial_belief_state);

		let root_node = SearchNode{
			id: 0,
			target_zone_id: None,
			parent: None,
			children: Vec::new(),
			remaining_zones: shuffled(&(0..self.map_shelves_domain.n_zones()).collect(), &mut self.discrete_sampler),
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

		q.push(root_node.id);

		let mut search_tree = SearchTree{
			nodes: vec![root_node]
		};

		// rrt 
		let checker = Funcs{m:&self.map_shelves_domain};
		let mut rrt = RRT::new(self.continuous_sampler.clone(), &checker);

		let mut best_expected_cost = std::f64::INFINITY;
		let mut it = 0;
		let mut rrt_queries_it = 0;
		while !q.is_empty() {
			it+=1;

			println!("iteration:{}", it);

			let u_id = q.pop().unwrap();
			let u = search_tree.nodes[u_id].clone();

			for target_zone_id in &u.remaining_zones {
				let mut remaining_zones = shuffled(&u.remaining_zones, &mut self.discrete_sampler);
				remaining_zones.retain(|zone_id| zone_id != target_zone_id);

				// compute belief state and reaching probability
				let mut v_belief_state = u.belief_state.clone();
				if let Some(u_target_zone_id) = u.target_zone_id {
					v_belief_state[u_target_zone_id] = 0.0; // we are at thisstage on theskeleton if the object was not there
				}
				v_belief_state = normalize_belief(&v_belief_state);
				let reaching_probability = u.reaching_probability * transition_probability(&u.belief_state, &v_belief_state);

				// Query motion planner
				// piece 1: go to target observe zone
				let observation_goal = ObservationGoal{m: &self.map_shelves_domain, zone_id: *target_zone_id};
				
				let (observation_planning_results, _) = rrt.plan_several(u.observation_state, &observation_goal, max_step, search_radius, n_iter_min, n_iter_max);
				rrt_queries_it = rrt_queries_it+1;

				println!("number of paths to observations:{}", observation_planning_results.len());
				for (observation_path, observation_path_cost) in observation_planning_results {

					let v_observation_state = *observation_path.last().unwrap();

					// piece 2: object is here: plan to reach goal corresponding to 
					let zone_position = self.map_shelves_domain.get_zone_positions()[*target_zone_id];
					let pickup_goal = SquareGoal::new(vec![(zone_position, bitvec![1])], self.goal_radius);

					let (pickup_planning_result, _) = rrt.plan(v_observation_state, &pickup_goal, max_step, search_radius, n_iter_min, n_iter_max);
					let (pickup_path, pickup_path_cost) = pickup_planning_result.expect("no pickup path found!");
					let v_pickup_state = *pickup_path.last().unwrap();
					rrt_queries_it = rrt_queries_it+1;

					//println!("rrt_queries_it:{}", rrt_queries_it);

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
					
					// addonly prune if lower bound of costs greater than solution already found
					if expected_cost < best_expected_cost {
						q.push(v_id);
					}
					else {
						println!("prune!");
					}
				}
			}

			// save leaf node
			if u.remaining_zones.is_empty() {
				if u.expected_cost < best_expected_cost {
					println!("found improving solution! with cost:{}", u.expected_cost);
					best_expected_cost = u.expected_cost;
				}
				solution_nodes.insert(OrderedFloat(u.expected_cost), u);
			}
		}

		println!("it:{}, nodes:{}", it, search_tree.nodes.len());
		println!("n leafs:{}", search_tree.nodes.iter().filter(|&n| n.remaining_zones.is_empty() ).count());

		let (best_expected_cost, best_final_node) = solution_nodes.first_key_value().expect("No solution found!");
		let policy = self.build_policy(&best_final_node, &search_tree);

		println!("best expected costs before shortcut:{}", best_expected_cost);
		println!("policy expected cost after shortcut:{}", policy.expected_costs);

		Ok(policy)
	}

	fn plan_astar(&mut self, &start: &[f64; 2], initial_belief_state: &BeliefState, max_step: f64, search_radius: f64, n_iter_min: usize, n_iter_max: usize) -> Result<Policy<2>, &'static str> {
		let mut solution_nodes = BTreeMap::new();
		let mut q = PriorityQueue::new();

		check_belief_state(initial_belief_state);

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
		let checker = Funcs{m:&self.map_shelves_domain};
		let mut rrt = RRT::new(self.continuous_sampler.clone(), &checker);

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

				// Query motion planner
				// piece 1: go to target observe zone
				let observation_goal = ObservationGoal{m: &self.map_shelves_domain, zone_id: *target_zone_id};
				
				let (observation_planning_result, _) = rrt.plan(u.observation_state, &observation_goal, max_step, search_radius, n_iter_min, n_iter_max);
				let (observation_path, observation_path_cost) = observation_planning_result.expect("no observation path found!");
				let v_observation_state = *observation_path.last().unwrap();

				// piece 2: object is here: plan to reach goal corresponding to 
				let zone_position = self.map_shelves_domain.get_zone_positions()[*target_zone_id];
				let pickup_goal = SquareGoal::new(vec![(zone_position, bitvec![1])], self.goal_radius);

				let (pickup_planning_result, _) = rrt.plan(v_observation_state, &pickup_goal, max_step, search_radius, n_iter_min, n_iter_max);
				let (pickup_path, pickup_path_cost) = pickup_planning_result.expect("no pickup path found!");
				let v_pickup_state = *pickup_path.last().unwrap();

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

				// estimate future cost: Admissible heuristic!
				let heuristic_and_probability_per_zone: Vec<(f64, f64)> = v.remaining_zones.iter()
				.map(|target_zone_id| (target_zone_id, self.map_shelves_domain.get_zone_positions()[*target_zone_id]))
				.map(|(target_zone_id, zone_position)| (target_zone_id, checker.cost_evaluator(&v.observation_state, &zone_position)))
				.map(|(target_zone_id, heuristic_to_zone)| (heuristic_to_zone, v.belief_state[*target_zone_id])).collect();

				let heuristic_future_expected_costs = heuristic_and_probability_per_zone.iter().fold(0.0, |h, (heuristic_to_zone, p)| h + p * heuristic_to_zone );

				let overall_expected_costs = v.expected_cost + reaching_probability * heuristic_future_expected_costs;

				q.push(v_id, Priority{prio: overall_expected_costs});
			}

			// save leaf node
			if u.remaining_zones.is_empty() {
				println!("found improving solution! with cost:{}", u.expected_cost);
				solution_nodes.insert(OrderedFloat(u.expected_cost), u);
				break;
			}
		}

		println!("it:{}, nodes:{}", it, search_tree.nodes.len());
		println!("n leafs:{}", search_tree.nodes.iter().filter(|&n| n.remaining_zones.is_empty() ).count());

		let (best_expected_cost, best_final_node) = solution_nodes.first_key_value().expect("No solution found!");
		let policy = self.build_policy(&best_final_node, &search_tree);

		println!("best expected costs before shortcut:{}", best_expected_cost);
		println!("policy expected cost after shortcut:{}", policy.expected_costs);

		Ok(policy)
	}

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
	}
}

#[cfg(test)]
mod tests {

use super::*;

#[test]
fn test_shuffle() {
	shuffled(&vec![0, 1, 2, 3, 4], &mut DiscreteSampler::new());
}

#[test]
fn test_plan_on_map2_astar() {
	let mut m = MapShelfDomain::open("data/map1_2_goals.pgm", [-1.0, -1.0], [1.0, 1.0]);
	m.add_zones("data/map1_2_goals_zone_ids.pgm", 0.5);

	let mut tamp_rrt = MapShelfDomainTampRRT::new(ContinuousSampler::new([-1.0, -1.0], [1.0, 1.0]), DiscreteSampler::new(), &m, 0.05);		
	
	let initial_belief_state = vec![1.0/2.0; 2];
	let policy = tamp_rrt.plan_astar(&[-0.9, 0.0], &initial_belief_state, 0.1, 2.0, 2500, 10000);
	let policy = policy.expect("nopath tree found!");

	let mut m2 = m.clone();
	m2.resize(5);
	m2.draw_zones_observability();
	m2.draw_policy(&policy);
	m2.save("results/test_map1_2_goals_tamp_rrt");
}

#[test]
fn test_plan_on_map2_bb() {
	let mut m = MapShelfDomain::open("data/map1_2_goals.pgm", [-1.0, -1.0], [1.0, 1.0]);
	m.add_zones("data/map1_2_goals_zone_ids.pgm", 0.5);

	let mut tamp_rrt = MapShelfDomainTampRRT::new(ContinuousSampler::new([-1.0, -1.0], [1.0, 1.0]), DiscreteSampler::new(), &m, 0.05);		
	
	let initial_belief_state = vec![1.0/2.0; 2];
	let policy = tamp_rrt.plan_branch_bound(&[-0.9, 0.0], &initial_belief_state, 0.1, 2.0, 2500, 10000);
	let policy = policy.expect("nopath tree found!");

	let mut m2 = m.clone();
	m2.resize(5);
	m2.draw_zones_observability();
	m2.draw_policy(&policy);
	m2.save("results/test_map1_2_goals_tamp_rrt_bb");
}

#[test]
fn test_plan_on_map2_bb_multiple() {
	let mut m = MapShelfDomain::open("data/map1_2_goals.pgm", [-1.0, -1.0], [1.0, 1.0]);
	m.add_zones("data/map1_2_goals_zone_ids.pgm", 0.5);

	let mut tamp_rrt = MapShelfDomainTampRRT::new(ContinuousSampler::new([-1.0, -1.0], [1.0, 1.0]), DiscreteSampler::new(), &m, 0.05);		
	
	let initial_belief_state = vec![1.0/2.0; 2];
	let policy = tamp_rrt.plan_branch_bound_multiple_viewpoints(&[-0.9, 0.0], &initial_belief_state, 0.1, 2.0, 2500, 10000);
	let policy = policy.expect("nopath tree found!");

	let mut m2 = m.clone();
	m2.resize(5);
	m2.draw_zones_observability();
	m2.draw_policy(&policy);
	m2.save("results/test_map1_2_goals_tamp_rrt_bb_multiple");
}

#[test]
fn test_plan_on_map7_plan_astar() {
	let mut m = MapShelfDomain::open("data/map7.pgm", [-1.0, -1.0], [1.0, 1.0]);
	m.add_zones("data/map7_6_goals_zone_ids.pgm", 0.5);

	let mut tamp_rrt = MapShelfDomainTampRRT::new(ContinuousSampler::new([-1.0, -1.0], [1.0, 1.0]), DiscreteSampler::new(), &m, 0.05);	
	
	let initial_belief_state = vec![1.0/6.0; 6];
	let policy = tamp_rrt.plan_astar(&[0.0, -0.8], &initial_belief_state, 0.1, 2.0, 2500, 10000);
	let policy = policy.expect("nopath tree found!");

	let mut m2 = m.clone();
	m2.resize(5);
	m2.draw_zones_observability();
	m2.draw_policy(&policy);
	m2.save("results/map7/test_map7_tamp_rrt");
}

}