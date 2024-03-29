#![allow(dead_code, unused_imports)]

use crate::common::*;
use crate ::pto_graph::*;
use crate ::pto::*;
use crate ::sample_space::*;
use crate ::pto_policy_refiner::*;
use bitvec::prelude::*;
use ptr::{null, null_mut};
use std::convert::TryInto;
use std::iter::FromIterator;
use std::ptr;
use std::usize;
use std::time::Instant;


pub type StateValidityCallbackType = extern fn(*const f64, usize) -> i64;
pub type TransitionValidityCallbackType = extern fn(from_raw: *const f64, usize, to_raw: *const f64, usize) -> i64;
pub type CostEvaluatorCallbackType = extern fn(from_raw: *const f64, usize, to_raw: *const f64, usize) -> f64;
pub type ObserverCallbackType = extern fn(from_raw: *const f64, usize, belief_state_raw: *const f64, usize,
										  belief_ids_raw: *mut *mut *mut usize, *mut usize);
pub type GoalCallbackType = extern fn(*const f64, usize, *mut bool, usize) -> bool;
pub type GoalExampleCallbackType = extern fn(usize, *mut f64, usize);

//////////////////////////////////////////////////
////////   CPlanningProblem
//////////////////////////////////////////////////
#[repr(C)]
pub struct CPlanningProblem{
	// input
	state_dim: usize,
	n_worlds: usize,
	low: (*mut f64, usize),
	up: (*mut f64, usize),
	world_validities: (*mut *mut usize, usize),
	state_validity_callback: Option<StateValidityCallbackType>,
	transition_validity_callback: Option<TransitionValidityCallbackType>,
	cost_evaluator_callback: Option<CostEvaluatorCallbackType>,
	observer_callback: Option<ObserverCallbackType>,
	start_belief_state: (*mut f64, usize),
	reachable_belief_states: (*mut *mut f64, usize),
	// goal
	goal_callback: Option<GoalCallbackType>,
	goal_example_callback: Option<GoalExampleCallbackType>,
	// search parameters
	n_iterations_min: usize,
	n_iterations_max: usize,
	max_step: f64,
	search_radius: f64,
	refine_iterations: usize,
	// output
	paths: Vec<Vec<Vec<f64>>>,
	paths_lengths: Vec<usize>,
	expected_costs: f64,
	n_iterations: usize,
	graph_growth_s: f64,
	belief_space_expansion_s: f64,
	dynamic_programming_s: f64,
	refinement_s: f64,
	total_s: f64
}

#[no_mangle]
pub extern "C" fn new_planning_problem() -> Box<CPlanningProblem> {
    Box::new(
		CPlanningProblem{
			// input
			state_dim: 0,
			n_worlds: 0,
			low: (ptr::null_mut(), 0),
			up: (ptr::null_mut(), 0),
			world_validities: (ptr::null_mut(), 0),
			state_validity_callback: None,
			transition_validity_callback : None,
			cost_evaluator_callback: None,
			observer_callback: None,
			start_belief_state: (ptr::null_mut(), 0),
			reachable_belief_states: (ptr::null_mut(), 0),
			// goal
			goal_callback: None,
			goal_example_callback: None,
			// parameters
			n_iterations_min: 0,
			n_iterations_max: 0,
			max_step: 0.0,
			search_radius: 0.0,
			refine_iterations: 0,
			// output
			paths: Vec::new(),
			paths_lengths: Vec::new(),
			expected_costs: 0.0,
			n_iterations: 0,
			graph_growth_s: 0.0,
			belief_space_expansion_s: 0.0,
			dynamic_programming_s: 0.0,
			refinement_s: 0.0,
			total_s: 0.0
		}
	)
}

#[no_mangle]
pub extern "C" fn delete_planning_problem(_: Option<Box<CPlanningProblem>>) {}

#[no_mangle]
pub extern "C" fn set_problem_dimensions(planning_problem: *mut CPlanningProblem, state_dim: usize, n_worlds: usize) {
	unsafe {
		(*planning_problem).state_dim = state_dim;
		(*planning_problem).n_worlds = n_worlds;
	}
}

#[no_mangle]
pub extern "C" fn set_lower_sampling_bound(planning_problem: *mut CPlanningProblem, low_raw: *mut f64, low_size: usize) {
	unsafe {
		assert_eq!((*planning_problem).state_dim, low_size);

		(*planning_problem).low = (low_raw, low_size);
	}
}

#[no_mangle]
pub extern "C" fn set_upper_sampling_bound(planning_problem: *mut CPlanningProblem, up_raw: *mut f64, up_size: usize) {
	unsafe {
		assert_eq!((*planning_problem).state_dim, up_size);

		(*planning_problem).up = (up_raw, up_size);
	}
}

#[no_mangle]
pub extern "C" fn set_world_validities(planning_problem: *mut CPlanningProblem, validities_raw: *mut *mut usize, validities_size: usize) {
	unsafe {
		(*planning_problem).world_validities = (validities_raw, validities_size);
	}
}

#[no_mangle]
pub extern "C" fn set_state_validity_callback(planning_problem: *mut CPlanningProblem, callback: StateValidityCallbackType) {
	unsafe {
		(*planning_problem).state_validity_callback = Some(callback);
	}
}

#[no_mangle]
pub extern "C" fn set_transition_validity_callback(planning_problem: *mut CPlanningProblem, callback: TransitionValidityCallbackType) {
	unsafe {
		(*planning_problem).transition_validity_callback = Some(callback);
	}
}

#[no_mangle]
pub extern "C" fn set_cost_evaluator_callback(planning_problem: *mut CPlanningProblem, callback: CostEvaluatorCallbackType) {
	unsafe {
		(*planning_problem).cost_evaluator_callback = Some(callback);
	}
}


#[no_mangle]
pub extern "C" fn set_observer_callback(planning_problem: *mut CPlanningProblem, callback: ObserverCallbackType) {
	unsafe {
		(*planning_problem).observer_callback = Some(callback);
	}
}

#[no_mangle]
pub extern "C" fn set_start_belief_state(planning_problem: *mut CPlanningProblem, start_belief_state_raw: *mut f64, start_belief_state_size :usize, reachable_belief_states_raw: *mut *mut f64, reachable_belief_states_size : usize) {
	unsafe {
		assert_eq!((*planning_problem).n_worlds, start_belief_state_size);

		(*planning_problem).start_belief_state = (start_belief_state_raw, start_belief_state_size);
		(*planning_problem).reachable_belief_states = (reachable_belief_states_raw, reachable_belief_states_size);
	}
}

#[no_mangle]
pub extern "C" fn set_goal_callback(planning_problem: *mut CPlanningProblem, callback: GoalCallbackType) {
	unsafe {
		(*planning_problem).goal_callback = Some(callback);
	}
}

#[no_mangle]
pub extern "C" fn set_goal_example_callback(planning_problem: *mut CPlanningProblem, callback: GoalExampleCallbackType) {
	unsafe {
		(*planning_problem).goal_example_callback = Some(callback);
	}
}

#[no_mangle]
pub extern "C" fn set_search_parameters(planning_problem: *mut CPlanningProblem, n_iterations_min: usize, n_iterations_max: usize, max_step: f64, search_radius: f64) {
	unsafe {
		(*planning_problem).n_iterations_min = n_iterations_min;
		(*planning_problem).n_iterations_max = n_iterations_max;
		(*planning_problem).max_step = max_step;
		(*planning_problem).search_radius = search_radius;
	}
}

#[no_mangle]
pub extern "C" fn set_refine_parameters(planning_problem: *mut CPlanningProblem, refine_iterations: usize) {
	unsafe {
		(*planning_problem).refine_iterations = refine_iterations;
	}
}

macro_rules! plan_inner {
    ($N:expr, $planning_problem:expr, $start:expr) => {
		let start_time = Instant::now();

		let fns = PTOFuncsAdapter::<$N>::new($planning_problem);
		let goal = GoalAdapter::<$N>::new($planning_problem);
		let mut prm = PTO::new(ContinuousSampler::new_true_random(fns.low, fns.up), DiscreteSampler::new_true_random(), &fns);
		prm.grow_graph(&$start.try_into().unwrap(), &goal, (*$planning_problem).max_step, (*$planning_problem).search_radius, (*$planning_problem).n_iterations_min, (*$planning_problem).n_iterations_max).expect("graph not grown up to solution");
		prm.print_summary();
		let policy = prm.plan_belief_space(&fns.start_belief_state);
		let mut policy_refiner = PTOPolicyRefiner::new(&policy, &fns, &prm.belief_graph);
		let (policy, _) = policy_refiner.refine_solution(RefinmentStrategy::PartialShortCut((*$planning_problem).refine_iterations));

		save_planning_metrics($planning_problem, prm.n_it, prm.graph_growth_s, prm.belief_space_expansion_s, prm.dynamic_programming_s, policy_refiner.refinement_s, start_time.elapsed().as_secs_f64());
		save_paths($planning_problem, &policy);
    };
}

#[no_mangle]
pub extern "C" fn plan(planning_problem: *mut CPlanningProblem, start_raw: *mut f64, start_size : usize) {
    unsafe {
		assert_eq!(start_size, (*planning_problem).state_dim);

		let start: Vec<f64> = Vec::from_raw_parts(start_raw, start_size, start_size);

		match (*planning_problem).state_dim {
			2 => {plan_inner!(2, planning_problem, start);}, 
			3 => {plan_inner!(3, planning_problem, start);},
			7 => {plan_inner!(7, planning_problem, start);},
			9 => {plan_inner!(9, planning_problem, start);},
			_ => panic!("case not yet handled!")
		};
	}
}

#[no_mangle]
pub extern "C" fn get_planning_metrics(planning_problem: *mut CPlanningProblem, n_iterations: *mut usize, graph_growth_s: *mut f64, belief_space_expansion_s: *mut f64, dynamic_programming_s: *mut f64, refinement_s: *mut f64, total_s: *mut f64) {
	unsafe {
		*n_iterations = (*planning_problem).n_iterations;
		*graph_growth_s = (*planning_problem).graph_growth_s;
		*belief_space_expansion_s = (*planning_problem).belief_space_expansion_s;
		*dynamic_programming_s = (*planning_problem).dynamic_programming_s;
		*refinement_s = (*planning_problem).refinement_s;
		*total_s = (*planning_problem).total_s;
	}
}

#[no_mangle]
pub extern "C" fn get_paths_info(planning_problem: *mut CPlanningProblem, number_of_paths: *mut usize, path_lengths: *mut *mut usize, expected_cost: *mut f64) {
	unsafe {
		*number_of_paths = (*planning_problem).paths.len();
		*path_lengths = (*planning_problem).paths_lengths.as_mut_ptr() as *mut usize;
		*expected_cost = (*planning_problem).expected_costs;
	}
}

#[no_mangle]
pub extern "C" fn get_paths_variable(planning_problem: *mut CPlanningProblem, path_id: usize, state_id: usize, c_state: *mut *mut f64, state_size: *mut usize) {
	unsafe {
		(*c_state) = (*planning_problem).paths[path_id][state_id].as_mut_ptr();
		(*state_size) = (*planning_problem).paths[path_id][state_id].len();
	}
}

pub fn save_planning_metrics(planning_problem: *mut CPlanningProblem, n_it: usize, graph_growth_s: f64, belief_space_expansion_s: f64, dynamic_programming_s: f64, refinement_s: f64, total_s: f64) {
	unsafe{
		(*planning_problem).n_iterations = n_it;
		(*planning_problem).graph_growth_s = graph_growth_s;
		(*planning_problem).belief_space_expansion_s = belief_space_expansion_s;
		(*planning_problem).dynamic_programming_s = dynamic_programming_s;
		(*planning_problem).refinement_s = refinement_s;
		(*planning_problem).total_s = total_s;
	}
}

pub fn save_paths<const N: usize>(planning_problem: *mut CPlanningProblem, policy: &Policy<N>) {
	let mut paths: Vec<Vec<Vec<f64>>> = Vec::new();
	let mut paths_lengths: Vec<usize> = Vec::new();

	for &leaf_id in &policy.leafs {
		let mut path : Vec<Vec<f64>> = Vec::new();
		let mut current_id = leaf_id;
		let mut current = &policy.nodes[current_id];
		
		path.push(current.state.try_into().unwrap());

		while current.parent.is_some() {			
			current_id = current.parent.unwrap();
			current = &policy.nodes[current_id];

			path.push(current.state.try_into().unwrap());
		}

		path.reverse();

		paths_lengths.push(path.len());
		paths.push(path);
	}

	unsafe{
		//println!("paths:{:?}, expected costs:{}", paths, policy.expected_costs);
		(*planning_problem).paths = paths;
		(*planning_problem).paths_lengths = paths_lengths;
		(*planning_problem).expected_costs = policy.expected_costs;
	}
}

//////////////////////////////////////////////////
////////   PTOFuncsAdapter
//////////////////////////////////////////////////

pub struct PTOFuncsAdapter<const N: usize> {
	planning_problem: *const CPlanningProblem,
	low: [f64; N],
	up: [f64; N],
	world_validities: Vec<WorldMask>,
	reachable_belief_states: Vec<BeliefState>,
	start_belief_state: Vec<f64>
}

impl<const N: usize> PTOFuncsAdapter<N> {
	pub fn new(planning_problem: *const CPlanningProblem) -> Self {
		unsafe {
			assert!(N==(*planning_problem).low.1, "N:{}, low size:{}", N, (*planning_problem).low.1);
			assert!(N==(*planning_problem).up.1, "N:{}, up size:{}", N, (*planning_problem).up.1);
		}

		let low: [f64; N];
		unsafe {
			let low_vec = Vec::from_raw_parts((*planning_problem).low.0, (*planning_problem).low.1, (*planning_problem).low.1);
			low = low_vec.try_into().unwrap();
		}

		let up: [f64; N];
		unsafe {
			let up_vec = Vec::from_raw_parts((*planning_problem).up.0, (*planning_problem).up.1, (*planning_problem).up.1);
			up = up_vec.try_into().unwrap();
		}
		
		// reachable_belief_states
		let mut reachable_belief_states: Vec<BeliefState> = Vec::new();
		
		unsafe {
			let belief_states_vec = Vec::from_raw_parts((*planning_problem).reachable_belief_states.0, (*planning_problem).reachable_belief_states.1, (*planning_problem).reachable_belief_states.1);
			for belief_state_raw in belief_states_vec {
				let belief_state_vec = Vec::from_raw_parts(belief_state_raw, (*planning_problem).n_worlds, (*planning_problem).n_worlds);
				reachable_belief_states.push(belief_state_vec);
			}
		}

		// world_validities
		let mut world_validities = Vec::<WorldMask>::new();

		unsafe {
			let validities_vec = Vec::from_raw_parts((*planning_problem).world_validities.0, (*planning_problem).world_validities.1, (*planning_problem).world_validities.1);
			for validity_raw in validities_vec {
				let validity_vec = Vec::from_raw_parts(validity_raw, (*planning_problem).n_worlds, (*planning_problem).n_worlds);
				let mut validity_bit_vec = bitvec![0; (*planning_problem).n_worlds];

				for i in 0..validity_vec.len() {
					validity_bit_vec.set(i, validity_vec[i] > 0);
				}

				world_validities.push(validity_bit_vec);
			}
		}

		// start belief state
		let start_belief_state: Vec<f64>;
		unsafe {
			start_belief_state = Vec::from_raw_parts((*planning_problem).start_belief_state.0, (*planning_problem).start_belief_state.1, (*planning_problem).start_belief_state.1);
		}

		PTOFuncsAdapter{
			planning_problem,
			low,
			up,
			world_validities,
			reachable_belief_states,
			start_belief_state
		}
	}
}

impl<const N: usize> PTOFuncs<N> for PTOFuncsAdapter<N> {
	fn n_worlds(&self) -> usize {
		unsafe {
			(*self.planning_problem).n_worlds
		}
	}

	fn state_validity(&self, state: &[f64; N]) -> Option<usize> {
		unsafe {
			let validity_id: i64 = (*self.planning_problem).state_validity_callback.unwrap()(state.as_ptr(), state.len());
			if validity_id >= 0 { Some(validity_id as usize) } else { None }
		}
	}

	fn transition_validator(&self, from: &PTONode<N>, to: &PTONode<N>) -> Option<usize> {
		unsafe {
			let validity_id = (*self.planning_problem).transition_validity_callback.unwrap()(from.state.as_ptr(), from.state.len(), to.state.as_ptr(), to.state.len());
			if validity_id >= 0 { Some(validity_id as usize) } else { None }
		}
	}
	
	fn reachable_belief_states(&self, _: &BeliefState) -> Vec<BeliefState> {
		self.reachable_belief_states.clone() // is reference ok?
	}
	
	fn world_validities(&self) -> Vec<WorldMask> {
		self.world_validities.clone()
	}

	fn observe(&self, state: &[f64; N], belief_state: &BeliefState) -> Vec<BeliefState> {
		let mut successors = Vec::<BeliefState>::new();

		unsafe {
			let (belief_state_ptr, belief_state_len, _) = belief_state.clone().into_raw_parts(); // why clone?

			let mut successor_belief_ids_p: *mut *mut usize = null_mut(); // pointer to the array
			let mut n_successors: usize = 0;

			(*self.planning_problem).observer_callback.unwrap()(state.as_ptr(), state.len(), belief_state_ptr, belief_state_len, &mut successor_belief_ids_p as *mut *mut *mut usize, &mut n_successors as *mut usize);
			let successor_belief_ids_vec: Vec<usize> = Vec::from_raw_parts(*successor_belief_ids_p, n_successors, n_successors);
			for id in successor_belief_ids_vec {
				successors.push(self.reachable_belief_states[id as usize].clone());
			}
		}

		successors
	}
}

//////////////////////////////////////////////////
////////   GoalAdapter
//////////////////////////////////////////////////

struct GoalAdapter<const N: usize> {
	planning_problem: *const CPlanningProblem,
}

impl<const N: usize> GoalAdapter<N> {
	pub fn new(planning_problem: *const CPlanningProblem) -> Self{
		Self{
			planning_problem
		}
	}
}

impl<const N: usize> GoalFuncs<N> for GoalAdapter<N> {
	fn goal(&self, state: &[f64; N]) -> Option<WorldMask> {
		unsafe {
			let mut validity = vec![false; (*self.planning_problem).n_worlds];
			let mut validity_bit_vec = bitvec![0; (*self.planning_problem).n_worlds];
			let is_goal = (*self.planning_problem).goal_callback.unwrap()(state.as_ptr(), state.len(), validity.as_mut_ptr(), validity.len());

			if !is_goal
			{
				return None;
			}

			for i in 0..(*self.planning_problem).n_worlds { // keep bitvec??
				validity_bit_vec.set(i, validity[i]);
			}

			Some(validity_bit_vec)
		}
	}

	fn goal_example(&self, world: usize) -> [f64;N] {
		unsafe {
			let mut state = [0.0; N];
			(*self.planning_problem).goal_example_callback.unwrap()(world, state.as_mut_ptr(), state.len()); // unwrap at construction time?
			state
		}
	}
}
