#![allow(dead_code, unused_imports)]

use crate::common::*;
use crate ::prm_graph::*;
use crate ::prm::*;
use crate ::sample_space::*;
use bitvec::prelude::*;
use ptr::{null, null_mut};
use std::convert::TryInto;
use std::iter::FromIterator;
use std::ptr;


pub type StateValidityCallbackType = extern fn(*const f64, usize) -> i64;
pub type TransitionValidityCallbackType = extern fn(from_raw: *const f64, usize, to_raw: *const f64, usize) -> i64;
pub type CostEvaluatorCallbackType = extern fn(from_raw: *const f64, usize, to_raw: *const f64, usize) -> f64;
pub type ObserverCallbackType = extern fn(from_raw: *const f64, usize, belief_state_raw: *const f64, usize,
										  belief_ids_raw: *mut *mut *mut usize, *mut usize);

#[derive(Clone, Copy)]
#[repr(C)]
pub struct CPlanningProblem{
	low: (*mut f64, usize),
	up: (*mut f64, usize),
	n_worlds: usize,
	world_validities: (*mut *mut usize, usize),
	state_validity_callback: Option<StateValidityCallbackType>,
	transition_validity_callback: Option<TransitionValidityCallbackType>,
	cost_evaluator_callback: Option<CostEvaluatorCallbackType>,
	observer_callback: Option<ObserverCallbackType>,
	start_belief_state: (*mut f64, usize),
	reachable_belief_states: (*mut *mut f64, usize)
}

#[no_mangle]
pub extern "C" fn new_planning_problem() -> *mut CPlanningProblem {
    Box::into_raw(Box::new(
		CPlanningProblem{
			low: (ptr::null_mut(), 0),
			up: (ptr::null_mut(), 0),
			n_worlds: 0,
			world_validities: (ptr::null_mut(), 0),
			state_validity_callback: None,
			transition_validity_callback : None,
			cost_evaluator_callback: None,
			observer_callback: None,
			start_belief_state: (ptr::null_mut(), 0),
			reachable_belief_states: (ptr::null_mut(), 0)
		}
	))
}

#[no_mangle]
pub extern "C" fn set_lower_sampling_bound(planning_problem: *mut CPlanningProblem, low_raw: *mut f64, low_size: usize) {
	unsafe {
		(*planning_problem).low = (low_raw, low_size);
	}
}

#[no_mangle]
pub extern "C" fn set_upper_sampling_bound(planning_problem: *mut CPlanningProblem, up_raw: *mut f64, up_size: usize) {
	unsafe {
		(*planning_problem).up = (up_raw, up_size);
	}
}

#[no_mangle]
pub extern "C" fn set_n_worlds(planning_problem: *mut CPlanningProblem, n_worlds: usize) {
	unsafe {
		(*planning_problem).n_worlds = n_worlds;
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
		(*planning_problem).start_belief_state = (start_belief_state_raw, start_belief_state_size);
		(*planning_problem).reachable_belief_states = (reachable_belief_states_raw, reachable_belief_states_size);
	}
}

#[no_mangle]
pub extern "C" fn plan(planning_problem: *const CPlanningProblem, start_raw: *mut f64, start_size : usize,  goal_c: extern fn(*const f64, usize, *mut bool, usize)) {
    unsafe {
		let fns = PRMFuncsAdapter::<2>::new(planning_problem);
		let start : [f64; 2] = Vec::from_raw_parts(start_raw, start_size, start_size).try_into().unwrap();

		let goal = |state: &[f64; 2]| -> WorldMask {
			let mut validity = vec![false; (*planning_problem).n_worlds];

			goal_c(state.as_ptr(), state.len(), validity.as_mut_ptr(), validity.len());

			let mut validity_bit_vec = bitvec![0; (*planning_problem).n_worlds];

			for i in 0..(*planning_problem).n_worlds { // keep bitvec??
				validity_bit_vec.set(i, validity[i]);
			}

			validity_bit_vec
		};

		let mut prm = PRM::new(ContinuousSampler::new(fns.low, fns.up),
							DiscreteSampler::new(),
							&fns);

		prm.grow_graph(&start, goal, 0.1, 5.0, 1000, 5000).expect("graph not grown up to solution");
		prm.print_summary();
		let _policy = prm.plan_belief_space(&fns.start_belief_state);

		println!("number of nodes in policy:{}",_policy.nodes.len());
	}

}

pub struct PRMFuncsAdapter<const N: usize> {
	planning_problem: *const CPlanningProblem,
	low: [f64; N],
	up: [f64; N],
	world_validities: Vec<WorldMask>,
	reachable_belief_states: Vec<BeliefState>,
	start_belief_state: Vec<f64>
}

impl<const N: usize> PRMFuncsAdapter<N> {
	pub fn new(planning_problem: *const CPlanningProblem) -> Self {
		unsafe {
			assert!(N==(*planning_problem).low.1, "N:{}, low size:{}", N, (*planning_problem).low.1);
			assert!(N==(*planning_problem).up.1, "N:{}, up size:{}", N, (*planning_problem).up.1);
		}

		let mut low = [0.0; N];

		unsafe {
			let low_vec = Vec::from_raw_parts((*planning_problem).low.0, (*planning_problem).low.1, (*planning_problem).low.1);
			low = low_vec.try_into().unwrap();
		}

		let mut up = [0.0; N];

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
					validity_bit_vec.set(i, if validity_vec[i] > 0 { true } else { false });
				}

				world_validities.push(validity_bit_vec);
			}
		}

		// start belief state
		let mut start_belief_state: Vec<f64> = Vec::new();

		unsafe {
			start_belief_state = Vec::from_raw_parts((*planning_problem).start_belief_state.0, (*planning_problem).start_belief_state.1, (*planning_problem).start_belief_state.1);
		}

		PRMFuncsAdapter{
			planning_problem,
			low,
			up,
			world_validities,
			reachable_belief_states,
			start_belief_state
		}
	}

}

impl<const N: usize> PRMFuncs<N> for PRMFuncsAdapter<N> {
	fn n_worlds(&self) -> usize {
		unsafe {
			(*self.planning_problem).n_worlds
		}
	}

	fn state_validity(&self, state: &[f64; N]) -> Option<usize> {
		unsafe {
			let validity_id = (*self.planning_problem).state_validity_callback.unwrap()(state.as_ptr(), state.len());
			if validity_id >= 0 { Some(validity_id as usize) } else { None }
		}
	}

	fn transition_validator(&self, from: &PRMNode<N>, to: &PRMNode<N>) -> Option<usize> {
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

			//println!("successor_belief_ids_p:{:?}", successor_belief_ids_p);
			//println!("successor_belief_ids:{:?}", *successor_belief_ids_p);
			//println!("successor_belief_ids, *({:?})={:?}", *successor_belief_ids_p, **successor_belief_ids_p);

			let successor_belief_ids_vec: Vec<usize> = Vec::from_raw_parts(*successor_belief_ids_p, n_successors, n_successors);

			//println!("successor_belief_ids_vec:{:?}", successor_belief_ids_vec);

			for id in successor_belief_ids_vec {
				successors.push(self.reachable_belief_states[id as usize].clone());
			}
		}

		successors
	}
}
