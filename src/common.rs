use itertools::izip;
use std::{iter::Zip, slice::Iter, iter::Iterator};
use bitvec::prelude::*;
use std::cmp::Ordering;
use queues::*;

use crate::belief_graph;

pub type WorldMask = BitVec;
pub type BeliefState = Vec<f64>;
pub type NodeId = usize;

pub trait GraphNode<const N: usize> {
	fn state(&self) -> &[f64; N];
}

pub trait Graph<const N: usize> {
	fn node(&self, id:usize) -> &dyn GraphNode<N>;
	fn n_nodes(&self) -> usize;
	fn children(&self, id: usize) -> Vec<usize>; // deprecated trait -> do not use
	fn parents(&self, id: usize) ->Vec<usize>; // deprecated trait -> do not use
}

#[derive(Clone)]
pub struct PolicyNode<const N: usize> {
	pub state: [f64; N],
	pub belief_state: Vec<f64>,
	pub parent: Option<usize>,
	pub children: Vec<usize>,
	// used for refining policy
	pub original_node_id: usize
}

#[derive(Clone)]
pub struct Policy<const N: usize> {
	pub nodes: Vec<PolicyNode<N>>,
	pub leafs: Vec<usize>
}

impl<const N: usize> Policy<N> {
	#[allow(clippy::style)]
	pub fn add_node(&mut self, state: &[f64; N], belief_state: &BeliefState, original_id: usize, is_leaf: bool) -> usize {
		let id = self.nodes.len();

		self.nodes.push(PolicyNode{
			state: *state,
			belief_state: belief_state.clone(),
			parent: None,
			children: Vec::new(),
			original_node_id: original_id
		});

		if is_leaf {
			self.leafs.push(id);
		}

		id
	}

	pub fn add_edge(&mut self, parent_id: usize, child_id: usize) {
		self.nodes[parent_id].children.push(child_id);
		self.nodes[child_id].parent = Some(parent_id);
	}

	pub fn leaf(&self, id: usize) -> &PolicyNode<N> {
		&self.nodes[self.leafs[id]]
	}

	pub fn path_to_leaf(&self, id: usize) -> Vec<[f64; N]> {
		let mut path = Vec::<[f64; N]>::new();
		
		let mut node = self.leaf(id);
		path.push(node.state);

		while node.parent.is_some() {
			node = &self.nodes[node.parent.unwrap()];
			path.push(node.state);
		}

		path.reverse();
		path
	}

	pub fn decompose(&self) -> (Vec<(BeliefState, Vec<usize>)>, Vec<Vec<usize>>) {
		let mut pieces = Vec::<(BeliefState, Vec<usize>)>::new();
		let mut skeleton = Vec::<Vec<usize>>::new();

		let mut n_pieces = 0;
		let mut fifo: Queue<usize> = queue![];
		fifo.add(0).unwrap();

		while fifo.size() > 0 {
			let id = fifo.remove().unwrap();

			let mut ids = vec![];
			let mut successors = vec![];

			let mut current_id = id;
			loop {
				assert_eq!(self.nodes[id].belief_state, self.nodes[current_id].belief_state);

				ids.push(current_id);

				match self.nodes[current_id].children.len() {
					0 => { break; } // final node
					1 => { current_id = self.nodes[current_id].children[0] }, // simple forward
					_ => { // branching
						for &child_id in &self.nodes[current_id].children {
							fifo.add(child_id).unwrap();
							n_pieces += 1;
							successors.push(n_pieces);
						}
						break;
					}
				}
			}

			let piece: (BeliefState, Vec<usize>) = (
				self.nodes[id].belief_state.clone(),
				ids
			);

			pieces.push(piece);
			skeleton.push(successors);
		}

		(pieces, skeleton)
	}
}

pub fn norm1<const N: usize>(a: &[f64; N], b: &[f64; N]) -> f64 {
	let mut d = 0.0;
	
	for (xa, xb) in izip!(a.iter(), b.iter())
	{
		d += (xb - xa).abs();
	}
	
	d
}

pub fn norm2<const N: usize>(a: &[f64; N], b: &[f64; N]) -> f64 {
	let mut d2 = 0.0;
	
	for (xa, xb) in izip!(a.iter(), b.iter())
	{
		let dx = xb - xa;
		d2 += dx * dx;
	}

	d2.sqrt()
}

pub fn steer<const N: usize>(from: &[f64;N], to: &mut [f64;N], max_step: f64) {
	let step = norm1(from, &to);

	if step > max_step {
		let lambda = max_step / step;
		for i in 0..N {
			to[i] = from[i] + (to[i] - from[i]) * lambda;
		}
	}
}

pub fn pairwise_iter<T>(v: &[T]) -> Zip<Iter<T>, Iter<T>> {
	v[0..v.len()-1].iter().zip(&v[1..])
}

pub struct Priority{
	pub prio: f64
}

impl Ord for Priority {
    fn cmp(&self, other: &Self) -> Ordering {
        if self.prio < other.prio { Ordering::Greater } else { Ordering::Less }
    }
}

impl PartialOrd for Priority {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for Priority {
    fn eq(&self, other: &Self) -> bool {
        self.prio == other.prio
    }
}

impl Eq for Priority {}

#[allow(clippy::style)]
pub fn is_compatible(belief_state: &BeliefState, validity: &WorldMask) -> bool {

	/*
	assert_eq!(belief_state.len(), validity.len());

	for i in 0..validity.len() {
		if !validity[i] && belief_state[i] > 0.0 {
			return false;
		}
	}
	*/

	for (&p, v) in belief_state.iter().zip(validity) {
		if p > 0.0 && ! v {
			return false;
		}
	}

	true
}

pub fn compute_compatibility(belief_states: &[BeliefState], world_validities: &[WorldMask]) -> Vec<Vec<bool>> {
	let mut compatibilities = vec![vec![false; world_validities.len()]; belief_states.len()];

	for (belief_id, _) in belief_states.iter().enumerate() {
		for (validity_id, _) in world_validities.iter().enumerate() {
			compatibilities[belief_id][validity_id] = is_compatible(&belief_states[belief_id], &world_validities[validity_id]);
		}
	}

	compatibilities
}

#[allow(clippy::style)]
pub fn assert_belief_state_validity(belief_state: &BeliefState) {
	assert!((belief_state.iter().fold(0.0, |s, p| p + s) - 1.0).abs() < 0.000001);
}

pub fn contains(wm1: &WorldMask, wm2: &WorldMask) -> bool {
	// wether wm1 contains wm2
	for (w1, w2) in wm1.iter().zip(wm2) {
		if *w2 && !*w1 {
			return false;
		}
	}
	true
}


pub trait GoalFuncs<const N: usize> {
	fn goal(&self, _: &[f64; N]) -> Option<WorldMask> {
		None
	}

	fn goal_example(&self, _:usize) -> [f64;N] {
		[0.0; N]
	}
}

pub struct SquareGoal<const N: usize> {
	goal_to_validity: Vec<([f64;N], WorldMask)>,
	world_to_goal: Vec<[f64; N]>,
	max_dist : f64,
}

impl<const N: usize> SquareGoal<N> {
	pub fn new(goal_to_validity: Vec<([f64;N], WorldMask)>, max_dist: f64) -> Self {
		let n_worlds = goal_to_validity.first().expect("should have at least one element").1.len();
		let mut world_to_goal: Vec<[f64; N]> = vec![[0.0; N]; n_worlds];

		let mut world_has_goal = vec![false; n_worlds];
		for world in 0..n_worlds {
			for (goal, validity) in &goal_to_validity {
				if validity[world] {

					assert!(!world_has_goal[world]); // validities shouldn't overlap
					
					world_to_goal[world] = goal.clone();
					world_has_goal[world] = true;
				}
			}
		}

		Self{
			goal_to_validity: goal_to_validity.clone(),
			world_to_goal,
			max_dist
		}
	}
}

impl<const N: usize> GoalFuncs<N> for SquareGoal<N> {
	fn goal(&self, state: &[f64; N]) -> Option<WorldMask> {
		for (goal, validity)  in &self.goal_to_validity {
			if norm1(state, goal) < self.max_dist {
				return Some(validity.clone());
			}
		}

		None
	}

	fn goal_example(&self, world:usize) -> [f64; N] {
		self.world_to_goal[world]
	}
}

#[cfg(test)]
mod tests {

use bitvec::vec;

use super::*;

#[test]
fn test_goal() {
	let goal = SquareGoal::new(vec![ ([0.1, 0.1], bitvec![1, 0]), ([0.9, 0.9], bitvec![0, 1]) ], 0.1);

	assert_eq!(goal.goal(&[0.11, 0.11]).unwrap(), bitvec![1, 0]);
	assert_eq!(goal.goal(&[0.5, 0.5]), None);
	assert_eq!(goal.goal(&[0.91, 0.91]).unwrap(), bitvec![0, 1]);

	assert_eq!(goal.goal_example(0), [0.1, 0.1]);
	assert_eq!(goal.goal_example(1), [0.9, 0.9]);
}

#[test]
fn test_wm_contains() {
	assert!(contains(&bitvec![1,1], &bitvec![1,1]));
	assert!(contains(&bitvec![1,1], &bitvec![1,0]));
	assert!(contains(&bitvec![1,1], &bitvec![0,1]));
	assert!(contains(&bitvec![1,1], &bitvec![0,0]));

	assert!(contains(&bitvec![1,0], &bitvec![1,0]));
	assert!(!contains(&bitvec![1,0], &bitvec![0,1]));
	assert!(!contains(&bitvec![0,0], &bitvec![0,1]));
}

#[test]
fn test_policy_decomposition() {
	/*
	  4   5
	   \ /
		1, 2, 3
		|
		0
	*/
	
	let mut policy = Policy{nodes: vec![], leafs: vec![]};

	policy.add_node(&[0.0, 0.0], &vec![0.5, 0.5], 0, false);

	policy.add_node(&[0.0, 1.0], &vec![0.5, 0.5], 0, false); // 1
	policy.add_node(&[0.0, 1.0], &vec![1.0, 0.0], 0, false); // 2
	policy.add_node(&[0.0, 1.0], &vec![0.0, 1.0], 0, false); // 3

	policy.add_node(&[-1.0, 2.0], &vec![1.0, 0.0], 0, false); // 4
	policy.add_node(&[11.0, 2.0], &vec![0.0, 1.0], 0, false); // 5

	policy.add_edge(0, 1);
	policy.add_edge(1, 2);
	policy.add_edge(1, 3);

	policy.add_edge(2, 4);
	policy.add_edge(3, 5);



	let (policy_pieces, _) = policy.decompose();

	assert_eq!(policy_pieces.len(), 3);
}

}