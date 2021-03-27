use itertools::izip;
use std::{iter::Zip, slice::Iter, iter::Iterator};
use bitvec::prelude::*;
use std::cmp::Ordering;

pub type WorldMask = BitVec;
pub type BeliefState = Vec<f64>;
pub type NodeId = usize;

pub trait GraphNode<const N: usize> {
	fn state(&self) -> &[f64; N];
}

pub trait Graph<const N: usize> {
	fn node(&self, id:usize) -> &dyn GraphNode<N>;
	fn n_nodes(&self) -> usize;
	fn children(&self, id: usize) -> Box<dyn Iterator<Item=usize>+ '_>;
	fn parents(&self, id: usize) -> Box<dyn Iterator<Item=usize>+ '_>;
}

pub struct PolicyNode<const N: usize> {
	pub state: [f64; N],
	pub belief_state: Vec<f64>,
	pub parent: Option<usize>,
	pub children: Vec<usize>,
}

pub struct Policy<const N: usize> {
	pub nodes: Vec<PolicyNode<N>>,
	pub leafs: Vec<usize>
}

impl<const N: usize> Policy<N> {
	#[allow(clippy::style)]
	pub fn add_node(&mut self, state: &[f64; N], belief_state: &BeliefState, is_leaf: bool) -> usize {
		let id = self.nodes.len();

		self.nodes.push(PolicyNode{
			state: *state,
			belief_state: belief_state.clone(),
			parent: None,
			children: Vec::new()
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
		path.push(node.state.clone());

		while node.parent.is_some() {
			node = &self.nodes[node.parent.unwrap()];
			path.push(node.state.clone());
		}

		path.reverse();
		path
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
	for (&p, v) in belief_state.iter().zip(validity) {
		if p > 0.0 && ! v {
			return false;
		}
	}

	true
}

#[allow(clippy::style)]
pub fn assert_belief_state_validity(belief_state: &BeliefState) {
	assert!((belief_state.iter().fold(0.0, |s, p| p + s) - 1.0).abs() < 0.000001);
}