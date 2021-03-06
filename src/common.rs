use itertools::izip;
use std::{iter::Zip, slice::Iter};
use bitvec::prelude::*;

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

pub fn pairwise_iter<T>(v: &Vec<T>) -> Zip<Iter<T>, Iter<T>> {
	v[0..v.len()-1].iter().zip(&v[1..])
}

pub type WorldMask = BitVec;
pub type BeliefState = Vec<f64>;

pub trait GraphNode<const N: usize> {
	fn state(&self) -> &[f64; N];
}

pub trait Graph<const N: usize> {
	fn node(&self, id:usize) -> &dyn GraphNode<N>;
	fn n_nodes(&self) -> usize;
	fn children(&self, id: usize) -> Vec<usize>;
	fn parents(&self, id: usize) -> Vec<usize>;
}

pub trait ObservationGraph {
	fn siblings(&self, parent_id: usize, id: usize) -> Vec<(usize, f64)>; // in case of observation branching, returns the siblings obtained from other observations along with their probability
}

use std::cmp::Ordering;

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

pub fn is_compatible(belief_state: &BeliefState, validity: &WorldMask) -> bool {
	for (&p, v) in belief_state.iter().zip(validity) {
		if p > 0.0 && ! v {
			return false;
		}
	}

	true
}


pub struct PolicyNode<const N: usize> {
	pub state: [f64; N],
	pub parent: Option<usize>,
	pub children: Vec<usize>,
}

pub struct Policy<const N: usize> {
	pub nodes: Vec<PolicyNode<N>>
}