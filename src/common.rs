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