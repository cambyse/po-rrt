use rand::{Rng, SeedableRng};
use rand_pcg::Pcg64;
use itertools::izip;

pub struct ContinuousSampler<const N: usize> {
	pub low: [f64; N],
	pub up: [f64; N],
	rng: Pcg64,
}

impl<const N: usize> ContinuousSampler<N> {
	pub fn new(low: [f64; N], up: [f64; N]) -> Self {
		// TODO use environment variable to seed the rng
		Self {
			low,
			up,
			rng: Pcg64::seed_from_u64(0)
		}
	}

	pub fn new_true_random(low: [f64; N], up: [f64; N]) -> Self {
		Self {
			low,
			up,
			rng: Pcg64::from_rng(rand::thread_rng()).unwrap()
		}
	}

	pub fn sample(&mut self) -> [f64; N] {
		let mut s = [0.0; N];
		for (v, l, u) in izip!(s.iter_mut(), self.low.iter(), self.up.iter()) {
			*v = self.rng.gen_range(*l..*u);
		}
		s	
	}
}

pub struct DiscreteSampler{
	rng: Pcg64
}

impl DiscreteSampler {
	#[allow(clippy::new_without_default)]
	pub fn new() -> Self {
		Self {
			rng: Pcg64::seed_from_u64(0)
		}
	}

	pub fn new_true_random() -> Self {
		Self {
			rng: Pcg64::from_rng(rand::thread_rng()).unwrap()
		}
	}

	pub fn sample(&mut self, n: usize) -> usize {
		self.rng.gen_range(0..n)
	}
}

#[cfg(test)]
mod tests {

use super::*;

#[test]
fn create_sample_space() {
		let space = ContinuousSampler::new([-1.0, -1.0], [1.0, 1.0]);
		
		assert_eq!([-1.0, -1.0], space.low);
		assert_eq!([1.0, 1.0], space.up);
	}

#[test]
fn draw_sample() {
		let mut space = ContinuousSampler::new([-1.0, -1.0], [1.0, 1.0]);
		
		for _ in 0..100 {
				let s = space.sample();
			
				for (v, l, u) in izip!(s.iter(), space.low.iter(), space.up.iter()) {
 
					assert!(l <= v);
					assert!(v <= u);
			}
		}
	}

#[test]
fn draw_discrete_sample() {
	let mut space = DiscreteSampler::new();
	
	for _ in 0..100 {
			let s = space.sample(10);
		
			assert!(s < 10);
	}
}
}
