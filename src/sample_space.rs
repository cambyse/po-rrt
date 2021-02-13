use rand::{Rng, SeedableRng};
use rand_pcg::Pcg64;
use itertools::izip;

pub struct SampleSpace<const N: usize> {
	pub low: [f64; N],
	pub up: [f64; N],
	rng: Pcg64,
}

impl<const N: usize> SampleSpace<N> {
	pub fn new(low: [f64; N], up: [f64; N]) -> Self {
		Self {
			low,
			up,
			rng: Pcg64::seed_from_u64(2)
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

		return s;	
	}
}

#[cfg(test)]
mod tests {

use super::*;

#[test]
fn create_sample_space() {
		let space = SampleSpace::new([-1.0, -1.0], [1.0, 1.0]);
		
		assert_eq!([-1.0, -1.0], space.low);
		assert_eq!([1.0, 1.0], space.up);
	}

#[test]
fn draw_sample() {
		let mut space = SampleSpace::new([-1.0, -1.0], [1.0, 1.0]);
		
		for _ in 0..100 {
				let s = space.sample();
			
				for (v, l, u) in izip!(s.iter(), space.low.iter(), space.up.iter()) {
 
					assert!(l <= v);
					assert!(v <= u);
			}
		}
	}
}
