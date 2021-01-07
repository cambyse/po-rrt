use rand::Rng;
use itertools::izip;

pub struct SampleSpace<const N: usize> {
	pub low: [f64; N],
	pub up: [f64; N],
}

impl<const N: usize> SampleSpace<N> {
	pub fn sample(&self) -> [f64; N] {
		let mut rng = rand::thread_rng();

		let mut s = [0.0; N];
		for (v, l, u) in izip!(s.iter_mut(), self.low.iter(), self.up.iter()) {
			*v = rng.gen_range(*l, *u);
		}

		return s;	
	}
}

#[cfg(test)]
mod tests {

use super::*;

#[test]
fn create_sample_space() {
		let space = SampleSpace{low: [-1.0, -1.0], up: [1.0, 1.0]};
		
		assert_eq!([-1.0, -1.0], space.low);
		assert_eq!([1.0, 1.0], space.up);
	}

#[test]
fn draw_sample() {
		let space = SampleSpace{low: [-1.0, -1.0], up: [1.0, 1.0]};
		
		for _ in 0..100 {
				let s = space.sample();
			
				for (v, l, u) in izip!(s.iter(), space.low.iter(), space.up.iter()) {
 
					assert!(l <= v);
					assert!(v <= u);
			}
		}
	}
}
