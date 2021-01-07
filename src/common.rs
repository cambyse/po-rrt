use itertools::izip;
use std::rc::{Weak, Rc};
use std::cell::RefCell;

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

pub fn backtrack<const N: usize>(from: &[f64;N], to: &mut [f64;N], max_step: f64) -> [f64; N] {
	let step = norm1(from, &to);

	if step > max_step {
		let lambda = max_step / step;
		for i in 0..N {
			to[i] = from[i] + (to[i] - from[i]) * lambda;
		}
	}

	*to
}

pub type WeakRef<const N: usize> = Weak<RefCell<Node<N>>>; // is RefCell really needed?
pub type NodeRef<const N: usize> = Rc<RefCell<Node<N>>>; // is RefCell really needed?

pub struct Node<const N: usize> {
	pub id: usize,    
	pub state: [f64; N],
	pub parent: WeakRef<N>,
	pub children: Vec<NodeRef<N>>,
}


/*impl<const N: usize> Drop for Node<N> {
		fn drop(&mut self) {
				println!("Dropping Node!");
		}
}*/
