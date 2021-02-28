use criterion::{criterion_group, criterion_main, Criterion};
use bitvec::prelude::*;


fn bitvec_or(a: &BitVec, b: &BitVec) {
	let mut or = a.clone();
	for i in 0..a.len() {
		or.set(i, a[i] ||  b[i])
	}
}

fn vecbool_or(a: &Vec<bool>, b: &Vec<bool>) {
	let mut or = a.clone();
	for i in 0..a.len() {
		or[i] |= b[i];
	}
}

fn array_or<const N: usize> (a: &[bool; N], b: &[bool; N]) {
	let mut or = a.clone();
	for i in 0..a.len() {
		or[i] |= b[i];
	}
}

fn criterion_benchmark(c: &mut Criterion) {
	c.bench_function("OR with BitVec, N=6",  |b| b.iter(||(bitvec_or(&bitvec![1,1,0,1,0,1], &bitvec![0,1,1,0,1,0]))));
	c.bench_function("OR with Vec<bool>, N=6",  |b| b.iter(||(vecbool_or(&vec![true, true, false, true, false, true], &vec![false, true, true, false, true, false]))));
	c.bench_function("OR with [bool; N]], N=6",  |b| b.iter(||(array_or(&[true, true, false, true, false, true], &[false, true, true, false, true, false]))));
	c.bench_function("OR with BitVec, N=12",  |b| b.iter(||(bitvec_or(&bitvec![1,1,0,1,0,1,1,1,0,1,0,1], &bitvec![0,1,1,0,1,0, 0,1,1,0,1,0]))));
	c.bench_function("OR with Vec<bool>, N=12",  |b| b.iter(||(vecbool_or(&vec![true, true, false, true, false, true, true, true, false, true, false, true], &vec![false, true, true, false, true, false, false, true, true, false, true, false]))));
	c.bench_function("OR with [bool; N]], N=12",  |b| b.iter(||(array_or(&[true, true, false, true, false, true, true, true, false, true], &[false, true, true, false, true, false, false, true, true, false]))));
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
