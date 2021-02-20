use criterion::{criterion_group, criterion_main, Criterion};
use bitvec::prelude::*;


fn bitvec_or(a: &BitVec, b: &BitVec) {
	//let mut or = a.clone();
	//or |= b.clone();

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

fn array_or(a: &[bool; 6], b: &[bool; 6]) {
	let mut or = a.clone();
	for i in 0..a.len() {
		or[i] |= b[i];
	}
}

fn criterion_benchmark(c: &mut Criterion) {
	c.bench_function("OR with BitVec",  |b| b.iter(||(bitvec_or(&bitvec![1,1,0,1,0,1], &bitvec![0,1,1,0,1,0]))));
	c.bench_function("OR with Vec<bool>",  |b| b.iter(||(vecbool_or(&vec![true, true, false, true, false, true], &vec![false, true, true, false, true, false]))));
	c.bench_function("OR with [bool; N]]",  |b| b.iter(||(array_or(&[true, true, false, true, false, true], &[false, true, true, false, true, false]))));
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
