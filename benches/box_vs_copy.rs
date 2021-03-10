use criterion::{criterion_group, criterion_main, Criterion};

struct DummyNode {
	children: Vec<usize>
}

impl DummyNode {
	fn children_box(&self) -> Box<dyn Iterator<Item=usize>+ '_> {
		Box::new(self.children.iter().map(|&id| id))
	}

	fn children_copy(&self) -> Vec<usize> {
		self.children.clone()
	}

	fn children_ref(&self) -> &Vec<usize> {
		&self.children
	}
}

fn criterion_benchmark(c: &mut Criterion) {
	let node_2 = DummyNode {
		children: vec![0, 1]
	};
	
	let node_5 = DummyNode {
		children: vec![0, 1, 2, 3, 4]
	};

	let node_15 = DummyNode {
		children: vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
	};

	let node_100 = DummyNode {
		children: vec![0; 100]
	};

	fn loop_box(elements: Box<dyn Iterator<Item=usize>+ '_>) -> usize {
		let mut s = 0;
		for e in elements {
			s += e;
		}
		s
	}

	fn loop_vec(elements: &Vec<usize>) -> usize {
		let mut s = 0;
		for e in elements.iter() {
			s += e;
		}
		s
	}

	c.bench_function("2 children box",  |b| b.iter(||  loop_box(node_2.children_box())  ));
	c.bench_function("2 children copy",  |b| b.iter(|| loop_vec(&node_2.children_copy())  ));
	c.bench_function("2 children refs",  |b| b.iter(|| loop_vec(&node_2.children_ref())  ));

	c.bench_function("5 children box",  |b| b.iter(||  loop_box(node_5.children_box())  ));
	c.bench_function("5 children copy",  |b| b.iter(|| loop_vec(&node_5.children_copy()) ));
	c.bench_function("5 children refs",  |b| b.iter(|| loop_vec(node_5.children_ref()) ));

	c.bench_function("15 children box",  |b| b.iter(|| loop_box(node_15.children_box()) ));
	c.bench_function("15 children copy",  |b| b.iter(|| loop_vec(&node_15.children_copy()) ));
	c.bench_function("15 children refs",  |b| b.iter(|| loop_vec(node_15.children_ref()) ));

	c.bench_function("100 children box",  |b| b.iter(|| loop_box(node_100.children_box()) ));
	c.bench_function("100 children copy",  |b| b.iter(|| loop_vec(&node_100.children_copy()) ));
	c.bench_function("100 children refs",  |b| b.iter(|| loop_vec(node_100.children_ref()) ));
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
