use criterion::{criterion_group, criterion_main, Criterion};
use po_rrt::common::*;
use po_rrt::pto::*;
use po_rrt::sample_space::*;
use po_rrt::map_io::*;
use bitvec::prelude::*;


fn prm_map(m: &Map, min_iter: usize) {
	let goal = SquareGoal::new(vec![([0.55, 0.9], bitvec![1; 4])], 0.05);
	let mut prm = PTO::new(ContinuousSampler::new([-1.0, -1.0], [1.0, 1.0]),
						   DiscreteSampler::new(),
						   m);

	prm.grow_graph(&[0.55, -0.8], &goal, 0.05, 5.0, min_iter, 100000).unwrap();
	prm.plan_belief_space(&vec![0.25; 4]);
}


fn criterion_benchmark(c: &mut Criterion) {
	let mut m = Map::open("data/map2.pgm", [-1.0, -1.0], [1.0, 1.0]);
	m.add_zones("data/map2_zone_ids.pgm", 0.1);
    c.bench_function("prm map 5000",  |b| b.iter(|| prm_map(&m, 5000)));
    c.bench_function("prm map 15000", |b| b.iter(|| prm_map(&m, 15000)));
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
