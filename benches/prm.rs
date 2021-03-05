use criterion::{criterion_group, criterion_main, Criterion};
use po_rrt::common::*;
use po_rrt::prm::*;
use po_rrt::sample_space::*;
use po_rrt::map_io::*;
use bitvec::prelude::*;


fn prm_map(m: &Map, min_iter: usize) {
	fn goal(state: &[f64; 2]) -> WorldMask {
		bitvec![if (state[0] - 0.55).abs() < 0.05 && (state[1] - 0.9).abs() < 0.05 { 1 } else { 0 }; 4]
	}

	let mut prm = PRM::new(ContinuousSampler::new([-1.0, -1.0], [1.0, 1.0]),
						   DiscreteSampler::new(),
						   m);

	prm.grow_graph(&[0.55, -0.8], goal, 0.05, 5.0, min_iter, 100000).unwrap();
	prm.plan_qmdp().unwrap();
	prm.react_qmdp(&[0.0, -0.8], &vec![0.25; 4], 0.1).unwrap();
}


fn criterion_benchmark(c: &mut Criterion) {
	let mut m = Map::open("data/map2.pgm", [-1.0, -1.0], [1.0, 1.0]);
	m.add_zones("data/map2_zone_ids.pgm", 0.1);
    c.bench_function("prm map 5000",  |b| b.iter(|| prm_map(&m, 5000)));
    c.bench_function("prm map 15000", |b| b.iter(|| prm_map(&m, 15000)));
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
