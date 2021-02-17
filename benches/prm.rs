use criterion::{criterion_group, criterion_main, Criterion};
use po_rrt::prm::*;
use po_rrt::sample_space::*;
use po_rrt::map_io::*;

fn prm_map(m: &Map, min_iter: usize) {
	fn goal(state: &[f64; 2]) -> bool {
		(state[0] - 0.0).abs() < 0.05 && (state[1] - 0.9).abs() < 0.05
	}

	let mut prm = PRM::new(ContinuousSampler::new([-1.0, -1.0], [1.0, 1.0]),
						   DiscreteSampler::new(),
						   m);

	let result = prm.grow_graph(&[0.55, -0.8], goal, 0.05, 5.0, min_iter, 100000);
	assert_eq!(result, Ok(()));
	let _ = prm.plan(&[0.0, -0.8], &vec![0.25, 0.25, 0.25, 0.25]).unwrap();
}


fn criterion_benchmark(c: &mut Criterion) {
	let mut m = Map::open("data/map2.pgm", [-1.0, -1.0], [1.0, 1.0]);
	m.add_zones("data/map2_zone_ids.pgm");
    c.bench_function("prm map 5000",  |b| b.iter(|| prm_map(&m, 5000)));
    c.bench_function("prm map 15000", |b| b.iter(|| prm_map(&m, 15000)));
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
