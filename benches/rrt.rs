use criterion::{criterion_group, criterion_main, Criterion};
use po_rrt::rrt::*;
use po_rrt::sample_space::*;
use po_rrt::map_io::*;

/*
fn rrt(max_iter: u32) {
	struct Funcs {}
	impl RRTFuncs<2> for Funcs {}

	fn goal(state: &[f64; 2]) -> bool {
		(state[0] - 0.9).abs() < 0.05 && (state[1] - 0.9).abs() < 0.05
	}	

	let mut samplers = RRTDefaultSamplers {
		state_sampler: ContinuousSampler::new([-1.0, -1.0], [1.0, 1.0]),
		belief_state_sampler: DiscreteSampler::new(),
	};

    let mut rrt = RRT::new(&mut samplers, &Funcs{});
    let _ = rrt.plan([0.0, 0.0], &vec![0.25; 4], goal, 0.1, 10.0, max_iter);
}

fn rrt_map(map: &Map, max_iter: u32) {
	fn goal(state: &[f64; 2]) -> bool {
		(state[0] - 0.0).abs() < 0.05 && (state[1] - 0.9).abs() < 0.05
	}

	let mut samplers = RRTDefaultSamplers {
		state_sampler: ContinuousSampler::new([-1.0, -1.0], [1.0, 1.0]),
		belief_state_sampler: DiscreteSampler::new(),
	};

    let mut rrt = RRT::new(&mut samplers, map);
    let _ = rrt.plan([0.0, -0.8], &vec![0.25; 4], goal, 0.1, 10.0, max_iter);
}*/


fn criterion_benchmark(c: &mut Criterion) {
	let m = Map::open("data/rrt_map_0.pgm", [-1.0, -1.0], [1.0, 1.0]);
    /*c.bench_function("rrt 5000",      |b| b.iter(|| rrt(5000)));
    c.bench_function("rrt 15000",     |b| b.iter(|| rrt(15000)));
    c.bench_function("rrt map 5000",  |b| b.iter(|| rrt_map(&m, 5000)));
    c.bench_function("rrt map 15000", |b| b.iter(|| rrt_map(&m, 15000)));*/
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
