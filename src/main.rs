#![allow(dead_code, unused_imports)]
#![feature(slice_group_by)]

use po_rrt::{
    prm::*,
    prm_graph::*,
    sample_space::*,
    map_io::*,
    common::*,
};
use bitvec::prelude::*;

fn main() {
    let mut m = Map::open("data/map4.pgm", [-1.0, -1.0], [1.0, 1.0]);
	m.add_zones("data/map4_zone_ids.pgm", 0.3);

	fn goal(state: &[f64; 2]) -> WorldMask {
		bitvec![if (state[0] + 0.55).abs() < 0.05 && (state[1] - 0.9).abs() < 0.05 { 1 } else { 0 }; 16]
	}

	let mut prm = PRM::new(ContinuousSampler::new([-1.0, -1.0], [1.0, 1.0]),
						   DiscreteSampler::new(),
						   &m);

	prm.grow_graph(&[0.55, -0.8], goal, 0.1, 5.0, 2000, 100000).expect("graph not grown up to solution");
	prm.print_summary();
	let policy = prm.plan_belief_space(&vec![1.0/16.0; 16] ); //&vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);

	let mut m2 = m.clone();
	m2.resize(5);
    m2.draw_full_graph(&prm.graph);
    m2.draw_zones_observability();
	m2.draw_policy(&policy);
	m2.save("results/test_plan_on_map4_pomdp_main");
}
