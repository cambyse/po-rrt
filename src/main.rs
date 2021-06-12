#![allow(dead_code, unused_imports)]
#![feature(slice_group_by)]

use po_rrt::{
    prm::*,
    prm_graph::*,
    sample_space::*,
    map_io::*,
	common::*,
	prm_policy_refiner::*
};
use bitvec::prelude::*;

fn main() {
    let mut m = Map::open("data/map4.pgm", [-1.0, -1.0], [1.0, 1.0]);
	m.add_zones("data/map4_zone_ids.pgm", 0.3);

	let goal = SquareGoal::new(vec![([0.55, 0.9], bitvec![1; 16])], 0.05);

	let mut prm = PRM::new(ContinuousSampler::new([-1.0, -1.0], [1.0, 1.0]),
						   DiscreteSampler::new(),
						   &m);

	prm.grow_graph(&[0.55, -0.8], goal, 0.1, 5.0, 2000, 100000).expect("graph not grown up to solution");
	prm.print_summary();
	let policy = prm.plan_belief_space(&vec![1.0/16.0; 16] ); //&vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);

	let mut policy_refiner = PRMPolicyRefiner::new(&policy, &m, &prm.belief_graph);
	let (_policy, _trees) = policy_refiner.refine_solution(0.3);

	/*
	let mut m2 = m.clone();
	m2.resize(5);
    m2.draw_full_graph(&prm.graph);
	m2.draw_zones_observability();
	m2.draw_refinment_trees(&trees);
	m2.draw_policy(&policy);
	m2.save("results/test_plan_on_map4_pomdp_main");*/
}
