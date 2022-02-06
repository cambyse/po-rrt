#![allow(dead_code, unused_imports)]
#![feature(slice_group_by)]

use po_rrt::{
    prm::*,
    prm_graph::*,
    sample_space::*,
    map_io::*,
	map_shelves_io::*,
	common::*,
	prm_policy_refiner::*,
	map_shelves_tamp_rrt::*
};
use bitvec::prelude::*;
use std::time::Instant;
use std::fs::File;
use std::io::Write;

fn main()
{
	let n_runs = 10;

	let prm_iter_min = 5000;
	let rrt_iter_min = 2500;

	let mut w = File::create("results/map_benchmark/costs_and_timings.txt").unwrap();

	for m in vec![2, 4, 6, 8] {
		let (prm_planning_times, prm_costs): (Vec<f64>, Vec<f64>) = 
		match m {
			2 => (0..n_runs).map(|_| test_plan_on_map1_2_goals(prm_iter_min)).unzip(),
			4 => (0..n_runs).map(|_| test_plan_on_map_benchmark_4_goals(prm_iter_min)).unzip(),
			6 => (0..n_runs).map(|_| test_plan_on_map_benchmark_6_goals(prm_iter_min)).unzip(),
			8 => (0..n_runs).map(|_| test_plan_on_map_benchmark_8_goals(prm_iter_min)).unzip(),
			_ => panic!("no bencmark function for it!")
		};

		let (rrt_planning_times, rrt_costs): (Vec<f64>, Vec<f64>) =
		match m {
			2 => (0..n_runs).map(|_| test_plan_tamp_rrt_on_map_benchmark_2_goals(rrt_iter_min)).unzip(),
			4 => (0..n_runs).map(|_| test_plan_tamp_rrt_on_map_benchmark_4_goals(rrt_iter_min)).unzip(),
			6 => (0..n_runs).map(|_| test_plan_tamp_rrt_on_map_benchmark_6_goals(rrt_iter_min)).unzip(),
			8 => (0..n_runs).map(|_| test_plan_tamp_rrt_on_map_benchmark_8_goals(rrt_iter_min)).unzip(),
			_ => panic!("no bencmark function for it!")
		};

		println!("PRM --- {} goals", m);
		println!("costs: {:?}", compute_statistics(&prm_costs));
		println!("planning_times: {:?}", compute_statistics(&prm_planning_times));

		println!("RRT* --- {} goals", m);
		println!("costs: {:?}", compute_statistics(&rrt_costs));
		println!("planning_times: {:?}", compute_statistics(&rrt_planning_times));

		writeln!(&mut w, "PRM --- {} goals", m).unwrap();
		writeln!(&mut w, "costs: {:?}", compute_statistics(&prm_costs)).unwrap();
		writeln!(&mut w, "planning_times: {:?}", compute_statistics(&prm_planning_times)).unwrap();
		writeln!(&mut w, "\n").unwrap();

		writeln!(&mut w, "RRT* --- {} goals", m).unwrap();
		writeln!(&mut w, "costs: {:?}", compute_statistics(&rrt_costs)).unwrap();
		writeln!(&mut w, "planning_times: {:?}", compute_statistics(&rrt_planning_times)).unwrap();
		writeln!(&mut w, "\n").unwrap();
		writeln!(&mut w, "\n").unwrap();
	}
	//let (prm_planning_times, prm_costs): (Vec<f64>, Vec<f64>) = (0..n_runs).map(|_| test_plan_on_map1_2_goals(prm_iter_min)).unzip();
	//let (rrt_planning_times, rrt_costs): (Vec<f64>, Vec<f64>) = (0..n_runs).map(|_| test_plan_tamp_rrt_on_map1_2_goals(rrt_iter_min)).unzip();

	// 2 goals
	//let (prm_planning_times, prm_costs): (Vec<f64>, Vec<f64>) = (0..n_runs).map(|_| test_plan_on_map_benchmark_2_goals(prm_iter_min)).unzip();
	//let (rrt_planning_times, rrt_costs): (Vec<f64>, Vec<f64>) = (0..n_runs).map(|_| test_plan_tamp_rrt_on_map_benchmark_2_goals(rrt_iter_min)).unzip();

	// 4 goals
	//let (prm_planning_times, prm_costs): (Vec<f64>, Vec<f64>) = (0..n_runs).map(|_| test_plan_on_map_benchmark_4_goals(prm_iter_min)).unzip();
	//let (rrt_planning_times, rrt_costs): (Vec<f64>, Vec<f64>) = (0..n_runs).map(|_| test_plan_tamp_rrt_on_map_benchmark_4_goals(rrt_iter_min)).unzip();

	// 6 goals
	//let (prm_planning_times, prm_costs): (Vec<f64>, Vec<f64>) = (0..n_runs).map(|_| test_plan_on_map_benchmark_6_goals(prm_iter_min)).unzip();
	//let (rrt_planning_times, rrt_costs): (Vec<f64>, Vec<f64>) = (0..n_runs).map(|_| test_plan_tamp_rrt_on_map_benchmark_6_goals(rrt_iter_min)).unzip();

	// 8 goals
	//let (prm_planning_times, prm_costs): (Vec<f64>, Vec<f64>) = (0..n_runs).map(|_| test_plan_on_map_benchmark_8_goals(prm_iter_min)).unzip();
	//let (rrt_planning_times, rrt_costs): (Vec<f64>, Vec<f64>) = (0..n_runs).map(|_| test_plan_tamp_rrt_on_map_benchmark_8_goals(rrt_iter_min)).unzip();

	//println!("PRM");
	//println!("prm_costs all: {:?}", prm_costs);
	//println!("planning_times: {:?}", compute_statistics(&prm_planning_times));
	//println!("costs: {:?}", compute_statistics(&prm_costs));
	
	//println!("RRT*");
	//println!("rrt_costs all: {:?}", rrt_costs);
	//println!("planning_times: {:?}", compute_statistics(&rrt_planning_times));
	//println!("costs: {:?}", compute_statistics(&rrt_costs));

	//writeln!(&mut w, "PRM").unwrap();
    //writeln!(&mut w, "planning_times: {:?}", compute_statistics(&prm_planning_times)).unwrap();
	//writeln!(&mut w, "costs: {:?}", compute_statistics(&prm_costs)).unwrap();
}

fn normalize_belief(unnormalized_belief_state: &BeliefState) -> BeliefState {
	let sum = unnormalized_belief_state.iter().fold(0.0, |sum, p| sum + p);
	unnormalized_belief_state.iter().map(|p| p / sum).collect()
}

fn test_plan_on_map4() {
    let mut m = Map::open("data/map4.pgm", [-1.0, -1.0], [1.0, 1.0]);
	m.add_zones("data/map4_zone_ids.pgm", 0.3);

	let goal = SquareGoal::new(vec![([0.55, 0.9], bitvec![1; 16])], 0.05);

	let mut prm = PRM::new(ContinuousSampler::new([-1.0, -1.0], [1.0, 1.0]),
						   DiscreteSampler::new(),
						   &m);

	prm.grow_graph(&[0.55, -0.8], &goal, 0.05, 5.0, 2000, 100000).expect("graph not grown up to solution");
	prm.print_summary();
	let policy = prm.plan_belief_space(&vec![1.0/16.0; 16] ); //&vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);

	let mut policy_refiner = PRMPolicyRefiner::new(&policy, &m, &prm.belief_graph);
	let (policy, trees) = policy_refiner.refine_solution(RefinmentStrategy::Reparent(0.3));

	let mut m2 = m.clone();
	m2.resize(5);
    m2.draw_full_graph(&prm.graph);
	m2.draw_zones_observability();
	m2.draw_refinment_trees(&trees);
	m2.draw_policy(&policy);
	m2.save("results/test_plan_on_map4_pomdp_main");
}

// 0000 : 0
// 1000 : 1
// 0100 : 2
// 1100 : 3
// 0010 : 4
// 1010 : 5
// 0110 : 6
// 1110 : 7
// 0001 : 8
// 1001 : 9
// 0101 : 10
// 1101 : 11
// 0011 : 12
// 1011 : 13
// 0111 : 14
// 1111 : 15

fn test_plan_on_map5_6_goals() {
	let mut m = MapShelfDomain::open("data/map5.pgm", [-1.0, -1.0], [1.0, 1.0]);
	m.add_zones("data/map5_6_goals_zone_ids.pgm", 0.4);

	let goal = SquareGoal::new(vec![([-0.75, 0.75], bitvec![1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
									([-0.25, 0.75], bitvec![0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
									([ 0.25, 0.75], bitvec![0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
									([ 0.75, 0.75], bitvec![0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]),
									([-0.75, 0.25], bitvec![0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]),
									([-0.25, 0.25], bitvec![0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0])],
									 0.05);

	let mut prm = PRM::new(ContinuousSampler::new([-1.0, -1.0], [1.0, 1.0]),
						DiscreteSampler::new(),
						&m);

	prm.grow_graph(&[0.0, -0.8], &goal, 0.05, 5.0, 5000, 100000).expect("graph not grown up to solution");
	prm.print_summary();

	let policy = prm.plan_belief_space(&vec![1.0/6.0; 6]);
	let mut policy_refiner = PRMPolicyRefiner::new(&policy, &m, &prm.belief_graph);
	let (policy, _) = policy_refiner.refine_solution(RefinmentStrategy::Reparent(0.3));

	let mut m2 = m.clone();
	m2.resize(5);
	m2.draw_full_graph(&prm.graph);
	m2.draw_zones_observability();
	m2.draw_policy(&policy);
	m2.save("results/map5/test_map5_6_goals_pomdp");
}

fn test_plan_on_map5_8_goals() {
	let mut m = MapShelfDomain::open("data/map5.pgm", [-1.0, -1.0], [1.0, 1.0]);
	m.add_zones("data/map5_8_goals_zone_ids.pgm", 0.4);

	let goal = SquareGoal::new(vec![([-0.75, 0.75], bitvec![1, 0, 0, 0, 0, 0, 0, 0]),
									([-0.25, 0.75], bitvec![0, 1, 0, 0, 0, 0, 0, 0]),
									([ 0.25, 0.75], bitvec![0, 0, 1, 0, 0, 0, 0, 0]),
									([ 0.75, 0.75], bitvec![0, 0, 0, 1, 0, 0, 0, 0]),
									([-0.75, 0.25], bitvec![0, 0, 0, 0, 1, 0, 0, 0]),
									([-0.25, 0.25], bitvec![0, 0, 0, 0, 0, 1, 0, 0]),
									([ 0.25, 0.25], bitvec![0, 0, 0, 0, 0, 0, 1, 0]),
									([ 0.75, 0.25], bitvec![0, 0, 0, 0, 0, 0, 0, 1])],
									 0.05);

	let mut prm = PRM::new(ContinuousSampler::new([-1.0, -1.0], [1.0, 1.0]),
						DiscreteSampler::new(),
						&m);

	prm.grow_graph(&[0.0, -0.8], &goal, 0.05, 5.0, 5000, 100000).expect("graph not grown up to solution");
	prm.print_summary();

	let policy = prm.plan_belief_space(&vec![1.0/8.0; 8]);

	let mut m2 = m.clone();
	m2.resize(5);
	m2.draw_full_graph(&prm.graph);
	m2.draw_zones_observability();
	m2.draw_policy(&policy);
	m2.save("results/map5/test_map5_8_goals_pomdp");
}

fn test_plan_on_map5_9_goals() {
	let mut m = MapShelfDomain::open("data/map5.pgm", [-1.0, -1.0], [1.0, 1.0]);
	m.add_zones("data/map5_9_goals_zone_ids.pgm", 0.4);

	let goal = SquareGoal::new(vec![([-0.75, 0.75], bitvec![1, 0, 0, 0, 0, 0, 0, 0, 0]),
									([-0.25, 0.75], bitvec![0, 1, 0, 0, 0, 0, 0, 0, 0]),
									([ 0.25, 0.75], bitvec![0, 0, 1, 0, 0, 0, 0, 0, 0]),
									([ 0.75, 0.75], bitvec![0, 0, 0, 1, 0, 0, 0, 0, 0]),
									([-0.75, 0.25], bitvec![0, 0, 0, 0, 1, 0, 0, 0, 0]),
									([-0.25, 0.25], bitvec![0, 0, 0, 0, 0, 1, 0, 0, 0]),
									([ 0.25, 0.25], bitvec![0, 0, 0, 0, 0, 0, 1, 0, 0]),
									([ 0.75, 0.25], bitvec![0, 0, 0, 0, 0, 0, 0, 1, 0]),
									([-0.75, -0.25],bitvec![0, 0, 0, 0, 0, 0, 0, 0, 1])],
									 0.05);

	let mut prm = PRM::new(ContinuousSampler::new([-1.0, -1.0], [1.0, 1.0]),
						DiscreteSampler::new(),
						&m);

	prm.grow_graph(&[0.0, -0.8], &goal, 0.05, 5.0, 5000, 100000).expect("graph not grown up to solution");
	prm.print_summary();

	let policy = prm.plan_belief_space(&vec![1.0/9.0; 9]);

	let mut policy_refiner = PRMPolicyRefiner::new(&policy, &m, &prm.belief_graph);
	let (refined_policy, _) = policy_refiner.refine_solution(RefinmentStrategy::PartialShortCut(1500));

	let mut m2 = m.clone();
	m2.resize(5);
	m2.draw_full_graph(&prm.graph);
	m2.draw_zones_observability();
	m2.draw_policy(&refined_policy);
	m2.save("results/map5/test_map5_9_goals_pomdp");
}
									
fn test_plan_on_map6_9_goals() {
	let mut m = MapShelfDomain::open("data/map6.pgm", [-1.0, -1.0], [1.0, 1.0]);
	m.add_zones("data/map6_9_goals_zone_ids.pgm", 0.4);

	
	let goal = SquareGoal::new(vec![([-0.5, 0.75], bitvec![1, 0, 0, 0, 0, 0, 0, 0, 0]),
									([-0.0, 0.75], bitvec![0, 1, 0, 0, 0, 0, 0, 0, 0]),
									([ 0.5, 0.75], bitvec![0, 0, 1, 0, 0, 0, 0, 0, 0]),
									([-0.5, 0.25], bitvec![0, 0, 0, 1, 0, 0, 0, 0, 0]),
									([-0.0, 0.25], bitvec![0, 0, 0, 0, 1, 0, 0, 0, 0]),
									([ 0.5, 0.25], bitvec![0, 0, 0, 0, 0, 1, 0, 0, 0]),
									([-0.5, -0.25],bitvec![0, 0, 0, 0, 0, 0, 1, 0, 0]),
									([-0.0, -0.25],bitvec![0, 0, 0, 0, 0, 0, 0, 1, 0]),
									([ 0.5, -0.25],bitvec![0, 0, 0, 0, 0, 0, 0, 0, 1])],
										0.05);

	let mut prm = PRM::new(ContinuousSampler::new([-1.0, -1.0], [1.0, 1.0]),
						DiscreteSampler::new(),
						&m);

	prm.grow_graph(&[0.0, -0.8], &goal, 0.05, 5.0, 5000, 100000).expect("graph not grown up to solution");
	prm.print_summary();

	//let initial_belief_state = vec![1.0/9.0; 9];
	let initial_belief_state = normalize_belief(&vec![0.1, 1.0, 1.0, 0.1, 1.0, 1.0, 0.1, 1.0, 1.0]); // skewed towards right
	//let initial_belief_state = normalize_belief(&vec![1.0, 1.0, 0.1, 1.0, 1.0, 0.1, 1.0, 1.0, 0.1]); // skewed towards left

	let policy = prm.plan_belief_space(&initial_belief_state);
	let mut policy_refiner = PRMPolicyRefiner::new(&policy, &m, &prm.belief_graph);
	let (refined_policy, _) = policy_refiner.refine_solution(RefinmentStrategy::PartialShortCut(1500));

	let mut m2 = m.clone();
	m2.resize(5);
	m2.draw_full_graph(&prm.graph);
	m2.draw_zones_observability();
	m2.draw_policy(&refined_policy);
	m2.save("results/map6/test_map6_9_goals_pomdp");
}

// typically intractable
fn test_plan_on_map5_12_goals() {
	let mut m = MapShelfDomain::open("data/map5.pgm", [-1.0, -1.0], [1.0, 1.0]);
	m.add_zones("data/map5_12_goals_zone_ids.pgm", 0.2);

	let goal = SquareGoal::new(vec![([-0.75, 0.75], bitvec![1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
									([-0.25, 0.75], bitvec![0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
									([ 0.25, 0.75], bitvec![0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
									([ 0.75, 0.75], bitvec![0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]),
									([-0.75, 0.25], bitvec![0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]),
									([-0.25, 0.25], bitvec![0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]),
									([ 0.25, 0.25], bitvec![0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]),
									([ 0.75, 0.25], bitvec![0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]),
									([-0.75, -0.25], bitvec![0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]),
									([-0.25, -0.25], bitvec![0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]),
									([ 0.25, -0.25], bitvec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]),
									([ 0.75, -0.25], bitvec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])],
									 0.05);

	let mut prm = PRM::new(ContinuousSampler::new([-1.0, -1.0], [1.0, 1.0]),
						DiscreteSampler::new(),
						&m);

	prm.grow_graph(&[0.0, -0.8], &goal, 0.05, 5.0, 5000, 100000).expect("graph not grown up to solution");
	prm.print_summary();

	let policy = prm.plan_belief_space(&vec![1.0/12.0, 1.0/12.0, 1.0/12.0, 1.0/12.0, 1.0/12.0, 1.0/12.0, 1.0/12.0, 1.0/12.0, 1.0/12.0, 1.0/12.0, 1.0/12.0, 1.0/12.0]);

	let mut m2 = m.clone();
	m2.resize(5);
	m2.draw_full_graph(&prm.graph);
	m2.draw_zones_observability();
	m2.draw_policy(&policy);
	m2.save("results/test_map5_12_goals_pomdp");
}

fn test_plan_on_map1_2_goals(n_iter_min: usize) -> (f64, f64) {
	let mut m = MapShelfDomain::open("data/map1_2_goals.pgm", [-1.0, -1.0], [1.0, 1.0]);
	m.add_zones("data/map1_2_goals_zone_ids.pgm", 0.5);

	
	let goal = SquareGoal::new(vec![([0.68, -0.45], bitvec![1, 0]),
									([0.68, 0.38], bitvec![0, 1])], 0.05);

	let mut prm = PRM::new(ContinuousSampler::new([-1.0, -1.0], [1.0, 1.0]),
						DiscreteSampler::new(),
						&m);

	let start = Instant::now();

	prm.grow_graph(&[-0.9, 0.0], &goal, 0.1, 2.0, n_iter_min, 100000).expect("graph not grown up to solution");
	prm.print_summary();

	let initial_belief_state = vec![1.0/2.0; 2];

	let policy = prm.plan_belief_space(&initial_belief_state);
	let mut policy_refiner = PRMPolicyRefiner::new(&policy, &m, &prm.belief_graph);
	let (refined_policy, _) = policy_refiner.refine_solution(RefinmentStrategy::PartialShortCut(1500));

	let duration = start.elapsed();
	println!("duration:{:?}", duration);

	//refined_policy.print();
	println!("refined policy cost:{}", refined_policy.expected_costs);

	let mut m2 = m.clone();
	m2.resize(5);
	m2.draw_full_graph(&prm.graph);
	m2.draw_zones_observability();
	m2.draw_policy(&policy);
	m2.draw_policy(&refined_policy);
	m2.save("results/map7/test_plan_on_map1_2_goals");

	(duration.as_secs_f64(), refined_policy.expected_costs)
}

fn test_plan_tamp_rrt_on_map1_2_goals(n_iter_min: usize) -> (f64, f64) {
	let mut m = MapShelfDomain::open("data/map1_2_goals.pgm", [-1.0, -1.0], [1.0, 1.0]);
	m.add_zones("data/map1_2_goals_zone_ids.pgm", 0.5);

	let mut tamp_rrt = MapShelfDomainTampRRT::new(ContinuousSampler::new_true_random([-1.0, -1.0], [1.0, 1.0]), DiscreteSampler::new(), &m, 0.05);	
	
	let start = Instant::now();

	let initial_belief_state = vec![1.0/2.0; 2];
	let policy = tamp_rrt.plan(&[-0.9, 0.0], &initial_belief_state, 0.1, 2.0, n_iter_min, 10000, TampSearch::BranchAndBound);
	let policy = policy.expect("nopath tree found!");

	let duration = start.elapsed();
	println!("duration:{:?}", duration);

	let mut m2 = m.clone();
	m2.resize(5);
	m2.draw_zones_observability();
	m2.draw_policy(&policy);
	m2.save("results/map7/test_plan_tamp_rrt_on_map1_2_goals");

	(duration.as_secs_f64(), policy.expected_costs)
}
	
// 2 goals
fn test_plan_on_map_benchmark_2_goals(n_iter_min: usize) -> (f64, f64) {
	let mut m = MapShelfDomain::open("data/map_benchmark.pgm", [-1.0, -1.0], [1.0, 1.0]);
	m.add_zones("data/map_benchmark_2_goals_zone_ids.pgm", 0.5);

	
	let goal = SquareGoal::new(vec![([-0.9, 0.0], bitvec![1, 0]),
									([ 0.9, 0.0], bitvec![0, 1])],
									0.05);

	let mut prm = PRM::new(ContinuousSampler::new_true_random([-1.0, -1.0], [1.0, 1.0]),
						DiscreteSampler::new_true_random(),
						&m);

	let start = Instant::now();

	prm.grow_graph(&[0.0, -1.0], &goal, 0.1, 2.0, n_iter_min, 100000).expect("graph not grown up to solution");
	prm.print_summary();

	let initial_belief_state = vec![1.0/2.0; 2];

	let policy = prm.plan_belief_space(&initial_belief_state);
	let mut policy_refiner = PRMPolicyRefiner::new(&policy, &m, &prm.belief_graph);
	let (refined_policy, _) = policy_refiner.refine_solution(RefinmentStrategy::PartialShortCut(1500));

	let duration = start.elapsed();

	let mut m2 = m.clone();
	m2.resize(5);
	m2.draw_full_graph(&prm.graph);
	m2.draw_zones_observability();
	m2.draw_policy(&policy);
	m2.draw_policy(&refined_policy);
	m2.save("results/map_benchmark/test_2_goals_prm");

	(duration.as_secs_f64(), refined_policy.expected_costs)
}

fn test_plan_tamp_rrt_on_map_benchmark_2_goals(n_iter_min: usize) -> (f64, f64) {
	let mut m = MapShelfDomain::open("data/map_benchmark.pgm", [-1.0, -1.0], [1.0, 1.0]);
	m.add_zones("data/map_benchmark_2_goals_zone_ids.pgm", 0.5);

	let mut tamp_rrt = MapShelfDomainTampRRT::new(ContinuousSampler::new_true_random([-1.0, -1.0], [1.0, 1.0]), DiscreteSampler::new(), &m, 0.05);	
	
	let start = Instant::now();

	let initial_belief_state = vec![1.0/2.0; 2];
	let policy = tamp_rrt.plan(&[0.0, -1.0], &initial_belief_state, 0.1, 2.0, n_iter_min, 10000, TampSearch::BranchAndBound);
	let policy = policy.expect("nopath tree found!");

	let duration = start.elapsed();

	let mut m2 = m.clone();
	m2.resize(5);
	m2.draw_zones_observability();
	m2.draw_policy(&policy);
	m2.save("results/map_benchmark/test_2_goals_tamp_rrt");

	(duration.as_secs_f64(), policy.expected_costs)
}

// 4 goals
fn test_plan_on_map_benchmark_4_goals(n_iter_min: usize) -> (f64, f64) {
	let mut m = MapShelfDomain::open("data/map_benchmark.pgm", [-1.0, -1.0], [1.0, 1.0]);
	m.add_zones("data/map_benchmark_4_goals_zone_ids.pgm", 0.5);

	
	let goal = SquareGoal::new(vec![([-0.9,-0.5], bitvec![1, 0, 0, 0]),
									([-0.9, 0.5], bitvec![0, 1, 0, 0]),
									([ 0.9, 0.5], bitvec![0, 0, 1, 0]),
									([ 0.9,-0.5], bitvec![0, 0, 0, 1])],
										0.05);

	let mut prm = PRM::new(ContinuousSampler::new_true_random([-1.0, -1.0], [1.0, 1.0]),
						DiscreteSampler::new_true_random(),
						&m);

	let start = Instant::now();

	prm.grow_graph(&[0.0, -1.0], &goal, 0.1, 2.0, n_iter_min, 100000).expect("graph not grown up to solution");
	prm.print_summary();

	let initial_belief_state = vec![1.0/4.0; 4];

	let policy = prm.plan_belief_space(&initial_belief_state);
	let mut policy_refiner = PRMPolicyRefiner::new(&policy, &m, &prm.belief_graph);
	let (refined_policy, _) = policy_refiner.refine_solution(RefinmentStrategy::PartialShortCut(1500));

	let duration = start.elapsed();

	let mut m2 = m.clone();
	m2.resize(5);
	m2.draw_full_graph(&prm.graph);
	m2.draw_zones_observability();
	m2.draw_policy(&policy);
	m2.draw_policy(&refined_policy);
	m2.save("results/map_benchmark/test_4_goals_prm");

	(duration.as_secs_f64(), refined_policy.expected_costs)
}

fn test_plan_tamp_rrt_on_map_benchmark_4_goals(n_iter_min: usize) -> (f64, f64) {
	let mut m = MapShelfDomain::open("data/map_benchmark.pgm", [-1.0, -1.0], [1.0, 1.0]);
	m.add_zones("data/map_benchmark_4_goals_zone_ids.pgm", 0.5);

	let mut tamp_rrt = MapShelfDomainTampRRT::new(ContinuousSampler::new_true_random([-1.0, -1.0], [1.0, 1.0]), DiscreteSampler::new(), &m, 0.05);	
	
	let start = Instant::now();

	let initial_belief_state = vec![1.0/4.0; 4];
	let policy = tamp_rrt.plan(&[0.0, -1.0], &initial_belief_state, 0.1, 2.0, n_iter_min, 10000, TampSearch::BranchAndBound);
	let policy = policy.expect("nopath tree found!");

	let duration = start.elapsed();

	let mut m2 = m.clone();
	m2.resize(5);
	m2.draw_zones_observability();
	m2.draw_policy(&policy);
	m2.save("results/map_benchmark/test_4_goals_tamp_rrt");

	(duration.as_secs_f64(), policy.expected_costs)
}

// 6 goals
fn test_plan_on_map_benchmark_6_goals(n_iter_min: usize) -> (f64, f64) {
	let mut m = MapShelfDomain::open("data/map_benchmark.pgm", [-1.0, -1.0], [1.0, 1.0]);
	m.add_zones("data/map_benchmark_6_goals_zone_ids.pgm", 0.5);

	
	let goal = SquareGoal::new(vec![([-0.9,-0.5], bitvec![1, 0, 0, 0, 0, 0]),
									([-0.9, 0.5], bitvec![0, 1, 0, 0, 0, 0]),
									([-0.5, 0.9], bitvec![0, 0, 1, 0, 0, 0]),
									([ 0.5, 0.9], bitvec![0, 0, 0, 1, 0, 0]),
									([ 0.9, 0.5], bitvec![0, 0, 0, 0, 1, 0]),
									([ 0.9,-0.5], bitvec![0, 0, 0, 0, 0, 1])],
										0.05);

	let mut prm = PRM::new(ContinuousSampler::new_true_random([-1.0, -1.0], [1.0, 1.0]),
						DiscreteSampler::new_true_random(),
						&m);

	let start = Instant::now();

	prm.grow_graph(&[0.0, -1.0], &goal, 0.1, 2.0, n_iter_min, 100000).expect("graph not grown up to solution");
	prm.print_summary();

	let initial_belief_state = vec![1.0/6.0; 6];
	//let initial_belief_state = normalize_belief(&vec![0.1, 0.1, 0.1, 0.1, 0.1, 1.0]); // skewed towards right
	//let initial_belief_state = normalize_belief(&vec![1.0, 1.0, 0.1, 1.0, 1.0, 0.1, 1.0, 1.0, 0.1]); // skewed towards left

	let policy = prm.plan_belief_space(&initial_belief_state);
	let mut policy_refiner = PRMPolicyRefiner::new(&policy, &m, &prm.belief_graph);
	let (refined_policy, _) = policy_refiner.refine_solution(RefinmentStrategy::PartialShortCut(1500));

	let duration = start.elapsed();

	let mut m2 = m.clone();
	m2.resize(5);
	m2.draw_full_graph(&prm.graph);
	m2.draw_zones_observability();
	m2.draw_policy(&policy);
	m2.draw_policy(&refined_policy);
	m2.save("results/map_benchmark/test_6_goals_prm");

	(duration.as_secs_f64(), refined_policy.expected_costs)
}

fn test_plan_tamp_rrt_on_map_benchmark_6_goals(n_iter_min: usize) -> (f64, f64) {
	let mut m = MapShelfDomain::open("data/map_benchmark.pgm", [-1.0, -1.0], [1.0, 1.0]);
	m.add_zones("data/map_benchmark_6_goals_zone_ids.pgm", 0.5);

	let mut tamp_rrt = MapShelfDomainTampRRT::new(ContinuousSampler::new_true_random([-1.0, -1.0], [1.0, 1.0]), DiscreteSampler::new_true_random(), &m, 0.05);	
	
	let start = Instant::now();

	let initial_belief_state = vec![1.0/6.0; 6];
	let policy = tamp_rrt.plan(&[0.0, -1.0], &initial_belief_state, 0.1, 2.0, n_iter_min, 10000, TampSearch::BranchAndBound);
	let policy = policy.expect("nopath tree found!");

	let duration = start.elapsed();

	let mut m2 = m.clone();
	m2.resize(5);
	m2.draw_zones_observability();
	m2.draw_policy(&policy);
	m2.save("results/map_benchmark/test_6_goals_tamp_rrt");

	(duration.as_secs_f64(), policy.expected_costs)
}

// 8 goals
fn test_plan_on_map_benchmark_8_goals(n_iter_min: usize) -> (f64, f64) {
	let mut m = MapShelfDomain::open("data/map_benchmark.pgm", [-1.0, -1.0], [1.0, 1.0]);
	m.add_zones("data/map_benchmark_8_goals_zone_ids.pgm", 0.5);

	let goal = SquareGoal::new(vec![([-0.9,-0.5], bitvec![1, 0, 0, 0, 0, 0, 0, 0]),
									([-0.9, 0.0], bitvec![0, 1, 0, 0, 0, 0, 0, 0]),
									([-0.9, 0.5], bitvec![0, 0, 1, 0, 0, 0, 0, 0]),
									([-0.5, 0.9], bitvec![0, 0, 0, 1, 0, 0, 0, 0]),
									([ 0.5, 0.9], bitvec![0, 0, 0, 0, 1, 0, 0, 0]),
									([ 0.9, 0.5], bitvec![0, 0, 0, 0, 0, 1, 0, 0]),
									([ 0.9, 0.0], bitvec![0, 0, 0, 0, 0, 0, 1, 0]),
									([ 0.9,-0.5], bitvec![0, 0, 0, 0, 0, 0, 0, 1])],
										0.05);

	let mut prm = PRM::new(ContinuousSampler::new_true_random([-1.0, -1.0], [1.0, 1.0]),
						DiscreteSampler::new_true_random(),
						&m);

	let start = Instant::now();

	prm.grow_graph(&[0.0, -1.0], &goal, 0.1, 2.0, n_iter_min, 100000).expect("graph not grown up to solution");
	prm.print_summary();

	let initial_belief_state = vec![1.0/8.0; 8];
	//let initial_belief_state = normalize_belief(&vec![0.1, 0.1, 0.1, 0.1, 0.1, 1.0]); // skewed towards right
	//let initial_belief_state = normalize_belief(&vec![1.0, 1.0, 0.1, 1.0, 1.0, 0.1, 1.0, 1.0, 0.1]); // skewed towards left

	let policy = prm.plan_belief_space(&initial_belief_state);
	let mut policy_refiner = PRMPolicyRefiner::new(&policy, &m, &prm.belief_graph);
	let (refined_policy, _) = policy_refiner.refine_solution(RefinmentStrategy::PartialShortCut(1500));

	let duration = start.elapsed();

	let mut m2 = m.clone();
	m2.resize(5);
	m2.draw_full_graph(&prm.graph);
	m2.draw_zones_observability();
	m2.draw_policy(&policy);
	m2.draw_policy(&refined_policy);
	m2.save("results/map_benchmark/test_8_goals_prm");

	(duration.as_secs_f64(), refined_policy.expected_costs)
}

fn test_plan_tamp_rrt_on_map_benchmark_8_goals(n_iter_min: usize) -> (f64, f64) {
	let mut m = MapShelfDomain::open("data/map_benchmark.pgm", [-1.0, -1.0], [1.0, 1.0]);
	m.add_zones("data/map_benchmark_8_goals_zone_ids.pgm", 0.5);

	let mut tamp_rrt = MapShelfDomainTampRRT::new(ContinuousSampler::new_true_random([-1.0, -1.0], [1.0, 1.0]), DiscreteSampler::new_true_random(), &m, 0.05);	
	
	let start = Instant::now();

	let initial_belief_state = vec![1.0/8.0; 8];
	let policy = tamp_rrt.plan(&[0.0, -1.0], &initial_belief_state, 0.1, 2.0, n_iter_min, 10000, TampSearch::BranchAndBound);
	let policy = policy.expect("nopath tree found!");

	let duration = start.elapsed();

	let mut m2 = m.clone();
	m2.resize(5);
	m2.draw_zones_observability();
	m2.draw_policy(&policy);
	m2.save("results/map_benchmark/test_8_goals_tamp_rrt");

	(duration.as_secs_f64(), policy.expected_costs)
}