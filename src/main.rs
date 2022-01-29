#![allow(dead_code, unused_imports)]
#![feature(slice_group_by)]

use po_rrt::{
    prm::*,
    prm_graph::*,
    sample_space::*,
    map_io::*,
	map_shelves_io::*,
	common::*,
	prm_policy_refiner::*
};
use bitvec::prelude::*;

fn main()
{
	test_plan_on_map7_6_goals();
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

									
fn test_plan_on_map7_6_goals() {
	let mut m = MapShelfDomain::open("data/map7.pgm", [-1.0, -1.0], [1.0, 1.0]);
	m.add_zones("data/map7_6_goals_zone_ids.pgm", 0.5);

	
	let goal = SquareGoal::new(vec![([-0.9,-0.5], bitvec![1, 0, 0, 0, 0, 0]),
									([-0.9, 0.5], bitvec![0, 1, 0, 0, 0, 0]),
									([-0.5, 0.9], bitvec![0, 0, 1, 0, 0, 0]),
									([ 0.5, 0.9], bitvec![0, 0, 0, 1, 0, 0]),
									([ 0.9, 0.5], bitvec![0, 0, 0, 0, 1, 0]),
									([ 0.9,-0.5], bitvec![0, 0, 0, 0, 0, 1])],
										0.05);

	let mut prm = PRM::new(ContinuousSampler::new([-1.0, -1.0], [1.0, 1.0]),
						DiscreteSampler::new(),
						&m);

	prm.grow_graph(&[0.0, -0.8], &goal, 0.05, 5.0, 5000, 100000).expect("graph not grown up to solution");
	prm.print_summary();

	let initial_belief_state = vec![1.0/6.0; 6];
	//let initial_belief_state = normalize_belief(&vec![0.1, 1.0, 1.0, 0.1, 1.0, 1.0, 0.1, 1.0, 1.0]); // skewed towards right
	//let initial_belief_state = normalize_belief(&vec![1.0, 1.0, 0.1, 1.0, 1.0, 0.1, 1.0, 1.0, 0.1]); // skewed towards left

	let policy = prm.plan_belief_space(&initial_belief_state);
	let mut policy_refiner = PRMPolicyRefiner::new(&policy, &m, &prm.belief_graph);
	let (refined_policy, _) = policy_refiner.refine_solution(RefinmentStrategy::PartialShortCut(1500));

	let mut m2 = m.clone();
	m2.resize(5);
	m2.draw_full_graph(&prm.graph);
	m2.draw_zones_observability();
	m2.draw_policy(&refined_policy);
	m2.save("results/map7/test_map7_6_goals_pomdp");
}