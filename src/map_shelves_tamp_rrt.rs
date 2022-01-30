use itertools::{all, enumerate, izip, merge, zip};

use crate::common::*;
use crate::nearest_neighbor::*;
use crate::sample_space::*;
use crate::map_shelves_io::*; // tests only
use bitvec::prelude::*;
use priority_queue::PriorityQueue;

#[derive(Clone)]
pub struct SearchNode {
	pub id: usize,
	pub state: [f64; 2],
    pub target_zone_id: Option<usize>,
	pub parent: Option<usize>,
    pub children: Vec<usize>,
	pub cost_from_root: f64,
	pub remaining_zones: Vec<usize>
}

pub struct SearchTree {
    pub nodes: Vec<SearchNode>,
}

pub struct MapShelfDomainTampRRT<'a> {
	continuous_sampler: ContinuousSampler<2>,
	pub map_shelves_domain: &'a MapShelfDomain,
	pub kdtree: KdTree<2>,
	pub n_worlds: usize,
	n_it: usize,
}

impl SearchTree {
	pub fn add_node(&mut self, parent_id: usize, state: [f64; 2], target_zone_id: usize, cost_from_root: f64, remaining_zones: &[usize]) -> usize {
		let id = self.nodes.len();
		let v = SearchNode{
			id,
			state,
			target_zone_id: Some(target_zone_id),
			parent: Some(parent_id),
			children: Vec::new(),
			cost_from_root,
			remaining_zones: remaining_zones.to_owned()
		};

		self.nodes.push(v);
		id
	}
}

impl<'a> MapShelfDomainTampRRT<'a> {
	pub fn new(continuous_sampler: ContinuousSampler<2>, map_shelves_domain: &'a MapShelfDomain) -> Self {
		Self { continuous_sampler,
			   map_shelves_domain, 
			   kdtree: KdTree::new([0.0; 2]),
			   n_worlds: map_shelves_domain.n_zones(), 
			   n_it: 0}
	}

	pub fn plan(&mut self, &start: &[f64; 2], _goal: &impl GoalFuncs<2>, _max_step: f64, _search_radius: f64, _n_iter_min: usize, _n_iter_max: usize) -> Result<(), &'static str> {
		let mut q = PriorityQueue::new();

		let root_node = SearchNode{
			id: 0,
			state: start,
			target_zone_id: Some(0),
			parent: None,
			children: Vec::new(),
			cost_from_root: 0.0,
			remaining_zones: (0..self.map_shelves_domain.n_zones()).collect()
		};

		q.push(root_node.id, Priority{prio: root_node.cost_from_root});

		let mut search_tree = SearchTree{
			nodes: vec![root_node]
		};

		let mut it = 0;
		while !q.is_empty() {
			it+=1;
			let (u_id, _) = q.pop().unwrap();
			let u = search_tree.nodes[u_id].clone();

			// expand
			for target_zone_id in &u.remaining_zones {
				let mut remaining_zones = u.remaining_zones.clone();
				remaining_zones.retain(|zone_id| zone_id != target_zone_id);
				
				// query motion planner
				// piece 1: go to target observe zone

				// piece 2: object is here: plan to reach goal corresponding to 

				//

				let v_id = search_tree.add_node(u.id, u.state, *target_zone_id, u.cost_from_root, &remaining_zones);
				let v = &search_tree.nodes[v_id];

				// addonly if not too bad
				q.push(v_id, Priority{prio: v.cost_from_root});
			}
		}

		println!("it:{}, nodes:{}", it, search_tree.nodes.len());
		println!("n leafs:{}", search_tree.nodes.iter().filter(|&n| n.remaining_zones.is_empty() ).count());

		Ok(())
	}

}

#[cfg(test)]
mod tests {

use super::*;

#[test]
fn test_plan_on_map2_pomdp() {
	let mut m = MapShelfDomain::open("data/map1_2_goals.pgm", [-1.0, -1.0], [1.0, 1.0]);
	m.add_zones("data/map1_2_goals_zone_ids.pgm", 0.5);

	let goal = SquareGoal::new(vec![([0.55, 0.9], bitvec![1; 4])], 0.05);

	let mut tamp_rrt = MapShelfDomainTampRRT::new(ContinuousSampler::new([-1.0, -1.0], [1.0, 1.0]), &m);			
	let _ = tamp_rrt.plan(&[0.0, -0.8], &goal, 0.05, 5.0, 5000, 100000);
}

#[test]
fn test_plan_on_map7() {
	let mut m = MapShelfDomain::open("data/map7.pgm", [-1.0, -1.0], [1.0, 1.0]);
	m.add_zones("data/map7_6_goals_zone_ids.pgm", 0.5);

	let goal = SquareGoal::new(vec![([-0.9,-0.5], bitvec![1, 0, 0, 0, 0, 0]),
									([-0.9, 0.5], bitvec![0, 1, 0, 0, 0, 0]),
									([-0.5, 0.9], bitvec![0, 0, 1, 0, 0, 0]),
									([ 0.5, 0.9], bitvec![0, 0, 0, 1, 0, 0]),
									([ 0.9, 0.5], bitvec![0, 0, 0, 0, 1, 0]),
									([ 0.9,-0.5], bitvec![0, 0, 0, 0, 0, 1])],
										0.05);

	let mut tamp_rrt = MapShelfDomainTampRRT::new(ContinuousSampler::new([-1.0, -1.0], [1.0, 1.0]), &m);			
	let _ = tamp_rrt.plan(&[0.0, -0.8], &goal, 0.05, 5.0, 5000, 100000);
}

}
