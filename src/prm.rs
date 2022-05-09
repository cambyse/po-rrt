use itertools::{all, enumerate, izip, merge, zip};

use crate::common::*;
use crate::nearest_neighbor::*;
use crate::sample_space::*;
use crate::map_io::*; // tests only
use crate::map_shelves_io::*; // tests only
use crate::map_shelves_io::colors; // tests only
use crate::pto_graph::*;
use std::time::Instant;


pub struct PRM<'a, F: PTOFuncs<N>, const N: usize> {
	continuous_sampler: ContinuousSampler<N>,
	pub fns: &'a F,
	pub kdtree: KdTree<N>,
	// prm graph growth
	pub graph: PTOGraph<N>,
	// debug
	pub n_it: usize
}

impl<'a, F: PTOFuncs<N>, const N: usize> PRM<'a, F, N> {
	pub fn new(continuous_sampler: ContinuousSampler<N>, fns: &'a F) -> Self {
		Self { continuous_sampler,
			   fns, 
			   kdtree: KdTree::new([0.0; N]),
			   graph: PTOGraph{nodes: vec![], validities: fns.world_validities()},
			   n_it: 0
			}
	}

	pub fn init(&mut self, &start: &[f64; N]) {
		self.graph.add_node(start, 0);
		self.kdtree.reset(start);
	}

	pub fn grow_graph(&mut self, max_step: f64, search_radius: f64, n_iter: usize) {

		println!("grow graph..");
		//let start_time = Instant::now();
		
		let mut i = 0;
		while i < n_iter { // order important! avoid calling the is_final_set_complete as long as the number of iterations is not reached!
			i+=1;
	
			// First sample state and world
			let new_state = self.continuous_sampler.sample();
			
			// Handle grow start case
			if self.graph.nodes.is_empty() {
				self.graph.add_node(new_state, 0);
				self.kdtree.reset(new_state);
				continue
			}

			// Add node
			let new_node_id = self.graph.add_node(new_state, 0);
			let new_node = &self.graph.nodes[new_node_id];

			if self.graph.nodes.is_empty() {
				self.graph.add_node(new_state, 0);
				self.kdtree.reset(new_state);
				continue;
			}

			// We find the neighbors in a specific radius of new_state.
			let radius = heuristic_radius(self.graph.nodes.len(), max_step, search_radius, N);

			//if i%100 == 0{
			//	println!("radius:{}", radius);
			//}

			// Finaly we connect to neighbors 
			let neighbour_ids: Vec<usize> = self.kdtree.nearest_neighbors(new_state, radius).iter()
			.map(|&kd_node| kd_node.id)
			.collect();

			self.kdtree.add(new_state, new_node_id);

			if neighbour_ids.is_empty() { 
				continue;
			}

			// No need to check differently the backward edges, assumption is pure path planning, and that paths are reversible
			// Idea: sample which ones we rewire to?
			let edges: Vec<(usize, usize)> = neighbour_ids.iter()
				.map(|&id| (id, &self.graph.nodes[id]))
				.map(|(id, node)| (id, self.fns.transition_validator(node, new_node)))
				.filter(|(_, validity_id)| validity_id.is_some())
				.map(|(id, validity_id)| (id, validity_id.unwrap()))
				.collect();
						
			// connect neighbors to new node
			for (id, _) in &edges {
				self.graph.add_edge(*id, new_node_id, 0);
			}

			// connect new node to neighbor
			for (id, _) in &edges {
				self.graph.add_edge(new_node_id, *id, 0);
			}

			self.n_it += 1;
		}
	}

	pub fn plan_path(&mut self, start: &[f64; N], goal: &[f64; N]) -> Vec<[f64; N]>  {
		let kd_start = self.kdtree.nearest_neighbor(*start);
		let kd_goal = self.kdtree.nearest_neighbor(*goal);

		let cost_to_goal = dijkstra(&self.graph, &vec![kd_goal.id], self.fns);
		extract_path(&self.graph, kd_start.id, &cost_to_goal, self.fns)
	}


	pub fn print_summary(&self) {
		//println!("number of iterations:{}", self.n_it);
		self.graph.print_summary();
	}
}

#[cfg(test)]
mod tests {

use super::*;

#[test]
fn test_plan_on_map0_prm() {
	let mut m = Map::open("data/map0.pgm", [-1.0, -1.0], [1.0, 1.0]);
	m.init_without_zones();

	let mut prm = PRM::new(ContinuousSampler::new([-1.0, -1.0], [1.0, 1.0]),
						   &m);

	prm.init(&[0.0, 0.0]);					   
	prm.grow_graph(0.1, 5.0, 2500);
	prm.print_summary();
	let path = prm.plan_path(&[0.0, 0.0], &[0.0, 0.9]);

	let mut m2 = m.clone();
	m2.resize(5);
	m2.draw_full_graph(&prm.graph);
	m2.draw_zones_observability();
	m2.draw_path(&path, colors::BLACK);
	m2.save("results/test_plan_on_map0_prm");
}
}