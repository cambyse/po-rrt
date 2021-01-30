use itertools::{all, enumerate, izip, merge, zip};

use crate::common::*;
use crate::nearest_neighbor::*;
use crate::sample_space::*;
use crate::map_io::*;
use std::{cmp::min, vec};


pub struct PRMNode<const N: usize> {
	pub state: [f64; N],
	pub validity: Vec<bool>,
	pub children: Vec<usize>,
}

pub struct PRMGraph<const N: usize> {
	pub nodes: Vec<PRMNode<N>>,
}

impl<const N: usize> PRMGraph<N> {
	fn new(state: [f64; N]) -> Self {
		let root = PRMNode { state, validity: Vec::new() , children: Vec::new() };
		Self { nodes: vec![root] }
	}

	fn add_node(&mut self, state: [f64; N], state_validity: Vec<bool>) -> usize {
		let id = self.nodes.len();
		let node = PRMNode { state, validity: state_validity, children: Vec::new() };
		self.nodes.push(node);
		id
	}

	fn add_edge(&mut self, from_id: usize, to_id: usize) {
		self.nodes[from_id].children.push(to_id);
	}

	fn get_path_to(&self, _: usize) -> Vec<[f64; N]> {
		let path = Vec::new();

		path
	}
}

pub trait PRMFuncs<const N: usize> {
	fn state_validity(&self, _state: &[f64; N]) -> Option<Vec<bool>> {
		None
	}

	fn transition_validator(&self, _from: &[f64; N], _to: &[f64; N]) -> bool {
		true
	}

	fn cost_evaluator(&self, a: &[f64; N], b: &[f64; N]) -> f64 {
		norm2(a,b)
	}
}

pub struct PRM<'a, F: PRMFuncs<N>, const N: usize> {
	sample_space: SampleSpace<N>,
	fns: &'a F,
	graph: PRMGraph<N>
}

impl<'a, F: PRMFuncs<N>, const N: usize> PRM<'a, F, N> {
	pub fn new(sample_space: SampleSpace<N>, fns: &'a F) -> Self {
		Self { sample_space, fns, graph: PRMGraph{nodes: vec![]} }
	}

	pub fn grow_graph(&mut self, start: &[f64; N],
				max_step: f64, search_radius: f64, n_iter_max: u32) {
		let mut kdtree = KdTree::new(*start);
		self.graph.add_node(*start, self.fns.state_validity(&start).expect("Start from a valid state!"));

		for i in 0..n_iter_max {
			if i % 100 == 0 {
				println!("number of iterations:{}", i);
			}

			let mut new_state = self.sample_space.sample();
			let kd_from = kdtree.nearest_neighbor(new_state);

			steer(&kd_from.state, &mut new_state, max_step); 

			let state_validity = self.fns.state_validity(&new_state);
			if state_validity.is_some() {
				// First, add node
				let new_node_id = self.graph.add_node(new_state, state_validity.unwrap());
				kdtree.add(new_state, new_node_id);

				// Second, we find the neighbors in a specific radius of new_state.
				let radius = {
					let n = self.graph.nodes.len() as f64;
					let s = search_radius * (n.ln()/n).powf(1.0/(N as f64));
					if s < max_step { s } else { max_step }
				};

				let neighbour_ids: Vec<usize> = kdtree.nearest_neighbors(new_state, radius).iter()
					.filter(|node| self.fns.transition_validator(&node.state, &new_state))
					.map(|node| node.id)
					.collect();
				
				// Third, connect to neighbors if transition possible
				for neighbor_id in neighbour_ids {
					let neighbor_state = &self.graph.nodes[neighbor_id].state;
					if self.fns.transition_validator(&neighbor_state, &new_state) {
						self.graph.add_edge(neighbor_id, new_node_id);
					}
				}
			}
		}
	}

	pub fn plan(&mut self, start: &[f64; N], prior: &Vec<f64>) -> Result<Vec<[f64; N]>, &'static str> {
		Err("")
	}

	fn get_path_cost(&self, path: &Vec<[f64; N]>) -> f64 {
		pairwise_iter(path)
			.map(|(a,b)| self.fns.cost_evaluator(a,b))
			.sum()
	}
}

#[cfg(test)]
mod tests {

use super::*;

#[test]
fn test_plan_on_map() {
	let mut m = Map::open("data/map2.pgm", [-1.0, -1.0], [1.0, 1.0]);
	m.add_zones("data/map2_zone_ids.pgm");

	fn goal(state: &[f64; 2]) -> bool {
		(state[0] - 0.55).abs() < 0.05 && (state[1] - 0.9).abs() < 0.05
	}

	let mut prm = PRM::new(SampleSpace{low: [-1.0, -1.0], up: [1.0, 1.0]}, &m);
	prm.grow_graph(&[0.55, -0.8], 0.05, 5.0, 10_000);
	
	prm.plan(&[0.55, -0.8], &vec![0.25, 0.25, 0.25, 0.25]);

	// loop:
	// prm.plan(position, prior);
	// potentiallement adapter graph si on arrive dans un monde improbable lors du precompute
	
	//let mut full = m.clone();
	//full.draw_full_graph(&graph);
	//full.save("results/test_prm_full_graph.pgm");

	let world = 1; 
	let mut from = m.clone();
	from.set_world(world);
	from.draw_graph_for_world(&prm.graph, world);
	from.save("results/test_prm_graph_from.pgm");
}
}

/* N = nombre de mondes
* PLAN()
* Option 1/ Pour chaque monde, calculer la distance à l'objectif de chaque node (Value iteration)
* 	-> avantage : valeurs valables partout
*   -> long calcul offline - scalability?
*   d(n) = min( d(m, n) + d(m) for m in children(n) )
*
* Option 2/ Au runtime, calculer meilleur chemin pour chacun des mondes (e.g. A*) et prendre decision en pondérant chacunes des options 
*	-> avantage : étape offline moins longue
*   -> possible à l'exécution ? (assez rapide?)
*
* Option 3/ Monte Carlo -> N
* 
* Option 4/ N mondes: RRT start N fois
* 
*
* EXECUTION()
* - Robot a un belief state : probabilité d'être dans chaque monde
* - Deduit le chemin a suivre en cherchant en calculant l'espérance de la distance à l'objectif de chacun de ces enfants
*
* Belief state : [0.25, 0.25, 0.25, 0.25] 
* 
*
* QUESTIONS:
* - Quand arreter de faire croitre le graph et quand même avoir une solution our chaque monde??
* - Biais pour sampling basé sur heristique
*/

// Compresser pour avoir N mondes même dan domaines ou le nombre de mondes explosent