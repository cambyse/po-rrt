use itertools::{all, enumerate, izip, merge, zip};

use crate::common::*;
use crate::nearest_neighbor::*;
use crate::sample_space::*;
use crate::map_io::*;
use std::cmp::min;


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
}

impl<'a, F: PRMFuncs<N>, const N: usize> PRM<'a, F, N> {
	pub fn new(sample_space: SampleSpace<N>, fns: &'a F) -> Self {
		Self { sample_space, fns }
	}

	pub fn plan(&mut self, start: [f64; N], goal: fn(&[f64; N]) -> bool,
				 max_step: f64, search_radius: f64, n_iter_max: u32) -> (Result<Vec<[f64; N]>, &'static str>, PRMGraph<N>) {
		let (graph, final_node_ids) = self.grow_graph(start, goal, max_step, search_radius, n_iter_max);

		(self.get_best_solution(&graph, &final_node_ids), graph)
	}

	fn grow_graph(&self, start: [f64; N], goal: fn(&[f64; N]) -> bool,
				max_step: f64, search_radius: f64, n_iter_max: u32) -> (PRMGraph<N>, Vec<usize>) {
		let mut final_node_ids = Vec::<usize>::new();
		let mut graph = PRMGraph::new(start);
		let mut kdtree = KdTree::new(start);

		for _ in 0..n_iter_max {
			let mut new_state = self.sample_space.sample();
			let kd_from = kdtree.nearest_neighbor(new_state);

			steer(&kd_from.state, &mut new_state, max_step);

			let state_validity = self.fns.state_validity(&new_state);
			if state_validity.is_some() {
				// First, add node
				let new_node_id = graph.add_node(new_state, state_validity.unwrap());
				kdtree.add(new_state, new_node_id);

				// Second, we find the neighbors in a specific radius of new_state.
				let radius = {
					let n = graph.nodes.len() as f64;
					let s = search_radius * (n.ln()/n).powf(1.0/(N as f64));
					if s < max_step { s } else { max_step }
				};

				let neighbour_ids: Vec<usize> = kdtree.nearest_neighbors(new_state, radius).iter()
					.filter(|node| self.fns.transition_validator(&node.state, &new_state))
					.map(|node| node.id)
					.collect();
				
				// Third, connect to neighbors if transition possible
				for neighbor_id in neighbour_ids {
					let neighbor_state = &graph.nodes[neighbor_id].state;
					if self.fns.transition_validator(&neighbor_state, &new_state) {
						graph.add_edge(neighbor_id, new_node_id);
					}
				}

				if goal(&new_state) {
					final_node_ids.push(new_node_id);
				}
			}
		}

		(graph, final_node_ids)
	}

	fn get_best_solution(&self, _: &PRMGraph<N>, _: &Vec<usize>) -> Result<Vec<[f64; N]>, &'static str> {
		Err("")
		/*final_node_ids.iter()
			.map(|id| {
				let path = rrttree.get_path_to(*id);
				let cost = self.get_path_cost(&path);
				(path, cost)
			})
			.min_by(|(_,a),(_,b)| a.partial_cmp(b).expect("NaN found"))
			.map(|(p, _)| p)
			.ok_or("No solution found")*/
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
	let (_, graph) = prm.plan([0.55, -0.8], goal, 0.05, 5.0, 2500);

	//let mut full = m.clone();
	//full.draw_full_graph(&graph);
	//full.save("results/test_prm_full_graph.pgm");

	let world = 1;
	let mut from = m.clone();
	from.set_world(world);
	from.draw_graph_for_world(&graph, world);
	from.save("results/test_prm_graph_from.pgm");
}
}

/*
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
* Option 3/ Monte Carlo
*
* EXECUTION()
* - Robot a un belief state : probabilité d'être dans chaque monde
* - Deduit le chemin a suivre en cherchant en calculant l'espérance de la distance à l'objectif de chacun de ces enfants
*
* QUESTIONS:
* - Quand arreter de faire croitre le graph et quand même avoir une solution our chaque monde??
* - Biais pour sampling basé sur heristique
*/