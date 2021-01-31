use itertools::{all, enumerate, izip, merge, zip};

use crate::common::*;
use crate::nearest_neighbor::*;
use crate::sample_space::*;
use crate::map_io::*;

/*****************************IO*****************************/
mod io {
	use std::{borrow::Borrow, cmp::min, convert::TryInto, vec};
	use std::io::BufWriter;
	use std::io::BufReader;
	use std::fs::File;
	
	extern crate serde;
	use serde::{Serialize, Deserialize};

	use super::*;

#[derive(Serialize, Deserialize)]
pub struct SerializablePRMNode {
	pub state: Vec<f64>,
	pub validity: Vec<bool>,
	pub children: Vec<usize>,
}

impl SerializablePRMNode {
	pub fn from_prm_node(node : &PRMNode<2>) -> Self {
		Self{
			state: node.state.to_vec(),
			validity: node.validity.clone(),
			children: node.children.clone()
		}
	}

	pub fn to_prm_node(&self) -> PRMNode<2> {
		PRMNode {
			state: self.state.clone().try_into().unwrap(),
			validity: self.validity.clone(),
			children: self.children.clone(),
		}
	}
}

#[derive(Serialize, Deserialize)]
pub struct SerializablePRMGraph {
	pub nodes: Vec<SerializablePRMNode>,
}

impl SerializablePRMGraph {
	pub fn from_prm_graph_(prm_graph: &PRMGraph<2>) -> SerializablePRMGraph {
		let nodes = &prm_graph.nodes;
		SerializablePRMGraph {
			nodes: nodes.into_iter().map(|node| SerializablePRMNode::from_prm_node(&node)).collect()
		}
	}

	pub fn save_(&self, filename: &str) {
		let writer = BufWriter::new(File::create(filename).unwrap());
		serde_json::to_writer_pretty(writer, &self).unwrap();
	}
}

	pub fn save(prm_graph: &PRMGraph<2>, filename: &str) {
		let graph = SerializablePRMGraph::from_prm_graph_(prm_graph);
		graph.save_(filename);
	}

	pub fn load(filename: &str) -> PRMGraph<2> {
		let reader = BufReader::new(File::open(filename).unwrap());
		let graph: SerializablePRMGraph = serde_json::from_reader(reader).unwrap();

		PRMGraph {
			nodes: graph.nodes.into_iter().map(|node| node.to_prm_node()).collect()
		}
	}
}

/************************************************************/

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

	fn transition_validator(&self, _from: &PRMNode<N>, _to: &PRMNode<N>) -> bool {
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
			let kd_from = kdtree.nearest_neighbor(new_state); // n log n

			steer(&kd_from.state, &mut new_state, max_step); 

			let state_validity = self.fns.state_validity(&new_state);
			if state_validity.is_some() {
				// First, add node
				let new_node_id = self.graph.add_node(new_state, state_validity.unwrap());
				let new_node = &self.graph.nodes[new_node_id];

				// Second, we find the neighbors in a specific radius of new_state.
				let radius = {
					let n = self.graph.nodes.len() as f64;
					let s = search_radius * (n.ln()/n).powf(1.0/(N as f64));
					if s < max_step { s } else { max_step }
				};

				let neighbour_ids: Vec<usize> = kdtree.nearest_neighbors(new_state, radius).iter()
					.map(|kd_node| (kd_node.id, &self.graph.nodes[kd_node.id]))	
					.filter(|(_, node)| self.fns.transition_validator(node, new_node))
					.map(|(id, _)| id)
					.collect();

				// Third, connect to neighbors if transition possible
				for neighbor_id in neighbour_ids {
					self.graph.add_edge(neighbor_id, new_node_id);
					/*let neighbor_state = &self.graph.nodes[neighbor_id].state;
					if self.fns.transition_validator(&new_state, &neighbor_state) {
						self.graph.add_edge(new_node_id, neighbor_id);
					}*/
				}

				kdtree.add(new_state, new_node_id);
			}
		}
	}

	pub fn plan(&mut self, _: &[f64; N], _: &Vec<f64>) -> Result<Vec<[f64; N]>, &'static str> {
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
	prm.grow_graph(&[0.55, -0.8], 0.05, 5.0, 2500);
	
	let _ = prm.plan(&[0.55, -0.8], &vec![0.25, 0.25, 0.25, 0.25]);

	// loop:
	// prm.plan(position, prior);
	// potentiallement adapter graph si on arrive dans un monde improbable lors du precompute

	io::save(&prm.graph, "results/prm.json");
	prm.graph = io::load("results/prm.json");

	let world = 1; 
	let mut full = m.clone();
	full.set_world(world);
	full.draw_full_graph(&prm.graph, world);
	//full.draw_graph_for_world(&prm.graph, world);
	full.save("results/test_prm_graph_from.pgm");


	/*let world = 0; 
	let mut from = m.clone();
	from.set_world(world);
	from.draw_graph_for_world(&prm.graph, world);
	from.save("results/test_prm_graph_from.pgm");*/
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