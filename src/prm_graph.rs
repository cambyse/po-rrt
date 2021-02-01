use itertools::{all, enumerate, izip, merge, zip};

use crate::common::*;
use crate::nearest_neighbor::*;
use crate::sample_space::*;
use crate::map_io::*;
use std::{cmp::min, collections::btree_set::Difference, convert::TryInto, vec};

use std::io::BufWriter;
use std::io::BufReader;
use std::fs::File;

extern crate serde;
use serde::{Serialize, Deserialize};

use priority_queue::PriorityQueue;

/***************************IO*****************************/

#[derive(Serialize, Deserialize)]
pub struct SerializablePRMNode {
	pub state: Vec<f64>,
	pub validity: Vec<bool>,
	pub parents: Vec<usize>,
	pub children: Vec<usize>,
}

impl SerializablePRMNode {
	pub fn from_prm_node(node : &PRMNode<2>) -> Self {
		Self{
			state: node.state.to_vec(),
			validity: node.validity.clone(),
			parents: node.parents.clone(),
			children: node.children.clone()
		}
	}

	pub fn to_prm_node(&self) -> PRMNode<2> {
		PRMNode {
			state: self.state.clone().try_into().unwrap(),
			validity: self.validity.clone(),
			parents: self.parents.clone(),
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
		let writer = BufWriter::new(File::create(filename).expect("can't create file"));
		serde_json::to_writer_pretty(writer, &self).expect("error happened while dumping prm graph to file");
	}
}

pub fn save(prm_graph: &PRMGraph<2>, filename: &str) {
	let graph = SerializablePRMGraph::from_prm_graph_(prm_graph);
	graph.save_(filename);
}

pub fn load(filename: &str) -> PRMGraph<2> {
	let reader = BufReader::new(File::open(filename).expect("impossible to open file"));
	let graph: SerializablePRMGraph = serde_json::from_reader(reader).unwrap();

	PRMGraph {
		nodes: graph.nodes.into_iter().map(|node| node.to_prm_node()).collect()
	}
}

/****************************PRM Graph******************************/
pub trait PRMFuncs<const N: usize> {
	fn state_validity(&self, _state: &[f64; N]) -> Option<Vec<bool>> {
		None
	}

	fn transition_validator(&self, from: &PRMNode<N>, to: &PRMNode<N>) -> bool {
		from.validity.iter().zip(&to.validity)
		.any(|(&a, &b)| a && b)
	}

	fn cost_evaluator(&self, a: &[f64; N], b: &[f64; N]) -> f64 {
		norm2(a,b)
	}
}

pub struct PRMNode<const N: usize> {
	pub state: [f64; N],
	pub validity: Vec<bool>,
	pub parents: Vec<usize>,
	pub children: Vec<usize>,
}

pub struct PRMGraph<const N: usize> {
	pub nodes: Vec<PRMNode<N>>,
}

impl<const N: usize> PRMGraph<N> {
	pub fn add_node(&mut self, state: [f64; N], state_validity: Vec<bool>) -> usize {
		let id = self.nodes.len();
		let node = PRMNode { state, validity: state_validity, parents: Vec::new(), children: Vec::new() };
		self.nodes.push(node);
		id
	}

	pub fn add_edge(&mut self, from_id: usize, to_id: usize) {
		self.nodes[from_id].children.push(to_id);
		self.nodes[to_id].parents.push(from_id);
	}

	pub fn get_path_to(&self, _: usize) -> Vec<[f64; N]> {
		let path = Vec::new();

		path
	}
}

/****************************Dijkstra******************************/
use std::cmp::Ordering;

struct Priority{
	prio: f64
}

impl Ord for Priority {
    fn cmp(&self, other: &Self) -> Ordering {
        if self.prio < other.prio { Ordering::Greater } else { Ordering::Less }
    }
}

impl PartialOrd for Priority {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for Priority {
    fn eq(&self, other: &Self) -> bool {
        self.prio == other.prio
    }
}

impl Eq for Priority {}

fn dijkstra<'a, F: PRMFuncs<N>, const N: usize>(graph: &PRMGraph<N>, final_node_ids: &Vec<usize>, world: usize, m: &F) -> Vec<f64> {
	let mut dist = vec![std::f64::INFINITY ;graph.nodes.len()];
	let mut prev = vec![0 ;graph.nodes.len()];
	let mut q = PriorityQueue::new();
	
	for &id in final_node_ids {
		dist[id] = 0.0;
		q.push(id, Priority{prio: 0.0});

		if ! graph.nodes[id].validity[world] {
			panic!("final nodes must be valid in considered world");
		}
	}

	while !q.is_empty() {
		let (u_id, _) = q.pop().unwrap();
		let u = &graph.nodes[u_id];
		
		for &v_id in &u.parents {
			let v = &graph.nodes[v_id];

			if v.validity[world] {
				let alt = dist[u_id] + m.cost_evaluator(&u.state, &v.state);

				if alt < dist[v_id] {
					dist[v_id] = alt;
					prev[v_id] = u_id;
					q.push(v_id, Priority{prio: alt});
				}
			}
		}
	}

	dist
}

/****************************Tests******************************/

#[cfg(test)]
mod tests {

use super::*;

fn create_minimal_graph() -> PRMGraph<2> {
	let mut graph = PRMGraph{nodes: Vec::new()};
	graph.add_node([0.0, 0.0], vec![true]);   
	graph.add_node([1.0, 0.0], vec![true]);   
	graph.add_edge(0, 1);

	graph
}

#[test]
fn test_graph_serialization() {
	let graph = create_minimal_graph();

	save(&graph, "results/test_graph_serialization.json");
	let _ = load("results/test_graph_serialization.json");
}

#[test]
fn test_dijkstra() {
	let graph = create_minimal_graph();

	struct Funcs {}
	impl PRMFuncs<2> for Funcs {}
	let dists = dijkstra(&graph, &vec![1], 0, &Funcs{});

	assert_eq!(dists, vec![1.0, 0.0]);
}
}