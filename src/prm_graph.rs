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
	/*
	0->-1
	*/
	let mut graph = PRMGraph{nodes: Vec::new()};
	graph.add_node([0.0, 0.0], vec![true]);   
	graph.add_node([1.0, 0.0], vec![true]);   
	graph.add_edge(0, 1);

	graph
}

fn create_grid_graph() -> PRMGraph<2> {
	/*
    6---7---8
    |   |   |
    3---4---5
    |   |   |
	0---1---2
	*/
	let mut graph = PRMGraph{nodes: Vec::new()};
	// nodes
	graph.add_node([0.0, 0.0], vec![true]);   // 0
	graph.add_node([1.0, 0.0], vec![true]);   // 1
	graph.add_node([2.0, 0.0], vec![true]);   // 2

	graph.add_node([0.0, 1.0], vec![true]);   // 3
	graph.add_node([1.0, 1.0], vec![true]);   // 4
	graph.add_node([2.0, 1.0], vec![true]);   // 5
	
	graph.add_node([0.0, 2.0], vec![true]);   // 6
	graph.add_node([1.0, 2.0], vec![true]);   // 7
	graph.add_node([2.0, 2.0], vec![true]);   // 8
 
	// edges
	graph.add_edge(0, 1); 	graph.add_edge(1, 0);
	graph.add_edge(1, 2);   graph.add_edge(2, 1);

	graph.add_edge(0, 3); 	graph.add_edge(3, 0);
	graph.add_edge(1, 4); 	graph.add_edge(4, 1);
	graph.add_edge(2, 5); 	graph.add_edge(5, 2);

	graph.add_edge(3, 4); 	graph.add_edge(4, 3);
	graph.add_edge(4, 5);   graph.add_edge(5, 4);

	graph.add_edge(3, 6); 	graph.add_edge(6, 3);
	graph.add_edge(4, 7); 	graph.add_edge(7, 4);
	graph.add_edge(5, 8); 	graph.add_edge(8, 5);

	graph.add_edge(6, 7); 	graph.add_edge(7, 6);
	graph.add_edge(7, 8);   graph.add_edge(8, 7);

	graph
}

fn create_oriented_grid_graph() -> PRMGraph<2> {
	/*
    2-<-3
    ^   ^
	0->-1
	*/
	let mut graph = PRMGraph{nodes: Vec::new()};

	// nodes
	graph.add_node([0.0, 0.0], vec![true]);   // 0
	graph.add_node([1.0, 0.0], vec![true]);   // 1

	graph.add_node([0.0, 1.0], vec![true]);   // 2
	graph.add_node([1.0, 1.0], vec![true]);   // 3

	// edges
	graph.add_edge(0, 1);
	graph.add_edge(0, 2);
	graph.add_edge(1, 3);
	graph.add_edge(3, 2);

	graph
}

#[test]
fn test_graph_serialization() {
	let graph = create_minimal_graph();

	save(&graph, "results/test_graph_serialization.json");
	let _ = load("results/test_graph_serialization.json");
}

#[test]
fn test_dijkstra_on_minimal_graph() {
	let graph = create_minimal_graph();

	struct Funcs {}
	impl PRMFuncs<2> for Funcs {}
	let dists = dijkstra(&graph, &vec![1], 0, &Funcs{});

	assert_eq!(dists, vec![1.0, 0.0]);
}

#[test]
fn test_dijkstra_on_grid_graph_single_goal() {
	let graph = create_grid_graph();

	struct Funcs {}
	impl PRMFuncs<2> for Funcs {}
	let dists = dijkstra(&graph, &vec![8], 0, &Funcs{});

	assert_eq!(dists, vec![4.0, 3.0, 2.0, 3.0, 2.0, 1.0, 2.0, 1.0, 0.0]);
}

#[test]
fn test_dijkstra_on_grid_graph_two_goals() {
	let graph = create_grid_graph();

	struct Funcs {}
	impl PRMFuncs<2> for Funcs {}
	let dists = dijkstra(&graph, &vec![7, 5], 0, &Funcs{});

	assert_eq!(dists, vec![3.0, 2.0, 1.0, 2.0, 1.0, 0.0, 1.0, 0.0, 1.0]);
}

#[test]
fn test_dijkstra_on_oriented_grid() {
	let graph = create_oriented_grid_graph();

	struct Funcs {}
	impl PRMFuncs<2> for Funcs {}
	let dists = dijkstra(&graph, &vec![3], 0, &Funcs{});

	assert_eq!(dists, vec![2.0, 1.0, std::f64::INFINITY, 0.0]);
}

#[test]
fn test_world_transitions() {
	struct Funcs {}
	impl PRMFuncs<2> for Funcs {}

	let f = Funcs{};

	fn dummy_node(validity : Vec<bool>) -> PRMNode<2>	{
		PRMNode{state: [0.0, 0.0], validity: validity, parents: Vec::new(), children: Vec::new()}
	}

	assert_eq!(f.transition_validator(&dummy_node(vec![true]), &dummy_node(vec![true])), true);
	assert_eq!(f.transition_validator(&dummy_node(vec![true]), &dummy_node(vec![false])), false);
	assert_eq!(f.transition_validator(&dummy_node(vec![true, false]), &dummy_node(vec![true, false])), true);
	assert_eq!(f.transition_validator(&dummy_node(vec![true, false]), &dummy_node(vec![false, true])), false);
	assert_eq!(f.transition_validator(&dummy_node(vec![true, true]), &dummy_node(vec![true, true])), true);
	assert_eq!(f.transition_validator(&dummy_node(vec![false, false]), &dummy_node(vec![true, true])), false);
}

}