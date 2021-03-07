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
use bitvec::prelude::*;
use minilp::{ComparisonOp, OptimizationDirection, Problem, Variable};

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
			validity: node.validity.iter().map(|b| !!b).collect(),
			parents: node.parents.clone(),
			children: node.children.clone()
		}
	}

	pub fn to_prm_node(&self) -> PRMNode<2> {
		PRMNode {
			state: self.state.clone().try_into().unwrap(),
			validity: self.validity.iter().collect(),
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
	fn state_validity(&self, _state: &[f64; N]) -> Option<WorldMask> {
		None
	}

	fn transition_validator(&self, from: &PRMNode<N>, to: &PRMNode<N>) -> bool {
		from.validity.iter().zip(&to.validity)
		.any(|(a, b)| *a && *b)
	}

	fn cost_evaluator(&self, a: &[f64; N], b: &[f64; N]) -> f64 {
		norm2(a,b)
	}

	fn reachable_belief_states(&self, belief_state: &Vec<f64>) -> Vec<Vec<f64>> {
		vec![belief_state.clone()]
	}

	fn observe(&self, _state: &[f64; N], belief_state: &Vec<f64>) -> Vec<Vec<f64>> {
		vec![belief_state.clone()]
	}
}

#[derive(Clone)]
pub struct PRMNode<const N: usize> {
	pub state: [f64; N],
	pub validity: WorldMask,
	pub parents: Vec<usize>,
	pub children: Vec<usize>,
}

impl<const N: usize> GraphNode<N> for PRMNode<N> {
	fn state(&self) -> &[f64; N] {
		&self.state
	}
}

#[derive(Clone)]
pub struct PRMGraph<const N: usize> {
	pub nodes: Vec<PRMNode<N>>,
}

impl<const N: usize> PRMGraph<N> {
	pub fn add_node(&mut self, state: [f64; N], state_validity: WorldMask) -> usize {
		let id = self.nodes.len();
		let node = PRMNode { state, validity: state_validity, parents: Vec::new(), children: Vec::new() };
		self.nodes.push(node);
		id
	}

	pub fn add_edge(&mut self, from_id: usize, to_id: usize) {
		self.nodes[from_id].children.push(to_id);
		self.nodes[to_id].parents.push(from_id);
	}

	pub fn remove_edge(&mut self, from_id: usize, to_id: usize) {
		self.nodes[from_id].children.retain(|&id|{id != to_id});
		self.nodes[to_id].parents.retain(|&id|{id != from_id});
	}

	pub fn get_path_to(&self, _: usize) -> Vec<[f64; N]> {
		let path = Vec::new();

		path
	}

	pub fn print_summary(&self) {
		let (sum, max) = self.nodes.iter()
			.map(|node| node.children.len())
			.fold((0, 0), |(sum, max), n_children| (sum + n_children, if n_children > max { n_children } else { max }));

		println!("number of nodes:{}", self.nodes.len());
		println!("average number of children:{}", sum / self.nodes.len());
		println!("max number of children:{}", max)
	}
}

impl<const N: usize> Graph<N> for PRMGraph<N> {
	fn node(&self, id:usize) -> &dyn GraphNode<N> {
		&self.nodes[id]
	}
	fn n_nodes(&self) -> usize {
		self.nodes.len()
	}
	fn children(&self, id: usize) -> Vec<usize> {
		self.nodes[id].children.clone()
	}
	fn parents(&self, id:usize) -> Vec<usize> {
		self.nodes[id].parents.clone()
	}
}

pub struct PRMGraphWorldView<'a, const N: usize> {
	pub graph: &'a PRMGraph<N>,
	pub world: usize
}

impl<'a, const N: usize> Graph<N> for PRMGraphWorldView<'a, N> {
	fn node(&self, id:usize) -> &dyn GraphNode<N> {
		self.graph.node(id)
	}
	fn n_nodes(&self) -> usize {
		self.graph.n_nodes()
	}
	fn children(&self, id: usize) -> Vec<usize> {
		self.graph.nodes[id].children.iter().filter(|&id| self.graph.nodes[*id].validity[self.world]).map(|&id| id).collect()
	}
	fn parents(&self, id:usize) -> Vec<usize> {
		self.graph.nodes[id].parents.iter().filter(|&id| self.graph.nodes[*id].validity[self.world]).map(|&id| id).collect()
	}
}

/****************************Dijkstra******************************/

pub fn dijkstra<'a, F: PRMFuncs<N>, const N: usize>(graph: & impl Graph<N>, final_node_ids: &Vec<usize>, m: &F) -> Vec<f64> {
	// https://fr.wikipedia.org/wiki/Algorithme_de_Dijkstra
	// complexité n log n ;graph.nodes.len()
	let mut dist = vec![std::f64::INFINITY; graph.n_nodes()];
	let mut q = PriorityQueue::new();
	
	for &id in final_node_ids {
		dist[id] = 0.0;
		q.push(id, Priority{prio: 0.0});
	}

	while !q.is_empty() {
		let (v_id, _) = q.pop().unwrap();
		let v = &graph.node(v_id);
		
		for u_id in graph.parents(v_id) {
			let u = &graph.node(u_id);

			let alternative = dist[v_id] + m.cost_evaluator(u.state(), v.state());

			if alternative < dist[u_id] {
				dist[u_id] = alternative;
				q.push(u_id, Priority{prio: alternative});
			}
		}
	}

	dist
}

/***********************Policy Extraction***********************/

pub fn get_policy_graph<const N: usize>(graph: &PRMGraph<N>, cost_to_goals: &Vec<Vec<f64>>) -> Result<PRMGraph<N>, &'static str> {
	let mut policy = graph.clone();
	let n_worlds = cost_to_goals.len();

	let get_world_validities = |id: usize| -> Vec<bool> {
		(0..n_worlds).into_iter()
			.map(|world|{cost_to_goals[world][id].is_finite()})
			.collect()
	};

	for (from_id, node) in graph.nodes.iter().enumerate() {
		for &to_id in &node.children {
			let world_validities = get_world_validities(to_id);

			// keep to if there is belief in which it would be the right decision: bs * cost(to) < bs * cost (other)
			// => this means solving an LP
			let mut problem = Problem::new(OptimizationDirection::Minimize);
			let mut belief: Vec<minilp::Variable> = Vec::new();
			
			// add variables, and force belief to 0 in infeasible worlds
			for world in 0..n_worlds {
				if world_validities[world] {
					belief.push(problem.add_var(1.0, (0.0, 1.0)));
				}
				else {
					belief.push(problem.add_var(1.0, (0.0, 0.0)));
				}
			} 

			// normalization constraint
			let eq_constraint : Vec<(Variable, f64)> = belief.iter().map(|w|{(*w, 1.0)}).collect();
			problem.add_constraint(&eq_constraint, ComparisonOp::Eq, 1.0);

			// improvment constraint
			for &other_id in &node.children {
				if other_id != to_id {
					let mut improvment_constraint : Vec<(Variable, f64)> = Vec::new();

					for world in 0..n_worlds {
						if world_validities[world] && cost_to_goals[world][other_id].is_finite() {
							improvment_constraint.push((belief[world], cost_to_goals[world][to_id] - cost_to_goals[world][other_id]));
						}
					}

					problem.add_constraint(&improvment_constraint, ComparisonOp::Le, 0.0);
				}
			}

			match problem.solve() {
				Ok(_) => {},
				Err(_) => policy.remove_edge(from_id, to_id)
			}
		}
	}

	Ok(policy)
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
	graph.add_node([0.0, 0.0], bitvec![1]);   
	graph.add_node([1.0, 0.0], bitvec![1]);   
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
	graph.add_node([0.0, 0.0], bitvec![1]);   // 0
	graph.add_node([1.0, 0.0], bitvec![1]);   // 1
	graph.add_node([2.0, 0.0], bitvec![1]);   // 2

	graph.add_node([0.0, 1.0], bitvec![1]);   // 3
	graph.add_node([1.0, 1.0], bitvec![1]);   // 4
	graph.add_node([2.0, 1.0], bitvec![1]);   // 5

	graph.add_node([0.0, 2.0], bitvec![1]);   // 6
	graph.add_node([1.0, 2.0], bitvec![1]);   // 7
	graph.add_node([2.0, 2.0], bitvec![1]);   // 8
 
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
	graph.add_node([0.0, 0.0], bitvec![1]);   // 0
	graph.add_node([1.0, 0.0], bitvec![1]);   // 1

	graph.add_node([0.0, 1.0], bitvec![1]);   // 2
	graph.add_node([1.0, 1.0], bitvec![1]);   // 3

	// edges
	graph.add_edge(0, 1);
	graph.add_edge(0, 2);
	graph.add_edge(1, 3);
	graph.add_edge(3, 2);

	graph
}

fn create_diamond_graph() -> PRMGraph<2> {
	/*
	  1
	 / \
	0---3
	 \ /
	  2
	*/
	let mut graph = PRMGraph{nodes: Vec::new()};

	// nodes
	graph.add_node([0.0, 0.0], bitvec![1]);   // 0
	graph.add_node([1.0, 1.0], bitvec![1]);   // 1

	graph.add_node([1.0, -1.0], bitvec![1]);  // 2
	graph.add_node([2.0, 0.0], bitvec![1]);   // 3

	// edges
	graph.add_edge(0, 1); 	graph.add_edge(1, 0);
	graph.add_edge(0, 3); 	graph.add_edge(3, 0);
	graph.add_edge(0, 2); 	graph.add_edge(2, 0);
	graph.add_edge(1, 3); 	graph.add_edge(3, 1);
	graph.add_edge(2, 3); 	graph.add_edge(3, 2);

	graph
}

fn create_diamond_graph_2_worlds() -> PRMGraph<2> {
	/*
	  1
	 / \
	0   3
	 \ /
	  2
	*/
	// 1 valid in world 0, 2 valid in world 1
	let mut graph = PRMGraph{nodes: Vec::new()};

	// nodes
	graph.add_node([0.0, 0.0], bitvec![1, 1]);   // 0
	graph.add_node([1.0, 1.0], bitvec![1, 0]);   // 1

	graph.add_node([1.0, -1.0], bitvec![0, 1]);  // 2
	graph.add_node([2.0, 0.0], bitvec![1, 1]);   // 3

	// edges
	graph.add_edge(0, 1); 	graph.add_edge(1, 0);
	graph.add_edge(0, 2); 	graph.add_edge(2, 0);
	graph.add_edge(1, 3); 	graph.add_edge(3, 1);
	graph.add_edge(2, 3); 	graph.add_edge(3, 2);

	graph
}

#[test]
fn test_graph_serialization() {
	let graph = create_minimal_graph();

	save(&graph, "results/test_graph_serialization.json");
	let _ = load("results/test_graph_serialization.json");
}

#[test]
fn test_policy_extraction_on_diamond_graph() {
	let graph = create_diamond_graph();

	struct Funcs {}
	impl PRMFuncs<2> for Funcs {}
	let dists = dijkstra(&graph, &vec![3], &Funcs{});

	let policy = get_policy_graph(&graph, &vec![dists]).unwrap();

	// pruning happend, only one edge should remain
	assert_eq!(policy.nodes[0].children, vec![3]);
	assert_eq!(policy.nodes[1].children, vec![3]);
	assert_eq!(policy.nodes[2].children, vec![3]);
}

#[test]
fn test_policy_extraction_on_diamond_graph_2_worlds() {
	let graph = create_diamond_graph_2_worlds();

	struct Funcs {}
	impl PRMFuncs<2> for Funcs {}
	let dists_w0 = dijkstra(&PRMGraphWorldView{graph: &graph, world: 0}, &vec![3], &Funcs{});
	let dists_w1 = dijkstra(&PRMGraphWorldView{graph: &graph, world: 1}, &vec![3], &Funcs{});

	let policy = get_policy_graph(&graph, &vec![dists_w0, dists_w1]).unwrap();

	// pruning of reversed edges happended, but 1 and 2 still reachable because we need them to be complete in each world
	assert_eq!(policy.nodes[3].children, vec![1, 2]);
	assert_eq!(policy.nodes[1].children, vec![3]);
	assert_eq!(policy.nodes[2].children, vec![3]);
}

#[test]
fn test_policy_extraction_on_grid_with_2_different_goals() {
	let graph = create_grid_graph();

	struct Funcs {}
	impl PRMFuncs<2> for Funcs {}
	let dists_w0 = dijkstra(&PRMGraphWorldView{graph: &graph, world: 0}, &vec![2], &Funcs{});
	let dists_w1 = dijkstra(&PRMGraphWorldView{graph: &graph, world: 0}, &vec![5], &Funcs{});

	let policy = get_policy_graph(&graph, &vec![dists_w0, dists_w1]).unwrap();

	// prune non advantageous part of the grid
	assert_eq!(policy.nodes[3].children, vec![0, 4]);
	assert_eq!(policy.nodes[4].children, vec![1, 5]);
	assert_eq!(policy.nodes[8].children, vec![5]);
}

#[test]
fn test_dijkstra_on_minimal_graph() {
	let graph = create_minimal_graph();

	struct Funcs {}
	impl PRMFuncs<2> for Funcs {}
	let dists = dijkstra(&graph, &vec![1], &Funcs{});

	assert_eq!(dists, vec![1.0, 0.0]);
}

#[test]
fn test_dijkstra_on_grid_graph_single_goal() {
	let graph = create_grid_graph();

	struct Funcs {}
	impl PRMFuncs<2> for Funcs {}
	let dists = dijkstra(&graph, &vec![8], &Funcs{});

	assert_eq!(dists, vec![4.0, 3.0, 2.0, 3.0, 2.0, 1.0, 2.0, 1.0, 0.0]);
}

#[test]
fn test_dijkstra_on_grid_graph_two_goals() {
	let graph = create_grid_graph();

	struct Funcs {}
	impl PRMFuncs<2> for Funcs {}
	let dists = dijkstra(&graph, &vec![7, 5], &Funcs{});

	assert_eq!(dists, vec![3.0, 2.0, 1.0, 2.0, 1.0, 0.0, 1.0, 0.0, 1.0]);
}

#[test]
fn test_dijkstra_without_final_node() {
	let graph = create_grid_graph();

	struct Funcs {}
	impl PRMFuncs<2> for Funcs {}
	let dists = dijkstra(&graph, &vec![], &Funcs{});

	assert_eq!(dists, vec![std::f64::INFINITY; 9]);
}

#[test]
fn test_dijkstra_on_oriented_grid() {
	let graph = create_oriented_grid_graph();

	struct Funcs {}
	impl PRMFuncs<2> for Funcs {}
	let dists = dijkstra(&graph, &vec![3], &Funcs{});

	assert_eq!(dists, vec![2.0, 1.0, std::f64::INFINITY, 0.0]);
}

#[test]
fn test_world_transitions() {
	struct Funcs {}
	impl PRMFuncs<2> for Funcs {}

	let f = Funcs{};

	fn dummy_node(validity : WorldMask) -> PRMNode<2>	{
		PRMNode{state: [0.0, 0.0], validity: validity, parents: Vec::new(), children: Vec::new()}
	}

	assert_eq!(f.transition_validator(&dummy_node(bitvec![1]), &dummy_node(bitvec![1])), true);
	assert_eq!(f.transition_validator(&dummy_node(bitvec![1]), &dummy_node(bitvec![0])), false);
	assert_eq!(f.transition_validator(&dummy_node(bitvec![1, 0]), &dummy_node(bitvec![1, 0])), true);
	assert_eq!(f.transition_validator(&dummy_node(bitvec![1, 0]), &dummy_node(bitvec![0, 1])), false);
	assert_eq!(f.transition_validator(&dummy_node(bitvec![1, 1]), &dummy_node(bitvec![1, 1])), true);
	assert_eq!(f.transition_validator(&dummy_node(bitvec![0, 0]), &dummy_node(bitvec![1, 1])), false);
}

}
