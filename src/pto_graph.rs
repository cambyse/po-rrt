use itertools::{all, enumerate, izip, merge, zip};

use crate::common::*;
use crate::nearest_neighbor::*;
use crate::sample_space::*;
use crate::map_io::*;
use std::{cmp::min, collections::btree_set::Difference, convert::TryInto, ops::Index, vec};

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
pub struct SerializablePTONode {
	pub state: Vec<f64>,
	pub validity_id: usize,
	pub parents: Vec<SerializablePTOEdge>,
	pub children: Vec<SerializablePTOEdge>
}

#[derive(Serialize, Deserialize)]
pub struct SerializablePTOEdge {
	pub id: usize,
	pub validity_id: usize,
}

impl SerializablePTOEdge {
	#[allow(clippy::nonminimal_bool)]
	pub fn from_pto_edge(edge : &PTOEdge) -> Self {
		Self{
			id: edge.id,			
			validity_id: edge.validity_id,
		}
	}

	pub fn to_pto_edge(&self) -> PTOEdge {
		PTOEdge {
			id: self.id,
			validity_id: self.validity_id,
		}
	}
}

impl SerializablePTONode {
	#[allow(clippy::nonminimal_bool)]
	pub fn from_pto_node(node : &PTONode<2>) -> Self {
		Self{
			state: node.state.to_vec(),			
			validity_id: node.validity_id,
			parents: node.parents.iter().map(|edge| SerializablePTOEdge::from_pto_edge(edge)).collect(),
			children: node.children.iter().map(|edge| SerializablePTOEdge::from_pto_edge(edge)).collect()
		}
	}

	pub fn to_pto_node(&self) -> PTONode<2> {
		PTONode {
			state: self.state.clone().try_into().unwrap(),
			validity_id: self.validity_id,
			parents: self.parents.iter().map(|edge| SerializablePTOEdge::to_pto_edge(edge)).collect(),
			children: self.children.iter().map(|edge| SerializablePTOEdge::to_pto_edge(edge)).collect(),
		}
	}
}


#[derive(Serialize, Deserialize)]
pub struct SerializablePTOGraph {
	pub nodes: Vec<SerializablePTONode>,
	pub validities: Vec<Vec<bool>>
}

impl SerializablePTOGraph {
	pub fn from_pto_graph_(pto_graph: &PTOGraph<2>) -> SerializablePTOGraph {
		let nodes = &pto_graph.nodes;
		SerializablePTOGraph {
			nodes: nodes.iter().map(|node| SerializablePTONode::from_pto_node(&node)).collect(),
			validities: pto_graph.validities.iter().map(|validity| SerializablePTOGraph::from_pto_validity(&validity) ).collect()
		}
	}

	pub fn save_(&self, filename: &str) {
		let writer = BufWriter::new(File::create(filename).expect("can't create file"));
		serde_json::to_writer_pretty(writer, &self).expect("error happened while dumping pto graph to file");
	}

	#[allow(clippy::nonminimal_bool)]
	pub fn from_pto_validity(validity: &WorldMask) -> Vec<bool> {
		validity.iter().map(|b| !!b).collect()
	}

	pub fn convert_to_pto_validity(validity: &[bool]) -> WorldMask {
		validity.iter().collect()
	}
}

pub fn save(pto_graph: &PTOGraph<2>, filename: &str) {
	let graph = SerializablePTOGraph::from_pto_graph_(pto_graph);
	graph.save_(filename);
}

pub fn load(filename: &str) -> PTOGraph<2> {
	let reader = BufReader::new(File::open(filename).expect("impossible to open file"));
	let graph: SerializablePTOGraph = serde_json::from_reader(reader).unwrap();

	PTOGraph {
		nodes: graph.nodes.into_iter().map(|node| node.to_pto_node()).collect(),
		validities: graph.validities.iter().map(|validity| SerializablePTOGraph::convert_to_pto_validity(validity)).collect()
	}
}

/****************************PTO Graph******************************/
pub trait PTOFuncs<const N: usize> {
	fn n_worlds(&self) -> usize {
		1
	}

	fn state_validity(&self, _state: &[f64; N]) -> Option<usize> {
		None
	}

	fn transition_validator(&self, from: &PTONode<N>, to: &PTONode<N>) -> Option<usize> {
		//let validity = from.validity.iter().zip(&to.validity)
		//.any(|(a, b)| *a && *b);

		let world_validities = self.world_validities();

		let from_validity = &world_validities[from.validity_id];
		let to_validity = &world_validities[to.validity_id];

		let validity: WorldMask = from_validity.iter().zip(to_validity)
		.map(|(a, b)| *a && *b)
		.collect();

		if world_validities.contains(&validity) {
			return Some(world_validities.iter().position(|v| *v == validity).expect("validity should be in the world validities"));
		}
		
		None
	}

	fn cost_evaluator(&self, a: &[f64; N], b: &[f64; N]) -> f64 {
		norm2(a,b)
	}

	#[allow(clippy::style)]
	fn reachable_belief_states(&self, belief_state: &BeliefState) -> Vec<BeliefState> {
		vec![belief_state.to_owned()]
	}

	#[allow(clippy::style)]
	fn world_validities(&self) -> Vec<WorldMask> {
		vec![]
	}

	#[allow(clippy::style)]
	fn observe(&self, _state: &[f64; N], belief_state: &BeliefState) -> Vec<BeliefState> {
		vec![belief_state.to_owned()]
	}
}

#[derive(Clone)]
pub struct PTONode<const N: usize> {
	pub state: [f64; N],
	pub validity_id: usize,
	pub parents: Vec<PTOEdge>,
	pub children: Vec<PTOEdge>,
}

#[derive(Clone)]
pub struct PTOEdge {
	pub id: usize,
	pub validity_id: usize
}

impl<const N: usize> GraphNode<N> for PTONode<N> {
	fn state(&self) -> &[f64; N] {
		&self.state
	}
}

#[derive(Clone)]
pub struct PTOGraph<const N: usize> {
	pub nodes: Vec<PTONode<N>>,
	pub validities: Vec<WorldMask>,
}

impl<const N: usize> PTOGraph<N> {
	pub fn add_node(&mut self, state: [f64; N], state_validity_id: usize) -> usize {
		let id = self.nodes.len();
		let node = PTONode { state, validity_id: state_validity_id, parents: Vec::new(), children: Vec::new() };
		self.nodes.push(node);
		id
	}

	pub fn add_edge(&mut self, from_id: usize, to_id: usize, validity_id: usize) {
		self.nodes[from_id].children.push(PTOEdge{id: to_id, validity_id});
		self.nodes[to_id].parents.push(PTOEdge{id:from_id, validity_id});
	}

	pub fn add_bi_edge(&mut self, id1: usize, id2: usize, validity_id: usize) {
		self.add_edge(id1, id2, validity_id);
		self.add_edge(id2, id1, validity_id);
	}

	pub fn remove_edge(&mut self, from_id: usize, to_id: usize) {
		self.nodes[from_id].children.retain(|edge|{edge.id != to_id});
		self.nodes[to_id].parents.retain(|edge|{edge.id != from_id});
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

impl<const N: usize> Graph<N> for PTOGraph<N> {
	fn node(&self, id:usize) -> &dyn GraphNode<N> {
		&self.nodes[id]
	}
	fn n_nodes(&self) -> usize {
		self.nodes.len()
	}
	fn children(&self, id: usize) -> Vec<usize> {
		self.nodes[id].children.iter().map(|edge| edge.id).collect()
	}
	fn parents(&self, id:usize) -> Vec<usize> {
		self.nodes[id].parents.iter().map(|edge| edge.id).collect()
	}
}

pub struct PTOGraphWorldView<'a, const N: usize> {
	pub graph: &'a PTOGraph<N>,
	pub world: usize
}

impl<'a, const N: usize> Graph<N> for PTOGraphWorldView<'a, N> {
	fn node(&self, id:usize) -> &dyn GraphNode<N> {
		self.graph.node(id)
	}
	fn n_nodes(&self) -> usize {
		self.graph.n_nodes()
	}
	fn children(&self, id: usize) -> Vec<usize> {
		//panic!("deprecated!");
		self.graph.nodes[id].children.iter()
			.map(|edge| edge.id)
			.filter(|&id| self.graph.validities[self.graph.nodes[id].validity_id][self.world])
			.collect()
	}
	fn parents(& self, id:usize) -> Vec<usize> {
		//panic!("deprecated!");
		self.graph.nodes[id].parents.iter()
			.map(|edge| edge.id)
			.filter(|&id| self.graph.validities[self.graph.nodes[id].validity_id][self.world])
			.collect()
	}
}

/****************************Dijkstra******************************/

pub fn dijkstra<F: PTOFuncs<N>, const N: usize>(graph: & impl Graph<N>, final_node_ids: &[usize], m: &F) -> Vec<f64> {
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

pub fn extract_path<F: PTOFuncs<N>, const N: usize>(graph: & impl Graph<N>, start_id: usize, expected_costs_to_goals: &[f64], m: &F) -> Vec<[f64; N]> {
    if graph.n_nodes() == 0 {
        panic!("no belief state graph!");
    }

	let mut path = Vec::new();

	let mut node_id = start_id;
	let mut node = graph.node(start_id);
	path.push(node.state().clone());

	while expected_costs_to_goals[node_id] != 0.0 {
		let (best_parent_id, _) = graph.parents(node_id).iter().map(|parent_id| (parent_id, graph.node(*parent_id).state())) //expected_costs_to_goals[*parent_id])
											 .map(|(&parent_id, &parent_state)| (parent_id, expected_costs_to_goals[parent_id] + m.cost_evaluator(&parent_state, node.state())))
											 .min_by(|(_, c_a), (__, c_b)| c_a.partial_cmp(c_b).unwrap()).unwrap();

		node_id = best_parent_id;
		node = graph.node(node_id);

		path.push(node.state().clone());
	}

	/*
    let mut policy: Policy<N> = Policy{nodes: vec![], leafs: vec![], expected_costs: 0.0};
    let mut lifo: Vec<(usize, usize)> = Vec::new(); // policy_node, belief_graph_node

    policy.add_node(&graph.nodes[0].state, &graph.nodes[0].belief_state, 0, false);

    lifo.push((0, 0));

    while !lifo.is_empty() {
        let (policy_node_id, belief_node_id) = lifo.pop().unwrap();

        let children_ids = get_best_expected_children(graph, belief_node_id, expected_costs_to_goals, cost_evaluator);

        //println!("---");

        for child_id in children_ids {
            let child = &graph.nodes[child_id];
            let is_leaf = expected_costs_to_goals[child_id] == 0.0;
            let child_policy_id = policy.add_node(&child.state, &graph.nodes[child_id].belief_state, child_id, is_leaf);
            policy.add_edge(policy_node_id, child_policy_id);

            //println!("add node, belief state {:?}, cost: {:?}", &graph.nodes[child_id].belief_state, &expected_costs_to_goals[child_id]);

            if ! is_leaf {
                lifo.push((child_policy_id, child_id));
            }
        }
    }
    policy.expected_costs = expected_costs_to_goals[0];
    policy*/

	path
}

/***********************Policy Extraction***********************/

pub fn get_policy_graph<const N: usize>(graph: &PTOGraph<N>, cost_to_goals: &[Vec<f64>]) -> Result<PTOGraph<N>, &'static str> {
	let mut policy = graph.clone();
	let n_worlds = cost_to_goals.len();

	let get_world_validities = |id: usize| -> Vec<bool> {
		(0..n_worlds).into_iter()
			.map(|world|{cost_to_goals[world][id].is_finite()})
			.collect()
	};

	for (from_id, node) in graph.nodes.iter().enumerate() {
		for to_edge in &node.children {
			let world_validities = get_world_validities(to_edge.id);

			// keep to if there is belief in which it would be the right decision: bs * cost(to) < bs * cost (other)
			// => this means solving an LP
			let mut problem = Problem::new(OptimizationDirection::Minimize);
			let mut belief: Vec<minilp::Variable> = Vec::new();
			
			// add variables, and force belief to 0 in infeasible worlds
			for &world_validity in world_validities.iter().take(n_worlds) {
				if world_validity {
					belief.push(problem.add_var(1.0, (0.0, 1.0)));
				}
				else {
					belief.push(problem.add_var(1.0, (0.0, 0.0)));
				}
			} 

			// normalization constraint
			let eq_constraint : Vec<(Variable, f64)> = belief.iter().map(|w|{(*w, 1.0)}).collect();
			problem.add_constraint(&eq_constraint, ComparisonOp::Eq, 1.0);

			// improvement constraint
			for other_edge in &node.children {
				if other_edge.id != to_edge.id {
					let mut improvment_constraint : Vec<(Variable, f64)> = Vec::new();

					for world in 0..n_worlds {
						if world_validities[world] && cost_to_goals[world][other_edge.id].is_finite() {
							improvment_constraint.push((belief[world], cost_to_goals[world][to_edge.id] - cost_to_goals[world][other_edge.id]));
						}
					}

					problem.add_constraint(&improvment_constraint, ComparisonOp::Le, 0.0);
				}
			}

			match problem.solve() {
				Ok(_) => {},
				Err(_) => policy.remove_edge(from_id, to_edge.id)
			}
		}
	}

	Ok(policy)
}

/****************************Tests******************************/

#[cfg(test)]
mod tests {

use super::*;

fn to_ids(children: &Vec<PTOEdge>) -> Vec<usize> {
	children.iter()
		.map(|edge| edge.id)
		.collect()
}

fn create_minimal_graph() -> PTOGraph<2> {
	/*
	0->-1
	*/
	let mut graph = PTOGraph{nodes: Vec::new(), validities: vec![bitvec![1]]};
	graph.add_node([0.0, 0.0], 0);   
	graph.add_node([1.0, 0.0], 0);   
	graph.add_edge(0, 1, 0);

	graph
}

fn create_grid_graph() -> PTOGraph<2> {
	/*
    6---7---8
    |   |   |
    3---4---5
    |   |   |
	0---1---2
	*/
	let mut graph = PTOGraph{nodes: Vec::new(), validities: vec![bitvec![1]]};
	// nodes
	graph.add_node([0.0, 0.0], 0);   // 0
	graph.add_node([1.0, 0.0], 0);   // 1
	graph.add_node([2.0, 0.0], 0);   // 2

	graph.add_node([0.0, 1.0], 0);   // 3
	graph.add_node([1.0, 1.0], 0);   // 4
	graph.add_node([2.0, 1.0], 0);   // 5

	graph.add_node([0.0, 2.0], 0);   // 6
	graph.add_node([1.0, 2.0], 0);   // 7
	graph.add_node([2.0, 2.0], 0);   // 8
 
	// edges
	graph.add_bi_edge(0, 1, 0);
	graph.add_bi_edge(1, 2, 0);

	graph.add_bi_edge(0, 3, 0);
	graph.add_bi_edge(1, 4, 0);
	graph.add_bi_edge(2, 5, 0);

	graph.add_bi_edge(3, 4, 0);
	graph.add_bi_edge(4, 5, 0);

	graph.add_bi_edge(3, 6, 0);
	graph.add_bi_edge(4, 7, 0);
	graph.add_bi_edge(5, 8, 0);

	graph.add_bi_edge(6, 7, 0);
	graph.add_bi_edge(7, 8, 0);
	graph
}

fn create_oriented_grid_graph() -> PTOGraph<2> {
	/*
    2-<-3
    ^   ^
	0->-1
	*/
	let mut graph = PTOGraph{nodes: Vec::new(), validities: vec![bitvec![1]]};

	// nodes
	graph.add_node([0.0, 0.0], 0);   // 0
	graph.add_node([1.0, 0.0], 0);   // 1

	graph.add_node([0.0, 1.0], 0);   // 2
	graph.add_node([1.0, 1.0], 0);   // 3

	// edges
	graph.add_edge(0, 1, 0);
	graph.add_edge(0, 2, 0);
	graph.add_edge(1, 3, 0);
	graph.add_edge(3, 2, 0);

	graph
}

fn create_diamond_graph() -> PTOGraph<2> {
	/*
	  1
	 / \
	0---3
	 \ /
	  2
	*/
	let mut graph = PTOGraph{nodes: Vec::new(), validities: vec![bitvec![1]]};

	// nodes
	graph.add_node([0.0, 0.0], 0);   // 0
	graph.add_node([1.0, 1.0], 0);   // 1

	graph.add_node([1.0, -1.0], 0);  // 2
	graph.add_node([2.0, 0.0], 0);   // 3

	// edges
	graph.add_bi_edge(0, 1, 0); 
	graph.add_bi_edge(0, 3, 0); 
	graph.add_bi_edge(0, 2, 0); 
	graph.add_bi_edge(1, 3, 0); 
	graph.add_bi_edge(2, 3, 0);

	graph
}

fn create_diamond_graph_2_worlds() -> PTOGraph<2> {
	/*
	  1
	 / \
	0   3
	 \ /
	  2
	*/
	// 1 valid in world 0, 2 valid in world 1
	let mut graph = PTOGraph{nodes: Vec::new(), validities: vec![bitvec![1, 0], bitvec![0, 1], bitvec![1, 1], ]};

	// nodes
	graph.add_node([0.0, 0.0], 2);   // 0
	graph.add_node([1.0, 1.0], 1);   // 1

	graph.add_node([1.0, -1.0], 0);  // 2
	graph.add_node([2.0, 0.0], 2);   // 3

	// edges
	graph.add_bi_edge(0, 1, 0);
	graph.add_bi_edge(0, 2, 1);
	graph.add_bi_edge(1, 3, 0);
	graph.add_bi_edge(2, 3, 1);

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
	impl PTOFuncs<2> for Funcs {}

	let dists = dijkstra(&graph, &vec![3], &Funcs{});

	let policy = get_policy_graph(&graph, &vec![dists]).unwrap();

	// pruning happend, only one edge should remain
	assert_eq!(to_ids(&policy.nodes[0].children), vec![3]);
	assert_eq!(to_ids(&policy.nodes[1].children), vec![3]);
	assert_eq!(to_ids(&policy.nodes[2].children), vec![3]);
}

#[test]
fn test_policy_extraction_on_diamond_graph_2_worlds() {
	let graph = create_diamond_graph_2_worlds();

	struct Funcs {}
	impl PTOFuncs<2> for Funcs {}
	let dists_w0 = dijkstra(&PTOGraphWorldView{graph: &graph, world: 0}, &vec![3], &Funcs{});
	let dists_w1 = dijkstra(&PTOGraphWorldView{graph: &graph, world: 1}, &vec![3], &Funcs{});

	let policy = get_policy_graph(&graph, &vec![dists_w0, dists_w1]).unwrap();

	// pruning of reversed edges happended, but 1 and 2 still reachable because we need them to be complete in each world
	assert_eq!(to_ids(&policy.nodes[3].children), vec![1, 2]);
	assert_eq!(to_ids(&policy.nodes[1].children), vec![3]);
	assert_eq!(to_ids(&policy.nodes[2].children), vec![3]);
}

#[test]
fn test_policy_extraction_on_grid_with_2_different_goals() {
	let graph = create_grid_graph();

	struct Funcs {}
	impl PTOFuncs<2> for Funcs {}
	let dists_w0 = dijkstra(&PTOGraphWorldView{graph: &graph, world: 0}, &vec![2], &Funcs{});
	let dists_w1 = dijkstra(&PTOGraphWorldView{graph: &graph, world: 0}, &vec![5], &Funcs{});

	let policy = get_policy_graph(&graph, &vec![dists_w0, dists_w1]).unwrap();

	// prune non advantageous part of the grid
	assert_eq!(to_ids(&policy.nodes[3].children), vec![0, 4]);
	assert_eq!(to_ids(&policy.nodes[4].children), vec![1, 5]);
	assert_eq!(to_ids(&policy.nodes[8].children), vec![5]);
}

#[test]
fn test_dijkstra_on_minimal_graph() {
	let graph = create_minimal_graph();

	struct Funcs {}
	impl PTOFuncs<2> for Funcs {}
	let dists = dijkstra(&graph, &vec![1], &Funcs{});

	assert_eq!(dists, vec![1.0, 0.0]);
}

#[test]
fn test_dijkstra_on_grid_graph_single_goal() {
	let graph = create_grid_graph();

	struct Funcs {}
	impl PTOFuncs<2> for Funcs {}
	let dists = dijkstra(&graph, &vec![8], &Funcs{});

	assert_eq!(dists, vec![4.0, 3.0, 2.0, 3.0, 2.0, 1.0, 2.0, 1.0, 0.0]);
}

#[test]
fn test_dijkstra_on_grid_graph_two_goals() {
	let graph = create_grid_graph();

	struct Funcs {}
	impl PTOFuncs<2> for Funcs {}
	let dists = dijkstra(&graph, &vec![7, 5], &Funcs{});

	assert_eq!(dists, vec![3.0, 2.0, 1.0, 2.0, 1.0, 0.0, 1.0, 0.0, 1.0]);
}

#[test]
fn test_dijkstra_without_final_node() {
	let graph = create_grid_graph();

	struct Funcs {}
	impl PTOFuncs<2> for Funcs {}
	let dists = dijkstra(&graph, &vec![], &Funcs{});

	assert_eq!(dists, vec![std::f64::INFINITY; 9]);
}

#[test]
fn test_dijkstra_on_oriented_grid() {
	let graph = create_oriented_grid_graph();

	struct Funcs {}
	impl PTOFuncs<2> for Funcs {}
	let dists = dijkstra(&graph, &vec![3], &Funcs{});

	assert_eq!(dists, vec![2.0, 1.0, std::f64::INFINITY, 0.0]);
}

#[test]
fn test_world_transitions() {
	struct Funcs {
		world_validities: Vec<WorldMask>
	}
	impl PTOFuncs<2> for Funcs {
		fn world_validities(&self) -> Vec<WorldMask> {
			return self.world_validities.clone();
		}
	}

	let f = Funcs{world_validities: vec![bitvec![1, 0], bitvec![0, 1], bitvec![1, 1]]};

	fn dummy_node(validity_id : usize) -> PTONode<2>	{
		PTONode{state: [0.0, 0.0], validity_id, parents: Vec::new(), children: Vec::new()}
	}

	assert_eq!(f.transition_validator(&dummy_node(0), &dummy_node(0)), Some(0));
	assert_eq!(f.transition_validator(&dummy_node(0), &dummy_node(1)), None);
	assert_eq!(f.transition_validator(&dummy_node(2), &dummy_node(2)), Some(2));
}

}
