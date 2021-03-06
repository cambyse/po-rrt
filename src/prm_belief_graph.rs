use itertools::{all, enumerate, izip, merge, zip};

use crate::common::*;
use crate::nearest_neighbor::*;
use crate::sample_space::*;
use crate::map_io::*; // tests only
use crate::prm_graph::*;
use crate::prm_reachability::*;
use bitvec::prelude::*;
use priority_queue::PriorityQueue;

pub struct PRMBeliefNode<const N: usize> {
	pub state: [f64; N],
	pub belief_state: BeliefState,
	pub parents: Vec<usize>,
    pub children: Vec<usize>,
}


impl<const N: usize> GraphNode<N> for PRMBeliefNode<N> {
	fn state(&self) -> &[f64; N] {
		&self.state
	}
}

pub struct PRMBeliefGraph<const N: usize> {
	pub belief_nodes: Vec<PRMBeliefNode<N>>,
}

impl<const N: usize> PRMBeliefGraph<N> {
	pub fn add_node(&mut self, state: [f64; N], belief_state: BeliefState) -> usize {
        let id = self.belief_nodes.len();
        self.belief_nodes.push(
            PRMBeliefNode{
                state,
                belief_state,
                parents: Vec::new(),
                children: Vec::new(),
            }
        );
        id
    }

	pub fn add_edge(&mut self, from_id: usize, to_id: usize) {
		self.belief_nodes[from_id].children.push(to_id);
		self.belief_nodes[to_id].parents.push(from_id);
	}
}

impl<const N: usize> Graph<N> for PRMBeliefGraph<N> {
	fn node(&self, id:usize) -> &dyn GraphNode<N> {
		&self.belief_nodes[id]
	}
	fn n_nodes(&self) -> usize {
		self.belief_nodes.len()
	}
	fn children(&self, id: usize) -> Vec<usize> {
		self.belief_nodes[id].children.clone()
	}
	fn parents(&self, id:usize) -> Vec<usize> {
		self.belief_nodes[id].parents.clone()
	}
}

impl<const N: usize> ObservationGraph for PRMBeliefGraph<N> {
    fn siblings(&self, parent_id: usize, id: usize) -> Vec<(usize, f64)> {

        fn transition_probability(parent_bs: &BeliefState, child_bs: &BeliefState) -> f64 {
            child_bs.iter().zip(parent_bs).fold(0.0, |s, (p, q)| s + p * q )
        }

        let mut siblings: Vec<(usize, f64)> = Vec::new();

        let parent = &self.belief_nodes[parent_id];
        let witness = &self.belief_nodes[id];
        for &child_id in &parent.children {
            let child = &self.belief_nodes[child_id];

            if witness.state == child.state {
                siblings.push((child_id, transition_probability(&parent.belief_state, &child.belief_state)))
            }
        }

        siblings
    }
}

pub fn conditional_dijkstra<'a, F: PRMFuncs<N>, const N: usize>(graph: & (impl Graph<N> + ObservationGraph), final_node_ids: &Vec<usize>, m: &F) -> Vec<f64> {
	// https://fr.wikipedia.org/wiki/Algorithme_de_Dijkstra
	// complexité n log n ;graph.nodes.len()
	let mut dist = vec![std::f64::INFINITY; graph.n_nodes()];
	let mut prev = vec![0; graph.n_nodes()];
	let mut q = PriorityQueue::new();
	
	for &id in final_node_ids {
		dist[id] = 0.0;
		q.push(id, Priority{prio: 0.0});
	}

	while !q.is_empty() {
		let (v_id, _) = q.pop().unwrap();
		
		for u_id in graph.parents(v_id) {
            let u = &graph.node(u_id);

            let mut alternative = dist[v_id];
            for (vv_id, p) in graph.siblings(u_id, v_id) {
                let vv = graph.node(vv_id);
                alternative += p * m.cost_evaluator(u.state(), vv.state());
            }
            
			if alternative < dist[u_id] {
				dist[u_id] = alternative;
				prev[u_id] = u_id;
				q.push(u_id, Priority{prio: alternative});
			}
		}
	}

	dist
}

#[cfg(test)]
mod tests {

use super::*;

fn create_graph_1() -> PRMBeliefGraph<2> {
    /*
     G
    / \
  (E) (F)
   |   |
   C   D
    \ /
     B
     |
     A

     E and F are conditionally valid (2 different worlds, need to go to A to observe, when starting from B)
    */
    
    /*    
     3
    / \
  ( ) ( )
   |   |
   1   2
    \ /
     0
     |
    ( )

    bs: [0.5, 0.5]
    */

    /*    
     9
    / \
   8  ( )
   |   |
   6   7
    \ /
     5
     |
     4

    bs: [1.0, 0.0]
    */

    /*    
    15
    / \
  ( )  14
   |   |
   12  13
    \ /
     11
     |
     10

    bs: [0.0, 1.0]
    */
    let mut belief_graph = PRMBeliefGraph{belief_nodes: Vec::new()};

    // nodes
    belief_graph.add_node([0.0, 1.0], vec![0.5, 0.5]); // 0
    belief_graph.add_node([-1.0, 2.0], vec![0.5, 0.5]); // 1
    belief_graph.add_node([1.0, 2.0], vec![0.5, 0.5]); // 2
    belief_graph.add_node([0.0, 4.0], vec![0.5, 0.5]); // 3

    belief_graph.add_node([0.0, 0.0], vec![1.0, 0.0]); // 4
    belief_graph.add_node([0.0, 1.0], vec![1.0, 0.0]); // 5
    belief_graph.add_node([-1.0, 2.0], vec![1.0, 0.0]);// 6
    belief_graph.add_node([1.0, 2.0], vec![1.0, 0.0]);// 7
    belief_graph.add_node([-1.0, 3.0], vec![1.0, 0.0]);// 8
    belief_graph.add_node([0.0, 4.0], vec![1.0, 0.0]);// 9


    belief_graph.add_node([0.0, 0.0], vec![0.0, 1.0]); // 10
    belief_graph.add_node([0.0, 1.0], vec![0.0, 1.0]); // 11
    belief_graph.add_node([-1.0, 2.0], vec![0.0, 1.0]); // 12
    belief_graph.add_node([1.0, 2.0], vec![0.0, 1.0]); // 13
    belief_graph.add_node([1.0, 3.0], vec![0.0, 1.0]); // 14
    belief_graph.add_node([0.0, 4.0], vec![0.0, 1.0]); // 15


    // edges
    belief_graph.add_edge(0, 1); belief_graph.add_edge(1, 0);
    belief_graph.add_edge(0, 2); belief_graph.add_edge(2, 0);

    belief_graph.add_edge(4, 5); belief_graph.add_edge(5, 4);
    belief_graph.add_edge(5, 6); belief_graph.add_edge(6, 5);
    belief_graph.add_edge(5, 7); belief_graph.add_edge(7, 5);
    belief_graph.add_edge(6, 8); belief_graph.add_edge(8, 6);
    belief_graph.add_edge(8, 9); belief_graph.add_edge(9, 8);

    belief_graph.add_edge(10, 11); belief_graph.add_edge(11, 10);
    belief_graph.add_edge(11, 12); belief_graph.add_edge(12, 11);
    belief_graph.add_edge(11, 13); belief_graph.add_edge(13, 11);
    belief_graph.add_edge(13, 14); belief_graph.add_edge(14, 13);
    belief_graph.add_edge(14, 15); belief_graph.add_edge(15, 14);

    belief_graph.add_edge(0, 4);
    belief_graph.add_edge(0, 10);

    belief_graph
}

#[test]
fn test_sibling_extraction() {
    let graph = create_graph_1();
    
    let siblings = graph.siblings(0, 4);

    assert_eq!(siblings, vec![(4, 0.6), (10, 0.4)]);
}

#[test]
fn test_conditional_dijkstra() {
    let graph = create_graph_1();
    
    struct Funcs {}
    impl PRMFuncs<2> for Funcs {}
    
    let dists = conditional_dijkstra(&graph, &vec![3, 9, 15], &Funcs{});
    assert!(dists[0] < dists[1]);
    assert!(dists[0] < dists[2]);

    assert!(dists[4] < dists[0]);
    assert!(dists[5] < dists[4]);
    assert!(dists[6] < dists[5]);
    assert!(dists[5] < dists[7]);
    assert!(dists[8] < dists[6]);
    assert!(dists[9] < dists[8]);

    assert!(dists[10] < dists[0]);
    assert!(dists[11] < dists[10]);
    assert!(dists[11] < dists[12]);
    assert!(dists[13] < dists[11]);
    assert!(dists[14] < dists[13]);
    assert!(dists[15] < dists[14]);
}
}