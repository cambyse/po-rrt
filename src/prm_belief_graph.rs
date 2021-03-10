use itertools::{all, enumerate, izip, merge, zip};

use crate::common::*;
use crate::nearest_neighbor::*;
use crate::sample_space::*;
use crate::map_io::*; // tests only
use crate::prm_graph::*;
use crate::prm_reachability::*;
use bitvec::prelude::*;
use priority_queue::PriorityQueue;

#[derive(PartialEq)]
pub enum BeliefNodeType {
    Unknown,
    Action,
    Observation,
}

pub struct PRMBeliefNode<const N: usize> {
	pub state: [f64; N],
    pub belief_state: BeliefState,
    pub belief_id: usize,
	pub parents: Vec<usize>,
    pub children: Vec<usize>,
    pub node_type: BeliefNodeType,
}


impl<const N: usize> GraphNode<N> for PRMBeliefNode<N> {
	fn state(&self) -> &[f64; N] {
		&self.state
	}
}

pub struct PRMBeliefGraph<const N: usize> {
    pub belief_nodes: Vec<PRMBeliefNode<N>>,
    pub reachable_belief_states: Vec<Vec<f64>>
}

impl<const N: usize> PRMBeliefGraph<N> {
	pub fn add_node(&mut self, state: [f64; N], belief_state: BeliefState, belief_id: usize) -> usize {
        let id = self.belief_nodes.len();
        self.belief_nodes.push(
            PRMBeliefNode{
                state,
                belief_state,
                belief_id,
                parents: Vec::new(),
                children: Vec::new(),
                node_type: BeliefNodeType::Unknown,
            }
        );
        id
    }

	pub fn add_edge(&mut self, from_id: usize, to_id: usize) {
		self.belief_nodes[from_id].children.push(to_id);
		self.belief_nodes[to_id].parents.push(from_id);
    }
    
    pub fn belief_id(&self, belief_state: &BeliefState) -> usize {
        self.reachable_belief_states.iter().position(|belief| belief == belief_state).expect("belief state should be found here") // TODO: improve
    }
}

impl<const N: usize> Graph<N> for PRMBeliefGraph<N> {
	fn node(&self, id:usize) -> &dyn GraphNode<N> {
		&self.belief_nodes[id]
	}
	fn n_nodes(&self) -> usize {
		self.belief_nodes.len()
	}
	fn children(&self, id: usize) -> Box<dyn Iterator<Item=usize> + '_> {
		Box::new(self.belief_nodes[id].children.iter().map(|&id| id))
	}
	fn parents(&self, id:usize) -> Box<dyn Iterator<Item=usize> + '_> {
		Box::new(self.belief_nodes[id].parents.iter().map(|&id| id))
	}
}

impl<const N: usize> ObservationGraph for PRMBeliefGraph<N> {
    fn siblings(&self, parent_id: usize, id: usize) -> Vec<(usize, f64)> {

        let mut siblings: Vec<(usize, f64)> = Vec::new();

        let parent = &self.belief_nodes[parent_id];

        match parent.node_type {
            BeliefNodeType::Action => siblings.push((id, 1.0)),
            BeliefNodeType::Observation => {
                for &child_id in &parent.children {
                    let child = &self.belief_nodes[child_id];
                    siblings.push((child_id, transition_probability(&parent.belief_state, &child.belief_state)))
                }
            }
            _ => {panic!("belief node type should be determined at this stage");}
        }
       
        siblings
    }
}

pub fn transition_probability(parent_bs: &BeliefState, child_bs: &BeliefState) -> f64 {
    child_bs.iter().zip(parent_bs).fold(0.0, |s, (p, q)| s + if *p > 0.0 { *q } else { 0.0 } )
}

pub fn conditional_dijkstra<'a, F: PRMFuncs<N>, const N: usize>(graph: & (impl Graph<N> + ObservationGraph), final_node_ids: &Vec<usize>, m: &F) -> Vec<f64> {
	// https://fr.wikipedia.org/wiki/Algorithme_de_Dijkstra
	// complexité n log n ;graph.nodes.len()
    let mut dist = vec![std::f64::INFINITY; graph.n_nodes()];
	let mut q = PriorityQueue::new();
    
    println!("number of belief nodes:{}", graph.n_nodes());

	for &id in final_node_ids {
		dist[id] = 0.0;
        q.push(id, Priority{prio: 0.0});
	}

    let mut it = 0;
	while !q.is_empty() {
        it+=1;
        let (v_id, _) = q.pop().unwrap();
        
        // debug
        if it % 10000 == 0 {
            println!("number of iterations:{}", it);
            println!("queue size:{}, v_id:{}", q.len(), v_id);
        }
        //

		for u_id in graph.parents(v_id) {
            let u = &graph.node(u_id);

            let mut alternative = 0.0;
            for (vv_id, p) in graph.siblings(u_id, v_id) {
                let vv = graph.node(vv_id);
                alternative += p * (m.cost_evaluator(u.state(), vv.state()) + dist[vv_id]);
            }
            
			if alternative < dist[u_id] {
                dist[u_id] = alternative;
                q.push(u_id, Priority{prio: alternative});
            }
		}
	}

	dist
}

#[cfg(test)]
mod tests {

use super::*;

fn create_graph_1(belief_states: &Vec<Vec<f64>>) -> PRMBeliefGraph<2> {
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
    (4)

    bs: [p, 1.0 - p]
    */

    /*    
     10
    / \
   9  ( )
   |   |
   7   8
    \ /
     6
     |
     5

    bs: [1.0, 0.0]
    */

    /*    
    16
    / \
  ( )  15
   |   |
   13  14
    \ /
     12
     |
     11

    bs: [0.0, 1.0]
    */
    let mut belief_graph = PRMBeliefGraph{belief_nodes: Vec::new(), reachable_belief_states: Vec::new()};
    
    // nodes
    belief_graph.add_node([0.0, 1.0], belief_states[0].clone(), 0); // 0
    belief_graph.add_node([-1.0, 2.0], belief_states[0].clone(), 0); // 1
    belief_graph.add_node([1.0, 2.0], belief_states[0].clone(), 0); // 2
    belief_graph.add_node([0.0, 4.0], belief_states[0].clone(), 0); // 3
    belief_graph.add_node([0.0, 0.0], belief_states[0].clone(), 0); // 4


    belief_graph.add_node([0.0, 0.0], belief_states[1].clone(), 1); // 5
    belief_graph.add_node([0.0, 1.0], belief_states[1].clone(), 1); // 6
    belief_graph.add_node([-1.0, 2.0], belief_states[1].clone(), 1);// 7
    belief_graph.add_node([1.0, 2.0], belief_states[1].clone(), 1);// 8
    belief_graph.add_node([-1.0, 3.0], belief_states[1].clone(), 1);// 9
    belief_graph.add_node([0.0, 4.0], belief_states[1].clone(), 1);// 10


    belief_graph.add_node([0.0, 0.0], belief_states[2].clone(), 2); // 11
    belief_graph.add_node([0.0, 1.0], belief_states[2].clone(), 2); // 12
    belief_graph.add_node([-1.0, 2.0], belief_states[2].clone(), 2); // 13
    belief_graph.add_node([1.0, 2.0], belief_states[2].clone(), 2); // 14
    belief_graph.add_node([10.0, 3.0], belief_states[2].clone(), 2); // 15
    belief_graph.add_node([0.0, 4.0], belief_states[2].clone(), 2); // 16


    // edges
    belief_graph.add_edge(0, 1); belief_graph.add_edge(1, 0);
    belief_graph.add_edge(0, 2); belief_graph.add_edge(2, 0);
    belief_graph.add_edge(0, 4); belief_graph.add_edge(4, 0); 

    belief_graph.add_edge(4, 5); // important, belief transition
    belief_graph.add_edge(5, 6); belief_graph.add_edge(6, 5);
    belief_graph.add_edge(6, 7); belief_graph.add_edge(7, 6);
    belief_graph.add_edge(6, 8); belief_graph.add_edge(8, 6);
    belief_graph.add_edge(7, 9); belief_graph.add_edge(9, 7);
    belief_graph.add_edge(9, 10); belief_graph.add_edge(10, 9);

    belief_graph.add_edge(4, 11); // important, belief transition
    belief_graph.add_edge(11, 12); belief_graph.add_edge(12, 11);
    belief_graph.add_edge(12, 13); belief_graph.add_edge(13, 12);
    belief_graph.add_edge(12, 14); belief_graph.add_edge(14, 12);
    belief_graph.add_edge(14, 15); belief_graph.add_edge(15, 14);
    belief_graph.add_edge(15, 16); belief_graph.add_edge(16, 15);

    belief_graph
}

#[test]
fn test_sibling_extraction() {
    let graph = create_graph_1(&vec![vec![0.6, 0.4], vec![1.0, 0.0], vec![0.0, 1.0]]);
    
    assert_eq!(graph.siblings(0, 4), vec![(4, 1.0)]);   // leads itself if no observation
    assert_eq!(graph.siblings(4, 5), vec![(5, 0.6), (11, 0.4)]);
}

#[test]
fn test_conditional_dijkstra() {
    let belief_states = vec![vec![0.4, 0.6], vec![1.0, 0.0], vec![0.0, 1.0]];

    let graph = create_graph_1(&belief_states);
    
    struct Funcs {}
    impl PRMFuncs<2> for Funcs {}
    
    let dists = conditional_dijkstra(&graph, &vec![3, 10, 16], &Funcs{});
    assert!(dists[0] < dists[1]);
    assert!(dists[0] < dists[2]);
    assert!(dists[4] < dists[0]);

    assert!(dists[6] < dists[5]);
    assert!(dists[6] < dists[8]);
    assert!(dists[7] < dists[6]);
    assert!(dists[9] < dists[7]);
    assert!(dists[10] < dists[9]);

    assert!(dists[12] < dists[11]);
    assert!(dists[12] < dists[13]);
    assert!(dists[14] < dists[12]);
    assert!(dists[15] < dists[14]);
    assert!(dists[16] < dists[15]);

    // belief transition
    assert_eq!(dists[4], belief_states[0][0] * dists[5] + belief_states[0][1] * dists[11]);
}

#[test]
fn test_transitions() {
    assert_eq!(transition_probability(&vec![1.0, 0.0], &vec![1.0, 0.0]), 1.0);
    assert_eq!(transition_probability(&vec![0.0, 1.0], &vec![1.0, 0.0]), 0.0);

    assert_eq!(transition_probability(&vec![0.4, 0.6], &vec![0.4, 0.6]), 1.0);
    assert_eq!(transition_probability(&vec![0.4, 0.6], &vec![1.0, 0.0]), 0.4);
    assert_eq!(transition_probability(&vec![0.5, 0.0, 0.5, 0.0], &vec![0.0, 0.5, 0.0, 0.5]), 0.0);
}
}