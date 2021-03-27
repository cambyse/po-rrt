use itertools::{all, enumerate, izip, merge, zip};

use crate::common::*;
use crate::nearest_neighbor::*;
use crate::sample_space::*;
use crate::map_io::*; // tests only
use bitvec::prelude::*;
use priority_queue::PriorityQueue;
use std::{collections::HashMap, ops::Index};

pub trait IBeliefNode<const N: usize> {
    fn state(&self) -> &[f64; N];
    fn belief_id(&self) -> usize;
    fn node_type(&self) -> &BeliefNodeType;
    fn children(&self) -> &[usize];
	fn parents(&self) -> &[usize];
}

pub trait IBeliefGraph<const N: usize> {
	fn node(&self, id:usize) -> &dyn IBeliefNode<N>;
    fn n_nodes(&self) -> usize;
    fn belief_state(&self, id:usize) -> &BeliefState;
}

#[derive(PartialEq, Debug)]
pub enum BeliefNodeType {
    Unknown,
    Action,
    Observation,
}

pub struct BeliefNode<const N: usize> {
	pub state: [f64; N],
    pub belief_state: BeliefState,
    pub belief_id: usize,
	pub parents: Vec<usize>,
    pub children: Vec<usize>,
    pub node_type: BeliefNodeType,
}

pub struct BeliefGraph<const N: usize> {
    pub nodes: Vec<BeliefNode<N>>,
    pub reachable_belief_states: Vec<Vec<f64>>
}

impl<const N: usize> BeliefGraph<N> {
	pub fn add_node(&mut self, state: [f64; N], belief_state: BeliefState, belief_id: usize, node_type: BeliefNodeType) -> usize {
        let id = self.nodes.len();
        self.nodes.push(
            BeliefNode{
                state,
                belief_state,
                belief_id,
                parents: Vec::new(),
                children: Vec::new(),
                node_type,
            }
        );
        id
    }

	pub fn add_edge(&mut self, from_id: usize, to_id: usize) {
		self.nodes[from_id].children.push(to_id);
		self.nodes[to_id].parents.push(from_id);
    }
    
    #[allow(clippy::style)]
    pub fn belief_id(&self, belief_state: &BeliefState) -> usize {
        self.reachable_belief_states.iter().position(|belief| belief == belief_state).expect("belief state should be found here") // TODO: improve
    }
}

#[allow(clippy::style)]
pub fn transition_probability(parent_bs: &BeliefState, child_bs: &BeliefState) -> f64 {
    child_bs.iter().zip(parent_bs).fold(0.0, |s, (p, q)| s + if *p > 0.0 { *q } else { 0.0 } )
}

// Belief graph trait
impl<const N: usize> IBeliefNode<N> for BeliefNode<N> {
    fn state(&self) -> &[f64; N] {
        &self.state
    }

    fn belief_id(&self) -> usize {
        self.belief_id
    }

    fn node_type(&self) -> &BeliefNodeType {
        &self.node_type
    }

    fn children(&self) -> &[usize] {
        &self.children
    }

	fn parents(&self) -> &[usize] {
        &self.parents
    }
}

impl<const N: usize> IBeliefGraph<N> for BeliefGraph<N> {
	fn node(&self, id:usize) -> &dyn IBeliefNode<N> {
        &self.nodes[id]
    }
	fn n_nodes(&self) -> usize {
        self.nodes.len()
    }
    fn belief_state(&self, id:usize) -> &BeliefState {
        &self.nodes[id].belief_state
    }
}
//

pub fn conditional_dijkstra<const N: usize>(graph: &impl IBeliefGraph<N>, final_node_ids: &[usize], cost_evaluator: impl Fn(&[f64; N], &[f64; N]) -> f64) -> Vec<f64> {
	// https://fr.wikipedia.org/wiki/Algorithme_de_Dijkstra
	// complexité n log n ;graph.nodes.len()
    let mut dist = vec![std::f64::INFINITY; graph.n_nodes()];
	let mut q = PriorityQueue::new();
    
    // debug
    println!("number of belief nodes:{}", graph.n_nodes());
    // 

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

		for &u_id in graph.node(v_id).parents() {
            let u = graph.node(u_id);

            let mut alternative = 0.0;
            if *u.node_type() == BeliefNodeType::Action {
                let v = graph.node(v_id);
                alternative += cost_evaluator(u.state(), v.state()) + dist[v_id]
            }
            else if *u.node_type() == BeliefNodeType::Observation {
                for &vv_id in u.children() {
                    let vv = graph.node(vv_id);
                    let p = transition_probability(graph.belief_state(u_id), graph.belief_state(vv_id));

                    //println!("belief avant:{:?} apres:{:?}", graph.belief_state(u_id), graph.belief_state(vv_id));
                    //assert_eq!(u.children().len(), 2);

                    alternative += p * (cost_evaluator(u.state(), vv.state()) + dist[vv_id]);
                }

                //println!("alternative: {}", alternative);
            }
            else {
                panic!("node type should be know at this stage!");
            }

			if alternative < dist[u_id] {
                dist[u_id] = alternative;
                q.push(u_id, Priority{prio: alternative});
            }
		}
    }

    // checks 
    /*
    for id in 0..graph.n_nodes() {
        let n = graph.node(id);

        if *n.node_type() == BeliefNodeType::Observation {
            println!("belief: {:?}, cost:{}", graph.belief_state(id), dist[id]);
        }

        if dist[id] < 5.0 && !final_node_ids.contains(&id) {
            assert!(n.children().len() > 0);
            if n.children().len() == 0 {
                println!("pb!!!, node_type:{:?}", n.node_type());
            }
        }

        for child_id in n.children() {
            let o = graph.node(*child_id);

            assert!(o.parents().contains(&id));

            if ! o.parents().contains(&id) {
                println!("pb!!!, node_type:{:?}", n.node_type());
            }
        }
    }
    */
    // debug
    println!("conditional dijkstra finished..");
    // 

	dist
}

pub fn extract_policy<const N: usize>(graph: &impl IBeliefGraph<N>, expected_costs_to_goals: &Vec<f64>) -> Policy<N> {
    if graph.n_nodes() == 0 {
        panic!("no belief state graph!");
    }

    let mut policy: Policy<N> = Policy{nodes: Vec::new()};
    let mut lifo: Vec<(usize, usize)> = Vec::new(); // policy_node, belief_graph_node

    policy.add_node(&graph.node(0).state(), &graph.belief_state(0));

    lifo.push((0, 0));

    while !lifo.is_empty() {
        let (policy_node_id, belief_node_id) = lifo.pop().unwrap();

        let children_ids = get_best_expected_children(graph, belief_node_id, expected_costs_to_goals);

        for child_id in children_ids {
            let child = &graph.node(child_id);
            let child_policy_id = policy.add_node(child.state(), graph.belief_state(child_id));
            policy.add_edge(policy_node_id, child_policy_id);

            //println!("add node, belief {:?}, cost: {:?}", &graph.belief_state(child_id), &expected_costs_to_goals[child_id]);

            if expected_costs_to_goals[child_id] > 0.0 {
                lifo.push((child_policy_id, child_id));
            }
        }
    }
    policy
}

pub fn get_best_expected_children<const N: usize>(graph: &impl IBeliefGraph<N>, belief_node_id: usize, expected_costs_to_goals: &Vec<f64>) -> Vec<usize> {    
    // cluster children by target belief state
    let mut belief_to_children = HashMap::new();
    for &child_id in graph.node(belief_node_id).children() {
        let child = graph.node(child_id);

        belief_to_children.entry(child.belief_id()).or_insert_with(Vec::new);
        belief_to_children.get_mut(&child.belief_id()).unwrap().push((child_id, expected_costs_to_goals[child_id]));
    }

    // choose the best for each belief state
    let mut best_children: Vec<usize> = Vec::new();

    for belief_id in belief_to_children.keys() {
        let mut best_id = belief_to_children[belief_id][0].0;
        let p = transition_probability(graph.belief_state(belief_node_id), &graph.belief_state(best_id));

        assert!(p > 0.0);
        
        let mut best_cost = p * belief_to_children[belief_id][0].1;
        for (child_id, cost) in belief_to_children[belief_id].iter().skip(0) {
            if p * *cost < best_cost {
                best_cost = p * *cost;
                best_id = *child_id;
            }
        }

        assert!(p * expected_costs_to_goals[best_id] <= expected_costs_to_goals[belief_node_id]);

        best_children.push(best_id);
    }

    best_children
}    

    

#[cfg(test)]
mod tests {

use super::*;

fn create_graph_1(belief_states: &Vec<Vec<f64>>) -> BeliefGraph<2> {
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
    let mut belief_graph = BeliefGraph{nodes: Vec::new(), reachable_belief_states: Vec::new()};
    
    // nodes
    belief_graph.add_node([0.0, 1.0], belief_states[0].clone(), 0, BeliefNodeType::Action); // 0
    belief_graph.add_node([-1.0, 2.0], belief_states[0].clone(), 0, BeliefNodeType::Action); // 1
    belief_graph.add_node([1.0, 2.0], belief_states[0].clone(), 0, BeliefNodeType::Action); // 2
    belief_graph.add_node([0.0, 4.0], belief_states[0].clone(), 0, BeliefNodeType::Action); // 3
    belief_graph.add_node([0.0, 0.0], belief_states[0].clone(), 0, BeliefNodeType::Observation); // 4


    belief_graph.add_node([0.0, 0.0], belief_states[1].clone(), 1, BeliefNodeType::Action); // 5
    belief_graph.add_node([0.0, 1.0], belief_states[1].clone(), 1, BeliefNodeType::Action); // 6
    belief_graph.add_node([-1.0, 2.0], belief_states[1].clone(), 1, BeliefNodeType::Action);// 7
    belief_graph.add_node([1.0, 2.0], belief_states[1].clone(), 1, BeliefNodeType::Action);// 8
    belief_graph.add_node([-1.0, 3.0], belief_states[1].clone(), 1, BeliefNodeType::Action);// 9
    belief_graph.add_node([0.0, 4.0], belief_states[1].clone(), 1, BeliefNodeType::Action);// 10


    belief_graph.add_node([0.0, 0.0], belief_states[2].clone(), 2, BeliefNodeType::Action); // 11
    belief_graph.add_node([0.0, 1.0], belief_states[2].clone(), 2, BeliefNodeType::Action); // 12
    belief_graph.add_node([-1.0, 2.0], belief_states[2].clone(), 2, BeliefNodeType::Action); // 13
    belief_graph.add_node([1.0, 2.0], belief_states[2].clone(), 2, BeliefNodeType::Action); // 14
    belief_graph.add_node([10.0, 3.0], belief_states[2].clone(), 2, BeliefNodeType::Action); // 15
    belief_graph.add_node([0.0, 4.0], belief_states[2].clone(), 2, BeliefNodeType::Action); // 16


    // edges
    belief_graph.add_edge(0, 1); belief_graph.add_edge(1, 0);
    belief_graph.add_edge(0, 2); belief_graph.add_edge(2, 0);
    belief_graph.add_edge(0, 4);

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
fn test_conditional_dijkstra() {
    let belief_states = vec![vec![0.4, 0.6], vec![1.0, 0.0], vec![0.0, 1.0]];

    let graph = create_graph_1(&belief_states);
    
    let dists = conditional_dijkstra(&graph, &vec![3, 10, 16], |a: &[f64; 2], b: &[f64; 2]| norm2(a, b) );
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