use itertools::{all, enumerate, izip, merge, zip};

use crate::common::*;
use crate::nearest_neighbor::*;
use crate::sample_space::*;
use crate::map_io::*; // tests only
use crate::prm_graph::*;
use crate::prm_reachability::*;
use bitvec::prelude::*;

pub struct PRMBeliefNode<const N: usize> {
	pub state: [f64; N],
	pub belief_state: BeliefState,
	pub parents: Vec<usize>,
    pub children: Vec<usize>,
    pub graph_node_id: usize, // to which node in roadmap graph it refers to
}

pub struct PRMBeliefGraph<const N: usize> {
	pub belief_nodes: Vec<PRMBeliefNode<N>>,
}

impl<const N: usize> PRMBeliefGraph<N> {
	pub fn add_node(&mut self, state: [f64; N], belief_state: BeliefState, graph_node_id:usize) -> usize {
        self.belief_nodes.push(
            PRMBeliefGraph{
                state,
                belief_state,
                parents: Vec::new(),
                children: Vec::new(),
                graph_node_id
            }
        );
    }
}


#[cfg(test)]
mod tests {

use super::*;

fn create_minimal_graph() -> PRMBeliefGraph<2> {
    let mut belief_graph = PRMBeliefGraph{nodes: Vec::new()};

}
}