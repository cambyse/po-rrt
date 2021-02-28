use itertools::{all, enumerate, izip, merge, zip};

use crate::common::*;
use crate::prm_graph::*;
use bitvec::prelude::*;

pub struct Reachability {
	validity: Vec<WorldMask>,
	reachability: Vec<WorldMask>,
	final_node_ids: Vec<usize>
}

impl Reachability {
	pub fn new() -> Self {
		Self{ validity: Vec::new(), reachability: Vec::new(), final_node_ids: Vec::new() }
	}

	pub fn set_root(&mut self, validity: WorldMask) {
		self.validity.push(validity.clone());
		self.reachability.push(validity);
	}

	pub fn add_node(&mut self, validity: WorldMask) {
		self.validity.push(validity.clone());
		self.reachability.push(bitvec![0; validity.len()]);
	}

	pub fn add_final_node(&mut self, id: usize) {
		self.final_node_ids.push(id);
	}

	pub fn add_edge(&mut self, from: usize, to: usize) {
		/*let mut tmp = self.reachability[from].clone();
		tmp &= self.validity[to].clone();
		self.reachability[to] |= tmp;
		*/

		/*self.reachability[to] = 
		izip!(self.reachability[from].iter(), self.reachability[to].iter(), self.validity[to].iter())
		.map(|(r_from, r_to, v_to)| *r_to || (*r_from && *v_to) )
		.collect();*/

		// this version appears to be the fastest
		for i in 0..self.reachability[to].len() {
			let r_to = self.reachability[to][i];
			let r_from = self.reachability[from][i];
			let v_to = self.validity[to][i];
			self.reachability[to].set(i, r_to || r_from && v_to);
		}
	}

	pub fn reachability(&self, id: usize) -> &WorldMask {
		&self.reachability[id]
	}

	pub fn final_nodes_for_world(&self, world: usize) -> Vec<usize> {
		self.final_node_ids.iter()
			.filter(|&id| self.reachability[*id][world])
			.map(|&id| id)
			.collect()
	}

	pub fn is_final_set_complete(&self) -> bool {
		if self.final_node_ids.is_empty() { return false; }

		// get first elements as starting point..
		let &first_final_id = self.final_node_ids.first().unwrap();
		let first_reachability = self.reachability[first_final_id].clone();

		self.final_node_ids.iter().skip(0)
			.fold(first_reachability, |reachability, &id| reachability | self.reachability[id].clone())
			.all()
	}
}

#[cfg(test)]
mod tests {

use super::*;

#[test]
fn test_reachability() {
	/*
		0
		|
		1
	   / \
	  2   3
	*/
	let mut reachability = Reachability::new();

	reachability.set_root(bitvec![1,1]); // 0
	reachability.add_node(bitvec![1,0]); // 1
	reachability.add_node(bitvec![1,0]); // 2
	reachability.add_node(bitvec![0,1]); // 3

	reachability.add_edge(0, 1);
	reachability.add_edge(1, 2);
	reachability.add_edge(1, 3);

	assert_eq!(reachability.reachability(0), &bitvec![1,1]);
	assert_eq!(reachability.reachability(1), &bitvec![1,0]);
	assert_eq!(reachability.reachability(2), &bitvec![1,0]);
	assert_eq!(reachability.reachability(3), &bitvec![0,0]);
}

#[test]
fn test_reachability_diamond_shape() {
	/*
		0
	   / \
	  1   2
	   \ /
	    3
	*/
	let mut reachability = Reachability::new();

	reachability.set_root(bitvec![1,1]); // 0
	reachability.add_node(bitvec![1,0]); // 1
	reachability.add_node(bitvec![0,1]); // 2
	reachability.add_node(bitvec![1,1]); // 3

	reachability.add_edge(0, 1);
	reachability.add_edge(0, 2);
	reachability.add_edge(1, 3);
	reachability.add_edge(2, 3);

	assert_eq!(reachability.reachability(0), &bitvec![1,1]);
	assert_eq!(reachability.reachability(1), &bitvec![1,0]);
	assert_eq!(reachability.reachability(2), &bitvec![0,1]);
	assert_eq!(reachability.reachability(3), &bitvec![1,1]);
}

#[test]
fn test_final_nodes_completness() {
	/*
		0
		|
		1
	   / \
	  2   3
	*/
	let mut reachability = Reachability::new();

	reachability.set_root(bitvec![1,1]); // 0
	reachability.add_node(bitvec![1,1]); // 1
	reachability.add_node(bitvec![1,0]); // 2
	reachability.add_node(bitvec![0,1]); // 3

	reachability.add_edge(0, 1);
	reachability.add_edge(1, 2);
	reachability.add_edge(1, 3);

	assert_eq!(reachability.is_final_set_complete(), false);

	reachability.add_final_node(2);
	assert_eq!(reachability.is_final_set_complete(), false);

	reachability.add_final_node(3);
	assert_eq!(reachability.is_final_set_complete(), true);

	assert_eq!(reachability.final_nodes_for_world(0), vec![2]);
	assert_eq!(reachability.final_nodes_for_world(1), vec![3]);
}

}