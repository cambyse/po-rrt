use itertools::{all, enumerate, izip, merge, zip};

use crate::common::*;
use bitvec::prelude::*;

pub struct Reachability {
	validity: Vec<WorldMask>,
	reachability: Vec<WorldMask>,
	final_node_ids: Vec<usize>,
	finality: Vec<WorldMask>
}

impl Reachability {
	pub fn new() -> Self {
		Self{ validity: Vec::new(), reachability: Vec::new(), final_node_ids: Vec::new(), finality: Vec::new() }
	}

	pub fn set_root(&mut self, validity: WorldMask) {
		self.validity.push(validity.clone());
		self.reachability.push(validity);
	}

	pub fn add_node(&mut self, validity: WorldMask) {
		self.validity.push(validity.clone());
		self.reachability.push(bitvec![0; validity.len()]);
	}

	pub fn add_final_node(&mut self, id: usize, finality: WorldMask) {
		self.final_node_ids.push(id);
		self.finality.push(finality);
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

		// this version appears to be the fastest, see cargo bench
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
		let mut finality = self.finality[0].clone();
		dbg!(&finality);

		for (&final_node_id, node_finality) in self.final_node_ids.iter().zip(self.finality.iter()) {
			let node_reachability = &self.reachability[final_node_id];
			for i in 0..node_reachability.len() {
				let &finality_i = &finality[i];
				finality.set(i, finality_i || node_reachability[i] && node_finality[i]);
			}
		}
		/*self.final_node_ids.iter().skip(0)
			.fold(first_finality, |finality, &id| finality | (self.reachability[id].clone() ) )
			.all()*/

		finality.all()
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

	reachability.add_final_node(2, bitvec![1,1]);
	assert_eq!(reachability.is_final_set_complete(), false);

	reachability.add_final_node(3, bitvec![1,1]);
	assert_eq!(reachability.is_final_set_complete(), true);

	assert_eq!(reachability.final_nodes_for_world(0), vec![2]);
	assert_eq!(reachability.final_nodes_for_world(1), vec![3]);
}

#[test]
fn test_final_nodes_completness_when_different_goals_for_different_worlds() {
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
	reachability.add_node(bitvec![1,1]); // 2
	reachability.add_node(bitvec![1,1]); // 3

	reachability.add_edge(0, 1);
	reachability.add_edge(1, 2);
	reachability.add_edge(1, 3);

	reachability.add_final_node(2, bitvec![1,0]);
	assert_eq!(reachability.is_final_set_complete(), false);

	reachability.add_final_node(3, bitvec![0,1]);
	assert_eq!(reachability.is_final_set_complete(), true);

	assert_eq!(reachability.final_nodes_for_world(0), vec![2]);
	assert_eq!(reachability.final_nodes_for_world(1), vec![3]);
}

}