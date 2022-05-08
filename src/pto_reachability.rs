use itertools::{all, enumerate, izip, merge, zip};
use std::{collections::HashSet, iter::Iterator, iter::Zip, slice::Iter};
use crate::common::*;
use bitvec::prelude::*;

pub struct Reachability {
	validities: Vec<WorldMask>,
	reachabilities: Vec<WorldMask>,
	final_node_ids: Vec<usize>,
	final_node_ids_set: HashSet<usize>,
	finalities: Vec<WorldMask>,
	finality: WorldMask,
	n_worlds: usize,
	dirty: bool
}

impl Reachability {
	#[allow(clippy::new_without_default)]
	pub fn new() -> Self {
		Self{ validities: Vec::new(), reachabilities: Vec::new(), final_node_ids: Vec::new(), final_node_ids_set: HashSet::new(), finalities: Vec::new(), finality: bitvec![], n_worlds: 0, dirty: false }
	}

	pub fn set_root(&mut self, validity: WorldMask) {
		self.n_worlds = validity.len();
		self.validities.push(validity.clone());
		self.reachabilities.push(validity);
		self.finality = bitvec![0; self.n_worlds];
	}

	pub fn add_node(&mut self, validity: WorldMask) {
		self.validities.push(validity.clone());
		self.reachabilities.push(bitvec![0; validity.len()]);
	}

	pub fn add_final_node(&mut self, id: usize, finality: WorldMask) {
		self.final_node_ids.push(id);
		self.final_node_ids_set.insert(id);
		self.finalities.push(finality);
		self.dirty = true;
	}

	pub fn add_edge(&mut self, from: usize, to: usize, edge_validity: &WorldMask) {
		// this version appears to be the fastest, see cargo bench
		for i in 0..self.reachabilities[to].len() {
			let r_to = self.reachabilities[to][i];
			let r_from = self.reachabilities[from][i];
			let v_from_to = edge_validity[i];
			self.reachabilities[to].set(i, r_to || r_from && v_from_to);

			if self.final_node_ids_set.contains(&to) { self.dirty = true; }
		}
	}

	pub fn reachability(&self, id: usize) -> &WorldMask {
		&self.reachabilities[id]
	}

	pub fn get_final_nodes_for_world(&self, world: usize) -> Vec<usize> {
		self.final_node_ids.iter().enumerate()
			.filter(|(i, &id)| self.reachabilities[id][world] && self.finalities[*i][world])
			.map(|(_, &id)| id)
			.collect()
	}

	pub fn get_final_node_ids(&self) -> Vec<usize> {
		let mut final_node_ids = Vec::new();
		for world in 0..self.n_worlds {
			for final_id in self.get_final_nodes_for_world(world) {
				if ! final_node_ids.contains(&final_id) {
					final_node_ids.push(final_id);
				}
			}
		}
		final_node_ids
	}

	pub fn final_nodes_with_validities(&self) -> Zip<Iter<usize>, Iter<WorldMask>> {
		self.final_node_ids.iter().zip(self.finalities.iter())
	}

	pub fn is_final_set_complete(&mut self) -> bool {
		if self.final_node_ids.is_empty() { return false; }
		
		if self.dirty {
			self.update_finality();	
			self.dirty = false;
		}

		self.finality.all()
	}

	fn update_finality(&mut self) {
		let finality = &mut self.finality;
		for (&final_node_id, node_finality) in self.final_node_ids.iter().zip(self.finalities.iter()) {
			let node_reachability = &self.reachabilities[final_node_id];
			for i in 0..node_reachability.len() {
				let &finality_i = &finality[i];
				finality.set(i, finality_i || node_reachability[i] && node_finality[i]);
			}
		}
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

	reachability.add_edge(0, 1, &bitvec![1,0]);
	reachability.add_edge(1, 2, &bitvec![1,0]);
	reachability.add_edge(1, 3, &bitvec![0,1]);

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

	reachability.add_edge(0, 1, &bitvec![1,0]);
	reachability.add_edge(0, 2, &bitvec![0,1]);
	reachability.add_edge(1, 3, &bitvec![1,1]);
	reachability.add_edge(2, 3, &bitvec![1,1]);

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

	reachability.add_edge(0, 1, &bitvec![1,1]);
	reachability.add_edge(1, 2, &bitvec![1,0]);
	reachability.add_edge(1, 3, &bitvec![0,1]);

	assert_eq!(reachability.is_final_set_complete(), false);

	reachability.add_final_node(2, bitvec![1,1]);
	assert_eq!(reachability.is_final_set_complete(), false);

	reachability.add_final_node(3, bitvec![1,1]);
	assert_eq!(reachability.is_final_set_complete(), true);

	assert_eq!(reachability.get_final_nodes_for_world(0), vec![2]);
	assert_eq!(reachability.get_final_nodes_for_world(1), vec![3]);
}

#[test]
fn test_final_nodes_completness_when_2_different_goals_for_2_different_worlds() {
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

	reachability.add_edge(0, 1, &bitvec![1,1]);
	reachability.add_edge(1, 2, &bitvec![1,1]);
	reachability.add_edge(1, 3, &bitvec![1,1]);

	reachability.add_final_node(2, bitvec![1,0]);
	assert_eq!(reachability.is_final_set_complete(), false);

	reachability.add_final_node(3, bitvec![0,1]);
	assert_eq!(reachability.is_final_set_complete(), true);

	assert_eq!(reachability.get_final_nodes_for_world(0), vec![2]);
	assert_eq!(reachability.get_final_nodes_for_world(1), vec![3]);

	let final_nodes_and_validities: Vec<(&usize, &WorldMask)> = reachability.final_nodes_with_validities().collect();
	
	assert_eq!(*final_nodes_and_validities[0].0, 2);
	assert_eq!(*final_nodes_and_validities[0].1, bitvec![1,0]);

	assert_eq!(*final_nodes_and_validities[1].0, 3);
	assert_eq!(*final_nodes_and_validities[1].1, bitvec![0,1]);
}

}