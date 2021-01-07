use crate::common::*;
use std::cell::RefCell;
use std::rc::Rc;

enum InsertionSide {
	Left,
	Right
}

type KdNodeRef<const N: usize> = Rc<RefCell<KdNode<N>>>; // is RefCell really needed?

pub struct KdNode<const N: usize> {
	pub id: usize,    
	pub state: [f64; N],
	axis: usize,
	splitting_value: f64,
	left: Option<KdNodeRef<N>>,
	right: Option<KdNodeRef<N>>,
}

impl<const N: usize> Node<N> {
	pub fn root(state: [f64; N]) ->KdNode<N> {
		KdNode{ id: 0, state: state, axis:0, splitting_value: state[0], left: None, right: None }
	}
}

pub struct KdTree<const N: usize> {
	pub root: Option<KdNodeRef<N>>
}

impl<const N: usize> KdTree<N> {
	pub fn new(state: [f64; N]) -> KdTree<N> {
		let n = Node::root(state);
		let rnode = Rc::new(RefCell::new(n));
		KdTree::<N>{ root: Some(rnode.clone()) }
	}

	pub fn add(&self, state: [f64; N], id: usize) {
		let mut current = self.root.as_ref().unwrap().clone();

		let insertion_side: Option<InsertionSide>;

		loop {
			let shallow_clone = current.clone(); // is there a way to avoid clones?
			if state[shallow_clone.borrow().axis] < shallow_clone.borrow().splitting_value {
				match &shallow_clone.borrow().left {
					Some(left) => current = left.clone(), // is there a way to avoid clones?
					None => {insertion_side = Some(InsertionSide::Left); break;},
				};
			}
			else {
				match &shallow_clone.borrow().right {
					Some(right) => current = right.clone(), // is there a way to avoid clones?
					None => {insertion_side = Some(InsertionSide::Right); break;},
				};
			}
		}

		let axis = (current.borrow().axis + 1) % N;
		let n = KdNode{
			id,
			state,
			axis,
			splitting_value: state[axis],
			left: None,
			right: None
		};

		match insertion_side.expect("insertion side must be determined beforehand") {
			InsertionSide::Left => current.borrow_mut().left = Some(Rc::new(RefCell::new(n))),
			InsertionSide::Right => current.borrow_mut().right = Some(Rc::new(RefCell::new(n))),
		}
	}

	pub fn nearest_neighbor(&self, state: [f64; N]) -> KdNodeRef<N> {
		let mut dmin = f64::INFINITY;
		let mut nearest = self.root.as_ref().unwrap().clone();

		KdTree::nearest_neighbor_from(state, self.root.as_ref().unwrap().clone(), &mut nearest, &mut dmin);

		nearest
	}

	fn nearest_neighbor_from(state: [f64; N], from: KdNodeRef<N>, nearest: &mut KdNodeRef<N>, dmin: &mut f64) {
		let d = norm2(&from.borrow().state, &state);

		// check current
		if d < *dmin {
			*dmin = d;
			*nearest = from.clone();
		}

		// go down
		if state[from.borrow().axis] < from.borrow().splitting_value { 
			// left first
			if state[from.borrow().axis] - *dmin < from.borrow().splitting_value && from.borrow().left.is_some() {
				KdTree::nearest_neighbor_from(state, from.borrow().left.as_ref().unwrap().clone(), nearest, dmin);
			}

			if state[from.borrow().axis] + *dmin >= from.borrow().splitting_value && from.borrow().right.is_some() {
				KdTree::nearest_neighbor_from(state, from.borrow().right.as_ref().unwrap().clone(), nearest, dmin);
			}
		}
		else { 
			// right first
			if state[from.borrow().axis] + *dmin >= from.borrow().splitting_value && from.borrow().right.is_some() {
				KdTree::nearest_neighbor_from(state, from.borrow().right.as_ref().unwrap().clone(), nearest, dmin);
			}

			if state[from.borrow().axis] - *dmin < from.borrow().splitting_value && from.borrow().left.is_some() {
				KdTree::nearest_neighbor_from(state, from.borrow().left.as_ref().unwrap().clone(), nearest, dmin);
			}
		}
	}
}


#[cfg(test)]
mod tests {

use super::*;

// tree creation
fn check_node(node: &KdNode<2>, id: usize, state: [f64; 2], axis: usize, splitting_value: f64) {
	assert_eq!(node.axis, axis);
	assert_eq!(node.splitting_value, splitting_value);
	assert_eq!(node.state, state);
	assert_eq!(node.id, id);
}

#[test]
fn test_kdtree_creation() {
	let tree = KdTree::new([3.0, 6.0]);
	let root = tree.root.unwrap();
	assert_eq!(root.borrow().id, 0);
	assert!(root.borrow().left.is_none());
	assert!(root.borrow().right.is_none());
	assert_eq!(root.borrow().splitting_value, 3.0);
}

#[test]
fn test_add_second_level_left() {
	let tree = KdTree::new([3.0, 6.0]);
	tree.add([2.0, 7.0], 1);
	
	let root = tree.root.unwrap();
	let left = root.borrow().left.as_ref().unwrap().clone();

	assert!(root.borrow().right.is_none());
	check_node(&left.borrow(), 1, [2.0, 7.0], 1, 7.0);
}

#[test]
fn test_add_second_level_right() {
	let tree = KdTree::new([3.0, 6.0]);
	tree.add([17.0, 15.0], 1);
	
	let root = tree.root.unwrap();
	let right = root.borrow().right.as_ref().unwrap().clone();

	assert!(root.borrow().left.is_none());
	check_node(&right.borrow(), 1, [17.0, 15.0], 1, 15.0);
}

// see examples https://www.geeksforgeeks.org/k-dimensional-tree/
#[test]
fn test_full_tree() {
	let tree = KdTree::new([3.0, 6.0]);

	tree.add([17.0, 15.0], 1);
	tree.add([13.0, 15.0], 2);
	tree.add([6.0, 12.0], 3);
	tree.add([9.0, 1.0], 4);
	tree.add([2.0, 7.0], 5);
	tree.add([10.0, 19.0], 6);

	// BFS left right
	let root = tree.root.unwrap();
	{
		let node = root.borrow().left.as_ref().unwrap().clone();

		check_node(&node.borrow(), 5, [2.0, 7.0], 1, 7.0);
	}
	{
		let node = root.borrow().right.as_ref().unwrap().clone();

		check_node(&node.borrow(), 1, [17.0, 15.0], 1, 15.0);
	}
	{
		let node = root.borrow().right.as_ref().unwrap().borrow()
								.left.as_ref().unwrap().clone();

		check_node(&node.borrow(), 3, [6.0, 12.0], 0, 6.0);
	}
	{
		let node = root.borrow().right.as_ref().unwrap().borrow()
								.right.as_ref().unwrap().clone();

		check_node(&node.borrow(), 2, [13.0, 15.0], 0, 13.0);
	}
	{
		let node = root.borrow().right.as_ref().unwrap().borrow()
								.left.as_ref().unwrap().borrow()
								.right.as_ref().unwrap().clone();

		check_node(&node.borrow(), 4, [9.0, 1.0], 1, 1.0);
	}
	{
		let node = root.borrow().right.as_ref().unwrap().borrow()
								.right.as_ref().unwrap().borrow()
								.left.as_ref().unwrap().clone();

		check_node(&node.borrow(), 6, [10.0, 19.0], 1, 19.0);
	}
}

// query nearest neighbors
#[test]
fn test_nearest_neighbor() {
	let tree = KdTree::new([3.0, 6.0]);

	tree.add([17.0, 15.0], 1);
	tree.add([13.0, 15.0], 2);
	tree.add([6.0, 12.0], 3);
	tree.add([9.0, 1.0], 4);
	tree.add([2.0, 7.0], 5);
	tree.add([10.0, 19.0], 6);

	{
		let node = tree.nearest_neighbor([17.0, 15.0]);
		assert_eq!(node.borrow().state, [17.0, 15.0]);
	}

	{
		let node = tree.nearest_neighbor([9.1, 1.0]);
		assert_eq!(node.borrow().state, [9.0, 1.0]);
	}
	 
	{
		let node = tree.nearest_neighbor([2.0, 8.0]);
		assert_eq!(node.borrow().state, [2.0, 7.0]);
	}	
}

/*
#[test]
fn test_kd_tree_lib() {
	// construct kd-tree
	let kdtree = kd_tree::KdTree::build_by_ordered_float(vec![
		[1.0, 2.0, 3.0],
		[3.0, 1.0, 2.0],
		[2.0, 3.0, 1.0],
	]);

	// search the nearest neighbor
	let found = kdtree.nearest(&[3.1, 0.9, 2.1]).unwrap();
	assert_eq!(found.item, &[3.0, 1.0, 2.0]);

	// search k-nearest neighbors
	let found = kdtree.nearests(&[1.5, 2.5, 1.8], 2);
	assert_eq!(found[0].item, &[2.0, 3.0, 1.0]);
	assert_eq!(found[1].item, &[1.0, 2.0, 3.0]);

	// search points within a sphere
	let found = kdtree.within_radius(&[2.0, 1.5, 2.5], 1.5);
	assert_eq!(found.len(), 2);
	assert!(found.iter().any(|&&p| p == [1.0, 2.0, 3.0]));
	assert!(found.iter().any(|&&p| p == [3.0, 1.0, 2.0]));
}*/
}