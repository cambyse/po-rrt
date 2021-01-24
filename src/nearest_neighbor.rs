use crate::common::*;

pub struct KdNode<const N: usize> {
	pub id: usize,
	pub state: [f64; N],
	left: Option<Box<KdNode<N>>>,
	right: Option<Box<KdNode<N>>>,
}

pub struct KdTree<const N: usize> {
	pub root: KdNode<N>,
}

impl<const N: usize> KdTree<N> {
	pub fn new(state: [f64; N]) -> Self {
		let root = KdNode { id: 0, state, left: None, right: None };
		Self { root }
	}

	pub fn add(&mut self, state: [f64; N], id: usize) {
		let mut current = &mut self.root;
		for axis in (0..N).into_iter().cycle() {
			let next = if state[axis] < current.state[axis] {
				&mut current.left
			} else {
				&mut current.right
			};

			if next.is_some() {
				current = next.as_mut().unwrap();
			} else {
				let node = KdNode { id, state, left: None, right: None };
				*next = Some(Box::new(node));
				return;
			}
		}
	}

	pub fn nearest_neighbor(&self, state: [f64; N]) -> &KdNode<N> {
		struct Args<'a, const N: usize> {
			state: [f64; N],
			dmin: f64,
			nearest: &'a KdNode<N>,
		}

		fn inner<'a, const N: usize>(a: &mut Args<'a, N>, from: &'a KdNode<N>, axis: usize) {
			{
				let d = norm2(&from.state, &a.state);
				if d < a.dmin {
					a.dmin = d;
					a.nearest = from;
				}
			}

			// go down
			let next_axis = (axis+1) % N;
			if a.state[axis] < from.state[axis] {
				// left first
				if a.state[axis] - a.dmin < from.state[axis] && from.left.is_some() {
					inner(a, from.left.as_ref().unwrap(), next_axis);
				}
				if a.state[axis] + a.dmin >= from.state[axis] && from.right.is_some() {
					inner(a, from.right.as_ref().unwrap(), next_axis);
				}
			} else {
				// right first
				if a.state[axis] + a.dmin >= from.state[axis] && from.right.is_some() {
					inner(a, from.right.as_ref().unwrap(), next_axis);
				}
				if a.state[axis] - a.dmin < from.state[axis] && from.left.is_some() {
					inner(a, from.left.as_ref().unwrap(), next_axis);
				}
			}
		}

		let mut a = Args { state, dmin: f64::INFINITY, nearest: &self.root };
		inner(&mut a, &self.root, 0);
		a.nearest
	}

	pub fn nearest_neighbors(&self, state: [f64; N], radius: f64) -> Vec<&KdNode<N>> {
		struct Args<'a, const N: usize> {
			state: [f64; N],
			radius: f64,
			nearest: Vec<&'a KdNode<N>>,
		}

		fn inner<'a, const N: usize>(a: &mut Args<'a, N>, from: &'a KdNode<N>, axis: usize) {
			{
				let d = norm2(&from.state, &a.state);
				if d <= a.radius {
					a.nearest.push(from);
				}
			}

			let next_axis = (axis+1) % N;

			if a.state[axis] - a.radius <= from.state[axis] && from.left.is_some() {
				inner(a, from.left.as_ref().unwrap(), next_axis);
			}
			if a.state[axis] + a.radius >= from.state[axis] && from.right.is_some() {
				inner(a, from.right.as_ref().unwrap(), next_axis);
			}
		}

		let mut a = Args { state, radius, nearest: vec![] };
		inner(&mut a, &self.root, 0);
		a.nearest
	}
}


#[cfg(test)]
mod tests {

use itertools::enumerate;
use super::*;

// tree creation
fn check_node(node: &KdNode<2>, id: usize, state: [f64; 2]) {
	assert_eq!(node.state, state);
	assert_eq!(node.id, id);
}

#[test]
fn test_kdtree_creation() {
	let tree = KdTree::new([3.0, 6.0]);
	let root = &tree.root;
	assert_eq!(root.id, 0);
	assert!(root.left.is_none());
	assert!(root.right.is_none());
}

#[test]
fn test_add_second_level_left() {
	let mut tree = KdTree::new([3.0, 6.0]);
	tree.add([2.0, 7.0], 1);

	assert!(tree.root.right.is_none());
	check_node(tree.root.left.as_ref().unwrap(), 1, [2.0, 7.0]);
}

#[test]
fn test_add_second_level_right() {
	let mut tree = KdTree::new([3.0, 6.0]);
	tree.add([17.0, 15.0], 1);
	
	assert!(tree.root.left.is_none());
	check_node(tree.root.right.as_ref().unwrap(), 1, [17.0, 15.0]);
}

// see examples https://www.geeksforgeeks.org/k-dimensional-tree/
#[test]
fn test_full_tree() {
	let mut tree = KdTree::new([3.0, 6.0]);

	tree.add([17.0, 15.0], 1);
	tree.add([13.0, 15.0], 2);
	tree.add([6.0, 12.0], 3);
	tree.add([9.0, 1.0], 4);
	tree.add([2.0, 7.0], 5);
	tree.add([10.0, 19.0], 6);

	// BFS left right
	let root = &tree.root;
	{
		let node = root.left.as_ref().unwrap();
		check_node(node, 5, [2.0, 7.0]);
	}
	{
		let node = root.right.as_ref().unwrap();
		check_node(node, 1, [17.0, 15.0]);
	}
	{
		let node = root.right.as_ref().unwrap()
						.left.as_ref().unwrap();
		check_node(node, 3, [6.0, 12.0]);
	}
	{
		let node = root.right.as_ref().unwrap()
					   .right.as_ref().unwrap();

		check_node(node, 2, [13.0, 15.0]);
	}
	{
		let node = root.right.as_ref().unwrap()
					   .left.as_ref().unwrap()
					   .right.as_ref().unwrap();
		check_node(node, 4, [9.0, 1.0]);
	}
	{
		let node = root.right.as_ref().unwrap()
					   .right.as_ref().unwrap()
					   .left.as_ref().unwrap();
		check_node(node, 6, [10.0, 19.0]);
	}
}

// query nearest neighbors
#[test]
fn test_nearest_neighbor() {
	let nodes = [
		[3.0, 6.0],
		[17.0, 15.0],
		[13.0, 15.0],
		[6.0, 12.0],
		[9.0, 1.0],
		[2.0, 7.0],
		[10.0, 19.0],
	];

	let centers = [
		[17.0, 15.0],
		[9.1, 1.0],
		[2.0, 8.0],
		[15.0, 13.0],
		[3.0, 5.0],
		[13.0, 7.0],
	];

	let mut tree = KdTree::new(nodes[0]);
	for (i,node) in enumerate(&nodes[1..]) {
		tree.add(*node, i+1);
	}

	for center in &centers {
		let mut node_dists = enumerate(&nodes)
			.map(|(id,node)| (id, norm2(&node, &center)))
			.collect::<Vec::<_>>();
		node_dists.sort_by(|(_,a),(_,b)| a.partial_cmp(b).unwrap());

		// test nearest_neighbor
		let node = tree.nearest_neighbor(*center);
		assert_eq!(node.id, node_dists[0].0);

		// test nearest_neighbors
		for radius in 1..10 {
			let radius = radius as f64;
			let mut nearest = node_dists.iter()
				.take_while(|(_,d)| *d <= radius)
				.map(|(n,_)| *n)
				.collect::<Vec<_>>();
			nearest.sort();
			let mut nearest_computed = tree.nearest_neighbors(*center, radius)
				.iter().map(|node| node.id).collect::<Vec<_>>();
			nearest_computed.sort();
			assert_eq!(nearest, nearest_computed);
		}
	}
}
}
