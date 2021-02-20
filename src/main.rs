#![allow(dead_code, unused_imports)]
#![feature(slice_group_by)]

pub mod common;
pub mod sample_space;
pub mod map_io;
pub mod nearest_neighbor;
pub mod rrt;
pub mod prm;
pub mod prm_graph;

use crate::prm::*;
use crate::prm_graph::*;
use crate::sample_space::*;
use crate::map_io::*;

fn main() {
    let mut m = Map::open("data/map2.pgm", [-1.0, -1.0], [1.0, 1.0]);
	m.add_zones("data/map2_zone_ids.pgm");

	fn goal(state: &[f64; 2]) -> bool {
		(state[0] - 0.0).abs() < 0.05 && (state[1] - 0.9).abs() < 0.05
	}

	let mut prm = PRM::new(ContinuousSampler::new([-1.0, -1.0], [1.0, 1.0]),
						   DiscreteSampler::new(),
						   &m);

	let result = prm.grow_graph(&[0.55, -0.8], goal, 0.05, 5.0, 2000, 100000);
	assert_eq!(result, Ok(()));
	let _ = prm.plan(&[0.0, -0.8], &vec![0.25, 0.25, 0.25, 0.25]).unwrap();

	prm.print_summary();
}