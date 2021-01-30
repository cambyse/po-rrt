use crate::{rrt::{RRTFuncs, RRTTree}};
use crate::{prm::{PRMFuncs, PRMGraph, PRMNode}};
use image::Luma;
use image::DynamicImage::ImageLuma8;
use core::f64;
use std::vec::Vec;
use std::collections::HashSet;

extern crate queues;
use queues::*;

#[derive(Debug, PartialEq)]
pub enum Belief {
	Always(bool),
	Choice(usize, f64),
}

#[derive(Clone)]
pub struct Map
{
	img: image::GrayImage,
	low: [f64; 2],
	//up: [f64; 2], // low + up are enough and make up redundant
	ppm: f64,
	zones: Option<image::GrayImage>,
	n_zones: usize,
	n_worlds: usize,
	zones_to_worlds: Vec<Vec<bool>>
}

// Given N zones, there are 2^N possible worlds
impl Map {
	pub fn open(filepath : &str, low: [f64; 2], up: [f64; 2]) -> Self {
		Self::build(Self::open_image(filepath), low, up)
	}

	pub fn save(&self, filepath: &str) {
		self.img.save(filepath).expect("Couldn't save image");
	}

	pub fn is_state_valid(&self, xy: &[f64; 2]) -> bool {
		let ij = self.to_pixel_coordinates(&*xy);
		let p = self.img.get_pixel(ij[1], ij[0]);

		p[0] == 255
	}

	pub fn is_state_valid_2(&self, xy: &[f64; 2]) -> Belief {
		let ij = self.to_pixel_coordinates(&*xy);
		let p = self.img.get_pixel(ij[1], ij[0]);

		match p[0] {
			255 => Belief::Always(true),
			0 => Belief::Always(false),
			p => Belief::Choice(self.get_zone_index(&ij).unwrap(), (p as f64) / 255.0)
		}
	}

	pub fn draw_path(&mut self, path: Vec<[f64;2]>) {
		for i in 1..path.len() {
			self.draw_line(path[i-1], path[i], 0);
		}
	}

	pub fn draw_tree(&mut self, rrttree: &RRTTree<2>) {
		for c in &rrttree.nodes {
			if let Some(parent_id) = c.parent_id {
				let parent = &rrttree.nodes[parent_id];
				self.draw_line(parent.state, c.state, 180);
			}
		}
	}

	fn to_pixel_coordinates(&self, xy: &[f64; 2]) -> [u32; 2] {
		let i: u32 = ((self.img.height() as f64) - (xy[1] - self.low[1]) * self.ppm) as u32;
		let j: u32 = ((xy[0] - self.low[0]) * self.ppm) as u32;

		[i, j]
	}

	fn get_zone_index(&self, ij: &[u32; 2]) -> Option<usize> {
		let p = self.zones.as_ref().expect("Zones missing").get_pixel(ij[1], ij[0]);
		match p[0] {
			255 => None,
			p => Some(p as usize),
		}
	}

	fn build(img: image::GrayImage, low: [f64; 2], up: [f64; 2])-> Map {
		let ppm = (img.width() as f64) / (up[0] - low[0]);

		Map{img, low, /*up,*/ ppm, zones: None, n_zones: 0, n_worlds: 0, zones_to_worlds: Vec::new()}
	}

	fn draw_line(&mut self, a: [f64; 2], b: [f64; 2], color: u8) {
		let a_ij = self.to_pixel_coordinates(&a);
		let b_ij = self.to_pixel_coordinates(&b);

		// TODO: simplify notations for casts

		let di = (b_ij[0] as i32 - a_ij[0] as i32).abs();
		let dj = (b_ij[1] as i32- a_ij[1] as i32).abs();
		
		let n = if di > dj { di } else { dj };
		for s in 0..n {
			let lambda = (s as f64) / (n as f64);
			let i = (a_ij[0] as i32 + (lambda * ((b_ij[0] as f64) - (a_ij[0] as f64))) as i32) as u32;
			let j = (a_ij[1] as i32+ (lambda * ((b_ij[1] as f64) - (a_ij[1] as f64))) as i32) as u32;
			
			self.img.put_pixel(j, i, Luma([color]));
		}
	}

	fn open_image(filepath : &str) -> image::GrayImage {
		let img = image::open(filepath).expect(&format!("Impossible to open image: {}", filepath));

		match img {
			ImageLuma8(gray_img) => gray_img,
			_ => panic!("Wrong image format!"),
		}
	}

	// zones specific
	pub fn add_zones(&mut self, filepath : &str) {
		// image
		self.zones = Some(Self::open_image(filepath));

		// number of zones
		let mut max_id = 0;
		for i in 0..self.zones.as_ref().unwrap().height() {
			for j in 0..self.zones.as_ref().unwrap().width() {
				let z = self.get_zone_index(&[i, j]);
				match z {
					Some(id) => {
						if id > max_id {
							max_id = id;
						}
					},
					None => {}
				}	
			}
		}
		
		self.n_zones = max_id + 1;
		self.n_worlds = (2 as u32).pow(self.n_zones as u32) as usize;

		// zone -> worlds
		for i in 0..self.n_zones {
			self.zones_to_worlds.push(self.zone_index_to_validity(i));
		}
	}

	pub fn n_zones(&self) -> usize {
		self.n_zones
	}

	pub fn n_worlds(&self) -> usize {
		self.n_worlds
	}

	fn zone_index_to_validity(&self, zone_index: usize) -> Vec<bool> {
		let mut validity = vec![true; self.n_worlds()];
		for world in 0..self.n_worlds() {
			if !self.get_zone_status(world, zone_index).expect("Call with a correct world id") {
				validity[world] = false
			}
		}

		validity
	}

	fn get_zone_status(&self, world: usize, zone_index: usize) -> Result<bool, ()> {
		if zone_index < self.n_zones() && world < self.n_worlds() {
			Ok(world & (1 << zone_index) != 0)
		} else {
			Err(())
		}
	}

	pub fn draw_full_graph(&mut self, graph: &PRMGraph<2>) {
		for from in &graph.nodes {
			for to_id in from.children.clone() {
				let to  = &graph.nodes[to_id];
				self.draw_line(from.state, to.state, 100);
			}
		}
	}

	pub fn draw_graph_for_world(&mut self, graph: &PRMGraph<2>, world:usize) {
		if world > self.n_worlds() {
			panic!("Invalid world id");
		}

		let mut visited = HashSet::new();
		let mut queue: Queue<usize> = queue![];
		visited.insert(0);
		queue.add(0).expect("Overflow!");

		while queue.size() > 0 {
			let from_id = queue.remove().unwrap();
			let from = &graph.nodes[from_id];

			for &to_id in &from.children {
				let to = &graph.nodes[to_id];

				if to.validity[world] {
					self.draw_line(from.state, to.state, 100);

					if !visited.contains(&to_id) {
						queue.add(to_id).expect("Overflow");
						visited.insert(to_id);
					}
				}
			}
		}
	}

	pub fn set_world(&mut self, world_id:usize) {
		for i in 0..self.zones.as_ref().unwrap().height() {
			for j in 0..self.zones.as_ref().unwrap().width() {
				let z = self.get_zone_index(&[i, j]);

				match z {
					Some(zone_id) => {
							let color = if self.zones_to_worlds[zone_id][world_id] { 255 } else { 0 };
							self.img.put_pixel(j, i, Luma([color]));
					},
					None => {}
				}
			}
		}
	}
} 

impl RRTFuncs<2> for Map {
	fn state_validator(&self, state: &[f64; 2]) -> bool {
		self.is_state_valid(state)
	}
}

impl PRMFuncs<2> for Map {
	fn state_validity(&self, state: &[f64; 2]) -> Option<Vec<bool>> {
		match self.is_state_valid_2(state) {
			Belief::Choice(zone_index, _) => {Some(self.zones_to_worlds[zone_index].clone())}, // TODO: improve readability
			Belief::Always(true) => {Some(vec![true; self.n_worlds()])},
			Belief::Always(false) => None
		}
	}
}

#[cfg(test)]
mod tests {

use super::*;
use std::fs;
use std::path::Path;

#[test]
fn open_image() {
	Map::open("data/map0.pgm", [-1.0, -1.0], [1.0, 1.0]);
}

#[test]
fn test_valid_state() {
	let m = Map::open("data/map0.pgm", [-1.0, -1.0], [1.0, 1.0]);
	assert!(m.is_state_valid(&[0.0, 0.0]));
}

#[test]
fn test_invalid_state() {
	let m = Map::open("data/map0.pgm", [-1.0, -1.0], [1.0, 1.0]);
	assert!(!m.is_state_valid(&[0.0, 0.6]));
}

#[test]
fn clone_draw_line_and_save_image() {
	let m = Map::open("data/map0.pgm", [-1.0, -1.0], [1.0, 1.0]);
	let mut m = m.clone();

	m.draw_line([-0.5, -0.5], [0.5, 0.5], 128);
	m.draw_path([[-0.3, -0.4], [0.0, 0.0] ,[0.4, 0.3]].to_vec());

	m.save("results/tmp.pgm");

	assert!(Path::new("results/tmp.pgm").exists());
	fs::remove_file("results/tmp.pgm").unwrap();
}
// MAP 1 zone
#[test]
fn test_map_1_construction() {
	let mut map = Map::open("data/map1.pgm", [-1.0, -1.0], [1.0, 1.0]);
	map.add_zones("data/map1_zone_ids.pgm");

	assert_eq!(map.n_zones(), 1);
	assert_eq!(map.n_worlds(), 2);
}

#[test]
fn test_map_1_states() {
	let mut map = Map::open("data/map1.pgm", [-1.0, -1.0], [1.0, 1.0]);
	map.add_zones("data/map1_zone_ids.pgm");

	// zone status
	assert_eq!(map.get_zone_status(0, 0).unwrap(), false);// world 0 corresponds to all variations invalid
	assert_eq!(map.get_zone_status(1, 0).unwrap(), true); // world 1 corresponds to door with index 2^0 open

	assert_eq!(map.get_zone_status(1, 1), Err(())); // zone id too high
	assert_eq!(map.get_zone_status(2, 0), Err(())); // world number too high

	// world validities
	assert_eq!(map.state_validity(&[0.1, 0.12]), None); // obstacle
	assert_eq!(map.state_validity(&[0.0, 0.0]).unwrap(), vec![true, true]); // free space
	assert_eq!(map.state_validity(&[0.57, 0.11]).unwrap(), vec![false, true]); // on door
}

// MAP 2 zones
#[test]
fn test_map_2_construction() {
	let mut map = Map::open("data/map2.pgm", [-1.0, -1.0], [1.0, 1.0]);
	map.add_zones("data/map2_zone_ids.pgm");

	assert_eq!(map.n_zones(),2);
	assert_eq!(map.n_worlds(), 4);
}

#[test]
fn test_map_2_states() {
	let mut map = Map::open("data/map2.pgm", [-1.0, -1.0], [1.0, 1.0]);
	map.add_zones("data/map2_zone_ids.pgm");

	// world validity
	assert_eq!(map.state_validity(&[0.1, 0.12]), None); // obstacle
	assert_eq!(map.state_validity(&[0.0, 0.0]).unwrap(), vec![true, true, true, true]); // free space

	assert_eq!(map.state_validity(&[0.51, -0.41]).unwrap(), vec![false, true, false, true]); // zone 0
	assert_eq!(map.state_validity(&[0.57, 0.09]).unwrap(), vec![false, false, true, true]); // zone 1
}

// MAP 4 zones
#[test]
fn test_map_4_construction() {
	let mut map = Map::open("data/map4.pgm", [-1.0, -1.0], [1.0, 1.0]);
	map.add_zones("data/map4_zone_ids.pgm");

	assert_eq!(map.n_zones(), 4);
	assert_eq!(map.n_worlds(), 16);
}

#[test]
fn test_map_4_states() {
	let mut map = Map::open("data/map4.pgm", [-1.0, -1.0], [1.0, 1.0]);
	map.add_zones("data/map4_zone_ids.pgm");

	// door validity
	assert_eq!(map.is_state_valid_2(&[0.0, 0.0]), Belief::Always(true));
	assert_eq!(map.is_state_valid_2(&[0.0, -0.24]), Belief::Always(false));
	assert_eq!(map.is_state_valid_2(&[-0.67, -0.26]), Belief::Choice(0, 128.0/255.0));

	// world validities
	assert_eq!(map.state_validity(&[-0.29, -0.23]), None);
	assert_eq!(map.state_validity(&[0.0, 0.0]).unwrap(), vec![true; 16]);
}
}