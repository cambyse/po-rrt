use crate::{rrt::{Reachable, RRTFuncs, RRTTree}};
use crate::{prm_graph::{PRMGraph, PRMNode, PRMFuncs}};
use crate::prm_policy_refiner::*;
use crate::common::*;
use image::{DynamicImage, GenericImageView, Luma};
use image::DynamicImage::ImageLuma8;
use core::{f64, panic};
use std::vec::Vec;
use std::collections::HashSet;
extern crate queues;
use queues::*;
use bitvec::prelude::*;
use image::Pixel;
use image::Rgb;

pub mod colors {
	use image::Rgb;

	macro_rules! define_colors {
		{$($name:ident: $values: expr;)*} => {
			$(pub const $name: Rgb<u8> = Rgb($values);)*
		}
	}

	define_colors! {
		BLACK:     [0,0,0];
		WHITE:     [255,255,255];
		RED:       [255,0,0];
		LIME:      [0,255,0];
		BLUE:      [0,0,255];
		YELLOW:    [255,255,0];
		CYAN:      [0,255,255];
		MAGENTA:   [255,0,255];
		MAROON:    [128,0,0];
		OLIVE:     [128,128,0];
		GREEN:     [0,128,0];
		PURPLE:    [128,0,128];
		TEAL:      [0,128,128];
		NAVY:      [0,0,128];
		GRAY1:     [30,30,30];
		GRAY2:     [60,60,60];
		GRAY3:     [90,90,90];
		GRAY4:     [120,120,120];
		GRAY5:     [150,150,150];
		GRAY6:     [180,180,180];
		GRAY7:     [210,210,210];
		GRAY8:     [240,240,240];
	}

	pub const COLOR_MAP: &[Rgb<u8>] = &[RED, OLIVE, BLUE, MAGENTA, LIME, NAVY];
	pub fn color_map(index: usize) -> Rgb<u8> {
		COLOR_MAP[index % COLOR_MAP.len()]
	}
}

use colors::*;

#[derive(Debug, PartialEq)]
pub enum Belief {
	Free,
	Obstacle,
	Zone(usize),
}

#[derive(Clone)]
pub struct Map {
	img: image::RgbImage,
	low: [f64; 2],
	//up: [f64; 2], // low + up are enough and make up redundant
	ppm: f64,
	zones: Option<image::GrayImage>,
	n_zones: usize,
	n_worlds: usize,
	zones_to_worlds: Vec<WorldMask>,
	world_validities: Vec<WorldMask>,
	zone_positions: Vec<[f64;2]>,
	visibility_distance: f64
}

// Given N zones, there are 2^N possible worlds
impl Map {
	pub fn open(filepath : &str, low: [f64; 2], up: [f64; 2]) -> Self {
		Self::build(Self::open_image(filepath), low, up)
	}

	pub fn save(&self, filepath: &str) {
		self.img.save(format!("{}.png", filepath)).expect("Couldn't save image");
	}

	fn build(img: image::GrayImage, low: [f64; 2], up: [f64; 2])-> Map {
		let ppm = (img.width() as f64) / (up[0] - low[0]);

		let img = DynamicImage::ImageLuma8(img).to_rgb8();

		Map{img, low, /*up,*/ ppm, zones: None, n_zones: 0, n_worlds: 0, zones_to_worlds: Vec::new(), world_validities: Vec::new(), zone_positions: Vec::new(), visibility_distance: 0.0}
	}

	fn open_image(filepath : &str) -> image::GrayImage {
		let img = image::open(filepath).unwrap_or_else(|_| panic!("Impossible to open image: {}", filepath));

		match img {
			ImageLuma8(gray_img) => gray_img,
			_ => panic!("Wrong image format!"),
		}
	}

	// zones specific
	pub fn init_without_zones(&mut self) {
		self.n_worlds = 1;
		self.world_validities.push(bitvec![1; self.n_worlds]);
	}

	pub fn add_zones(&mut self, filepath : &str, visibility_distance: f64) {
		self.zones = Some(Self::open_image(filepath));

		self.init_zone_ids();
		self.init_zone_positions();
		self.visibility_distance = visibility_distance;

		// zone -> worlds
		for i in 0..self.n_zones {
			self.zones_to_worlds.push(self.zone_index_to_world_mask(i));
		}

		self.world_validities = self.zones_to_worlds.clone();
		self.world_validities.push(bitvec![1; self.n_worlds]);
		//println!("{:?}:", &self.zones_to_worlds);
	}

	fn init_zone_ids(&mut self) {
		let mut max_id = 0;
		for i in 0..self.zones.as_ref().unwrap().height() {
			for j in 0..self.zones.as_ref().unwrap().width() {
				let z = self.get_zone_index(i, j);
				if let Some(id) = z {
					if id > max_id {
						max_id = id;
					}
				} 	
			}
		}
		
		self.n_zones = max_id + 1;
		self.n_worlds = (2_u32).pow(self.n_zones as u32) as usize;
	}

    fn init_zone_positions(&mut self) {
		let mut zone_to_pixels: Vec<Vec<[u32; 2]>> = vec![Vec::new(); self.n_zones];
		for i in 0..self.zones.as_ref().unwrap().height() {
			for j in 0..self.zones.as_ref().unwrap().width() {
				let z = self.get_zone_index(i, j);
				if let Some(id) = z {
					zone_to_pixels[id].push([i, j]);
				}  	
			}
		}
		
		for pixel_coords in &zone_to_pixels {
			let sum = pixel_coords.iter().fold( [0, 0], | [sum_i, sum_j], [i, j] | [sum_i+i, sum_j+j] );
			let ij = [sum[0] / pixel_coords.len() as u32, sum[1] / pixel_coords.len() as u32];
			self.zone_positions.push(self.to_coordinates(&ij))
		}
	}

	pub fn is_state_valid(&self, xy: &[f64; 2]) -> Belief {
		let ij = self.to_pixel_coordinates(&*xy);
		let p = self.img.get_pixel(ij[1], ij[0]);

		match p[0] {
			255 => Belief::Free,
			0 => Belief::Obstacle,
			_ => Belief::Zone(self.get_zone_index(ij[0], ij[1]).unwrap())
		}
	}

	fn to_pixel_coordinates(&self, xy: &[f64; 2]) -> [u32; 2] {
		let i: u32 = (((self.img.height() - 1) as f64) - (xy[1] - self.low[1]) * self.ppm) as u32;
		let j: u32 = ((xy[0] - self.low[0]) * self.ppm) as u32;

		[i, j]
	}

	fn to_coordinates(&self, ij: &[u32; 2]) -> [f64; 2] {
		let x: f64 = ij[1] as f64 / self.ppm + self.low[1];
		let y: f64 = (self.img.height() - 1 - ij[0]) as f64 / self.ppm + self.low[0];

		[x, y]
	}

	fn get_zone_index(&self, i: u32, j: u32) -> Option<usize> {
		let p = self.zones.as_ref().expect("Zones missing").get_pixel(j, i);
		match p[0] {
			255 => None,
			p => Some(p as usize),
		}
	}

	fn zone_index_to_world_mask(&self, zone_index: usize) -> WorldMask {
		let mut world_mask = bitvec![1; self.n_worlds];
		for world in 0..self.n_worlds {
			if !self.get_zone_status(world, zone_index).expect("Call with a correct world id") {
				world_mask.set(world, false);
			}
		}
		world_mask
	}

	fn get_zone_status(&self, world: usize, zone_index: usize) -> Result<bool, ()> {
		if zone_index < self.n_zones && world < self.n_worlds {
			Ok(world & (1 << zone_index) != 0)
		} else {
			Err(())
		}
	}

	fn get_traversed_space(&self, a: &[f64; 2], b: &[f64; 2]) -> Belief {
		let mut traversed_space = Belief::Free;

		let a_ij = self.to_pixel_coordinates(a);
		let b_ij = self.to_pixel_coordinates(b);

		let a = (a_ij[0] as i32, a_ij[1] as i32);
		let b = (b_ij[0] as i32, b_ij[1] as i32);

		for (i, j) in line_drawing::Bresenham::new(a, b) {
			let pixel = self.img.get_pixel(j as u32, i as u32);
			match pixel[0] {
				255 => {},
				0 => return Belief::Obstacle,
				_ => {	
					let zone_index = self.get_zone_index(i as u32, j as u32).unwrap();
					if let Belief::Zone(previous) = traversed_space {
						assert!(zone_index == previous, "multiple zone traversal not supported");
					}				
					traversed_space = Belief::Zone(zone_index);
				}
			}
		}

		traversed_space
	}

	#[allow(clippy::style)]
	fn get_successor_belief_states(&self, belief_state: &BeliefState, zone_id: usize) -> Vec<Vec<f64>> {
		let mut output_beliefs: Vec<Vec<f64>> = Vec::new();

		fn normalize(belief: &mut Vec<f64>) {
			let sum = belief.iter().fold(0.0, |sum, p| sum + p );

			for p in belief {
				*p /= sum;
			}
		}

		let mask = &self.zones_to_worlds[zone_id];

		// assume closed
		let mut belief_close = belief_state.to_owned();
		for w in 0..mask.len() {
			belief_close[w] = if mask[w] { 0.0 } else { belief_state[w] };
		}
		normalize(&mut belief_close);
		if ! belief_close.iter().any(|&p| p.is_nan()) {
			output_beliefs.push(belief_close);
		}

		// assume open
		let mut belief_open = belief_state.to_owned();
		for w in 0..mask.len() {
			belief_open[w] = if mask[w] { belief_state[w] } else { 0.0 };
		}
		normalize(&mut belief_open);
		if ! belief_open.iter().any(|&p| p.is_nan()) {
			output_beliefs.push(belief_open);
		}

		output_beliefs
	}

	#[allow(clippy::style)]
	pub fn observe_impl(&self, state: &[f64; 2], belief_state: &BeliefState) -> Vec<BeliefState> {
		let mut output_beliefs: Vec<BeliefState> = Vec::new();
		output_beliefs.push(belief_state.clone());

		for zone_id in 0..self.n_zones {
			if  norm2(state, &self.zone_positions[zone_id]) < self.visibility_distance {
				let fov_feasability = self.get_traversed_space(&state, &self.zone_positions[zone_id]) != Belief::Obstacle;

				if fov_feasability {
					let beliefs = output_beliefs.clone();
					output_beliefs.clear();

					for belief in beliefs {
						output_beliefs.extend(self.get_successor_belief_states(&belief, zone_id));
					}
				}
			}
		}
		output_beliefs
	}

	// drawing functions
	pub fn resize(&mut self, factor: u32) {
		let w = self.img.width() * factor;
		let h = self.img.height() * factor;

		self.img = image::imageops::resize(&self.img, w, h, image::imageops::FilterType::Nearest);
		self.ppm *= factor as f64;

		if let Some(zone_img) = &self.zones {
			self.zones = Some(image::imageops::resize(zone_img, w, h, image::imageops::FilterType::Nearest));
		}
	}

	pub fn draw_path(&mut self, path: &[[f64;2]], color: Rgb<u8>) {
		for i in 1..path.len() {
			self.draw_line(path[i-1], path[i], color, 1.0);
		}
	}

	pub fn draw_hit_zone(&mut self, hit_zone: impl Fn(&[f64; 2]) -> bool) {
		const N: usize = 100;
		for i in 0..N {
			for j in 0..N {
				let f = (i as f64) / (N as f64);
				let i = self.low[0] * (1.0-2.0*f);

				let f = (j as f64) / (N as f64);
				let j = self.low[1] * (1.0-2.0*f);

				let p = [i,j];

				if hit_zone(&p) {
					self.draw_circle(&p, 0.025, RED);
				}
			}
		}
	}

	pub fn draw_tree(&mut self, rrttree: &RRTTree<2>, belief_id: Option<usize>) {
		for c in &rrttree.nodes {
			if let Some(ref parent_link) = c.parent {
				let parent = &rrttree.nodes[parent_link.id];
							
				if let Some(belief_id) = belief_id {
					// roots
					if c.belief_state_id == belief_id && c.belief_state_id != parent.belief_state_id {
						self.draw_circle(&c.state, 0.01, PURPLE);
					}

					// observations
					if parent.belief_state_id == belief_id && c.belief_state_id != parent.belief_state_id {
						self.draw_circle(&c.state, 0.025, NAVY);
					}

					if c.belief_state_id != belief_id || parent.belief_state_id != belief_id {
						continue;
					}
				}

				let color = color_map(c.belief_state_id);
				self.draw_line(parent.state, c.state, color, 0.3);
			}
		}
	}

	pub fn draw_policy(&mut self, policy: &Policy<2>) {
		for parent in &policy.nodes {
			if parent.children.len() > 1 {
				self.draw_circle(&parent.state, 0.025, NAVY);

				/*
				println!("parent belief:{:?}", &parent.belief_state);
				for &child_id in &parent.children {
					let child = &policy.nodes[child_id];
					println("child belief:{:?}", &child.belief_state);
				}
				*/
			}
			for &child_id in &parent.children {
				let child = &policy.nodes[child_id];
				self.draw_line(parent.state, child.state, BLACK, 1.0);
			}
		}
	}

	pub fn draw_refinment_trees(&mut self, refinment_trees: &Vec<RefinmentTree<2>>) {
		for tree in refinment_trees.iter().skip(0) {
			self.draw_refinment_tree(tree);
			//break;
		}
	}

	pub fn draw_refinment_tree(&mut self, refinment_tree: &RefinmentTree<2>) {
		for to in &refinment_tree.nodes {
			match to.parent {
				Some(parent) => {
					let from = &refinment_tree.nodes[parent.id];
					self.draw_line(from.state, to.state, YELLOW, 1.0);
				},
				_ => {}
			}
		}
	}

	fn draw_line(&mut self, a: [f64; 2], b: [f64; 2], color: Rgb<u8>, alpha: f32) {
		let a_ij = self.to_pixel_coordinates(&a);
		let b_ij = self.to_pixel_coordinates(&b);

		let a = (a_ij[0] as f32, a_ij[1] as f32);
		let b = (b_ij[0] as f32, b_ij[1] as f32);

		for ((i, j), line_alpha) in line_drawing::XiaolinWu::<f32, i32>::new(a, b) {
			if 0 <= i && i < self.img.height() as i32 && 0 <= j && j < self.img.width() as i32 {
				let pixel = self.img.get_pixel_mut(j as u32, i as u32);
				pixel.apply2(&color, |old, new| {
					let alpha = alpha * line_alpha;
					((1.0 - alpha) * (old as f32) + alpha * (new as f32)) as u8
				});
			}
		}
	}

	fn draw_circle(&mut self, xy: &[f64; 2], radius: f64, color: Rgb<u8>) {
		for angle_step in 1..51 {
			let angle_from = (angle_step - 1) as f64 * 2.0 * std::f64::consts::PI / 50.0;
			let angle_to = (angle_step) as f64 * 2.0 * std::f64::consts::PI / 50.0;

			let xy_from = [xy[0] + radius * angle_from.cos(), xy[1] + radius * angle_from.sin()];
			let xy_to = [xy[0] + radius * angle_to.cos(), xy[1] + radius * angle_to.sin()];

			self.draw_line(xy_from, xy_to, color, 1.0);
		}
	}
	
	pub fn draw_full_graph(&mut self, graph: &PRMGraph<2>) {
		for from in &graph.nodes {
			for to_edge in from.children.clone() {
				let to  = &graph.nodes[to_edge.id];
				self.draw_line(from.state, to.state, GRAY5, 0.15);
			}
		}
	}

	pub fn draw_graph_from_root(&mut self, graph: &PRMGraph<2>) {
		self.draw_graph_from_root_impl(graph, &|_|{true})
	}

	pub fn draw_graph_for_world(&mut self, graph: &PRMGraph<2>, world:usize) {
		if world > self.n_worlds {
			panic!("Invalid world id");
		}

		self.draw_graph_from_root_impl(graph, &|node|{graph.validities[node.validity_id][world]})
	}

	pub fn draw_graph_from_root_impl(&mut self, graph: &PRMGraph<2>, validator: &dyn Fn(&PRMNode<2>) -> bool) {
		let mut visited = HashSet::new();
		let mut queue: Queue<usize> = queue![];
		visited.insert(0);
		queue.add(0).expect("Overflow!");

		while queue.size() > 0 {
			let from_id = queue.remove().unwrap();
			let from = &graph.nodes[from_id];

			for to_edge in &from.children {
				let to = &graph.nodes[to_edge.id];

				if validator(to) {
					self.draw_line(from.state, to.state, GRAY7, 0.5);

					if !visited.contains(&to_edge.id) {
						queue.add(to_edge.id).expect("Overflow");
						visited.insert(to_edge.id);
					}
				}
			}
		}
	}

	pub fn draw_world(&mut self, world_id:usize) {
		for i in 0..self.zones.as_ref().unwrap().height() {
			for j in 0..self.zones.as_ref().unwrap().width() {
				let z = self.get_zone_index(i, j);

				if let Some(zone_id) = z {
					let color = if self.zones_to_worlds[zone_id][world_id] { WHITE } else { BLACK };
					self.img.put_pixel(j, i, color);
				}	
			}
		}
	}

	pub fn draw_zones_observability(&mut self) {
		for xy in &self.zone_positions.clone() {
			self.draw_circle(xy, self.visibility_distance, TEAL);
		}
	}
} 

impl RRTFuncs<2> for Map {
	fn state_validator(&self, state: &[f64; 2]) -> Reachable {
		match self.is_state_valid(state) {
			Belief::Free => Reachable::Always,
			Belief::Obstacle => Reachable::Never,
			Belief::Zone(zone) => Reachable::Restricted(&self.zones_to_worlds[zone]),
		}
	}

	fn transition_validator(&self, a: &[f64; 2], b: &[f64; 2]) -> Reachable {
		match self.get_traversed_space(a, b) {
			Belief::Free => Reachable::Always,
			Belief::Obstacle => Reachable::Never,
			Belief::Zone(zone) => Reachable::Restricted(&self.zones_to_worlds[zone]),
		}
	}

	fn observe_new_beliefs(&self, state: &[f64; 2], belief_state: &BeliefState) -> Vec<BeliefState> {
		let mut output_beliefs = self.observe_impl(state, belief_state);

		if output_beliefs.len() > 1 && output_beliefs[0] == *belief_state {
			output_beliefs.remove(0);
		}

		output_beliefs
	}
}

impl PRMFuncs<2> for Map {
	fn n_worlds(&self) -> usize {
		self.n_worlds
	}

	fn state_validity(&self, state: &[f64; 2]) -> Option<usize> {
		match self.is_state_valid(state) {
			Belief::Zone(zone_index) => {Some(zone_index)}, // TODO: improve readability
			Belief::Free => {Some(self.world_validities.len() - 1)},
			Belief::Obstacle => None
		}
	}

	fn transition_validator(&self, from: &PRMNode<2>, to: &PRMNode<2>) -> Option<usize> {
		// needed -> TODO: compare with and without and benchmark
		/*
		let symbolic_validity = from.validity.iter().zip(&to.validity)
		.any(|(a, b)| *a && *b);

		if !symbolic_validity {
			return None
		}*/
		//
		
		let geometric_validitiy = self.get_traversed_space(&from.state, &to.state);

		match geometric_validitiy {
			Belief::Zone(zone_index) => {Some(zone_index)}, // TODO: improve readability
			Belief::Free => {Some(self.world_validities.len() - 1)},
			Belief::Obstacle => None
		}
	}

	fn reachable_belief_states(&self, belief_state: &BeliefState) -> Vec<BeliefState> {
		let mut reachable_beliefs: Vec<BeliefState> = Vec::new();
		let mut reachable_beliefs_hashes = HashSet::new();
		let mut lifo: Vec<(BeliefState, Vec<usize>)> = Vec::new(); // bs, doors to check

		reachable_beliefs.push(belief_state.clone());
		lifo.push((belief_state.clone(), (0..self.n_zones).collect()));

		
		while !lifo.is_empty() {
			let (belief, zones_to_check) = lifo.pop().unwrap();

			for &zone_id in &zones_to_check {
				let zones = zones_to_check.clone(); // TODO: improve here
				let remaining_zones: Vec<usize> = zones.into_iter().filter(|id| *id != zone_id).collect();

				let successors = self.get_successor_belief_states(&belief, zone_id);

				for successor in &successors {
					if !reachable_beliefs.contains(successor) {
						if!reachable_beliefs_hashes.contains(&hash(successor)) {
							reachable_beliefs.push(successor.clone());
							reachable_beliefs_hashes.insert(hash(successor));
						}
						lifo.push((successor.clone(), remaining_zones.clone()));
					}
				}
			}
		}

		reachable_beliefs
	}

	fn world_validities(&self) -> Vec<WorldMask> {
		self.world_validities.clone()
	}

	fn observe(&self, state: &[f64; 2], belief_state: &BeliefState) -> Vec<BeliefState> {
		self.observe_impl(state, belief_state)
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
fn coordinate_conversion() {
	let m = Map::open("data/map0.pgm", [-1.0, -1.0], [1.0, 1.0]);

	let xys = [[0.5, 0.3], [-0.5, 0.1], [-0.1, -0.4]];

	for xy in &xys {
		let ij = m.to_pixel_coordinates(&xy);
		let xy_after_roundtrip = m.to_coordinates(&ij);
		
		let d = norm2(&xy, &xy_after_roundtrip);
		let dmax = (2.0 as f64).sqrt() * 2.0 / 200.0;
	
		assert!(d <= dmax);
	}
}

#[test]
fn test_valid_state() {
	let m = Map::open("data/map0.pgm", [-1.0, -1.0], [1.0, 1.0]);
	assert_eq!(m.is_state_valid(&[0.0, 0.0]), Belief::Free);
}

#[test]
fn test_invalid_state() {
	let m = Map::open("data/map0.pgm", [-1.0, -1.0], [1.0, 1.0]);
	assert_eq!(m.is_state_valid(&[0.0, 0.6]), Belief::Obstacle);
}

#[test]
fn clone_draw_line_and_save_image() {
	let m = Map::open("data/map0.pgm", [-1.0, -1.0], [1.0, 1.0]);
	let mut m = m.clone();

	m.draw_line([-0.5, -0.5], [0.5, 0.5], GRAY5, 1.0);
	m.draw_path(&[[-0.3, -0.4], [0.0, 0.0] ,[0.4, 0.3]].to_vec(), GRAY5);

	m.save("results/tmp");

	assert!(Path::new("results/tmp.png").exists());
	fs::remove_file("results/tmp.png").unwrap();
}
// MAP 1 zone
#[test]
fn test_map_1_construction() {
	let mut map = Map::open("data/map1.pgm", [-1.0, -1.0], [1.0, 1.0]);
	map.add_zones("data/map1_zone_ids.pgm", 0.1);

	assert_eq!(map.n_zones, 1);
	assert_eq!(map.n_worlds, 2);
}

#[test]
fn test_map_1_states() {
	let mut map = Map::open("data/map1.pgm", [-1.0, -1.0], [1.0, 1.0]);
	map.add_zones("data/map1_zone_ids.pgm", 0.1);

	let world_validities = map.world_validities();

	// zone status
	assert_eq!(map.get_zone_status(0, 0).unwrap(), false);// world 0 corresponds to all variations invalid
	assert_eq!(map.get_zone_status(1, 0).unwrap(), true); // world 1 corresponds to door with index 2^0 open

	assert_eq!(map.get_zone_status(1, 1), Err(())); // zone id too high
	assert_eq!(map.get_zone_status(2, 0), Err(())); // world number too high

	// world validities
	assert_eq!(map.state_validity(&[0.1, 0.12]), None); // obstacle
	assert_eq!(world_validities[map.state_validity(&[0.0, 0.0]).unwrap()], bitvec![1, 1]); // free space
	assert_eq!(world_validities[map.state_validity(&[0.57, 0.11]).unwrap()], bitvec![0, 1]); // on door
}

// MAP 2 zones
#[test]
fn test_map_2_construction() {
	let mut map = Map::open("data/map2.pgm", [-1.0, -1.0], [1.0, 1.0]);
	map.add_zones("data/map2_zone_ids.pgm", 0.1);

	assert_eq!(map.n_zones,2);
	assert_eq!(map.n_worlds, 4);
}

#[test]
fn test_map_2_states() {
	let mut map = Map::open("data/map2.pgm", [-1.0, -1.0], [1.0, 1.0]);
	map.add_zones("data/map2_zone_ids.pgm", 0.1);
	let world_validities = map.world_validities();

	// world validity
	assert_eq!(map.state_validity(&[0.1, 0.12]), None); // obstacle
	assert_eq!(world_validities[map.state_validity(&[0.0, 0.0]).unwrap()], bitvec![1,1,1,1]); // free space

	assert_eq!(world_validities[map.state_validity(&[0.51, -0.41]).unwrap()], bitvec![0,1,0,1]); // zone 0
	assert_eq!(world_validities[map.state_validity(&[0.57, 0.09]).unwrap()], bitvec![0,0,1,1]); // zone 1
}

#[test]
fn test_map_2_observation_model_in_zones() {
	let mut map = Map::open("data/map2.pgm", [-1.0, -1.0], [1.0, 1.0]);
	map.add_zones("data/map2_zone_ids.pgm", 0.1);

	// regular belief state transitions
	let posteriors = map.observe(&[0.54, -0.5], &vec![0.25; 4]);
	assert_eq!(posteriors[0], vec![0.5, 0.0, 0.5, 0.0]); // zone 0 close
	assert_eq!(posteriors[1], vec![0.0, 0.5, 0.0, 0.5]); // zone 0 open

	// peculiar belief state transitions
	let posteriors = map.observe(&[0.54, -0.5], &vec![1.0, 0.0, 0.0, 0.0]); // already in open belief state!
	assert_eq!(posteriors, vec![vec![1.0, 0.0, 0.0, 0.0]]);

	let posteriors = map.observe(&[0.54, -0.5], &vec![0.0, 1.0, 0.0, 0.0]); // already in close belief state!
	assert_eq!(posteriors, vec![vec![0.0, 1.0, 0.0, 0.0]]);

	let posteriors = map.observe(&[0.54, -0.5], &vec![0.5, 0.5, 0.0, 0.0]);
	assert_eq!(posteriors[0], vec![1.0, 0.0, 0.0, 0.0]); // zone 0 close
	assert_eq!(posteriors[1], vec![0.0, 1.0, 0.0, 0.0]); // zone 0 open
	assert_eq!(posteriors.len(), 2); // zone 0 open
}

#[test]
fn test_map_2_observation_model_outside_zones() {
	let mut map = Map::open("data/map2.pgm", [-1.0, -1.0], [1.0, 1.0]);
	map.add_zones("data/map2_zone_ids.pgm", 0.1);

	let posteriors = map.observe(&[-0.3, -0.5], &vec![0.25; 4]);
	assert_eq!(posteriors, vec![vec![0.25; 4]]);
}

#[test]
fn test_map_2_observation_model_when_2_zones_seen_at_the_same_time() {
	let mut map = Map::open("data/map2_fov.pgm", [-1.0, -1.0], [1.0, 1.0]);
	map.add_zones("data/map2_fov_zone_ids.pgm", 2.0);

	let posteriors = map.observe(&[0.395, -0.245], &vec![0.25; 4]);
	assert_eq!(posteriors, vec![vec![1.0, 0.0, 0.0, 0.0], vec![0.0, 0.0, 1.0, 0.0], vec![0.0, 1.0, 0.0, 0.0], vec![0.0, 0.0, 0.0, 1.0]]);
}

#[test]
fn test_map_2_reachable_beliefs() {
	let mut map = Map::open("data/map2.pgm", [-1.0, -1.0], [1.0, 1.0]);
	map.add_zones("data/map2_zone_ids.pgm", 0.1);

	let reachable_beliefs = map.reachable_belief_states(&vec![0.25; 4]);
	assert_eq!(reachable_beliefs.len(), 9); 
	assert!(reachable_beliefs.contains(&vec![0.25; 4]));
	assert!(reachable_beliefs.contains(&vec![0.5, 0.5, 0.0, 0.0]));
	assert!(reachable_beliefs.contains(&vec![0.5, 0.0, 0.5, 0.0]));
	assert!(reachable_beliefs.contains(&vec![0.0, 0.5, 0.0, 0.5]));
	assert!(reachable_beliefs.contains(&vec![0.0, 0.0, 0.5, 0.5]));
	assert!(reachable_beliefs.contains(&vec![1.0, 0.0, 0.0, 0.0]));
	assert!(reachable_beliefs.contains(&vec![0.0, 1.0, 0.0, 0.0]));
	assert!(reachable_beliefs.contains(&vec![0.0, 0.0, 1.0, 0.0]));
	assert!(reachable_beliefs.contains(&vec![0.0, 0.0, 0.0, 1.0]));
}

#[test]
fn test_map_2_world_validities() {
	let mut map = Map::open("data/map2.pgm", [-1.0, -1.0], [1.0, 1.0]);
	map.add_zones("data/map2_zone_ids.pgm", 0.1);

	let world_validities = map.world_validities();
	assert_eq!(world_validities.len(), 3); 
	assert!(world_validities.contains(&bitvec![1, 1, 1, 1])); // always valid
	assert!(world_validities.contains(&bitvec![0, 1, 0, 1])); // valid only when first zone open 
	assert!(world_validities.contains(&bitvec![0, 0, 1, 1])); // valid only when second door open
}

// MAP 4 zones
#[test]
fn test_map_4_construction() {
	let mut map = Map::open("data/map4.pgm", [-1.0, -1.0], [1.0, 1.0]);
	map.add_zones("data/map4_zone_ids.pgm", 0.1);

	assert_eq!(map.n_zones, 4);
	assert_eq!(map.n_worlds, 16);
}

#[test]
fn test_map_4_states() {
	let mut map = Map::open("data/map4.pgm", [-1.0, -1.0], [1.0, 1.0]);
	map.add_zones("data/map4_zone_ids.pgm", 0.1);

	let world_validities = map.world_validities();

	// door validity
	assert_eq!(map.is_state_valid(&[0.0, 0.0]), Belief::Free);
	assert_eq!(map.is_state_valid(&[0.0, -0.24]), Belief::Obstacle);
	assert_eq!(map.is_state_valid(&[-0.67, -0.26]), Belief::Zone(0));

	// world validities
	assert_eq!(map.state_validity(&[-0.29, -0.23]), None);
	assert_eq!(world_validities[map.state_validity(&[0.0, 0.0]).unwrap()], bitvec![1; 16]);

	// world ennumerations
	let mut world_mask = bitvec![0; 16];
	let zone = 1;

	for i in 0..map.n_worlds {
		if i & (zone << 1) != 0 {
			*world_mask.get_mut(i).unwrap() = true;
		}
	}

	assert_eq!(map.zones_to_worlds[zone], world_mask);
}

#[test]
fn test_resize_image() {
	let mut m = Map::open("data/map0.pgm", [-1.0, -1.0], [1.0, 1.0]);
	m.resize(2);
	
	m.save("results/tmp");
	assert!(Path::new("results/tmp.png").exists());
	fs::remove_file("results/tmp.png").unwrap();
}

#[test]
fn test_traversed_zone() {
	let mut m = Map::open("data/map2_thin.pgm", [-1.0, -1.0], [1.0, 1.0]);
	m.add_zones("data/map2_thin_zone_ids.pgm", 0.2);
	
	let space = m.get_traversed_space(&[0.55, -0.8], &[0.55, -0.3]);

	assert_eq!(space, Belief::Zone(0));
}

}
