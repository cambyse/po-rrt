use crate::NodeRef;
use image::Luma;
use image::DynamicImage::ImageLuma8;
use std::vec::Vec;

#[derive(Clone)]
pub struct Map
{
	img: image::GrayImage,
	low: [f64; 2],
	//up: [f64; 2], // low + up are enough and make up redundant
	ppm: f64,
}

impl Map {
	pub fn open(filepath : &str, low: [f64; 2], up: [f64; 2]) -> Map{
		let img = image::open(filepath).expect(&format!("Impossible to open image: {}", filepath));

		match img {
			ImageLuma8(gray_img) => {
				return Map::build(gray_img, low, up);
			},
			_ => panic!("Wrong image format!"),
		}
	}

	pub fn save(&self, filepath: &str) {
		self.img.save(filepath).expect("Couldn't save image");
	}

	pub fn is_state_valid(&self, xy: &[f64; 2]) -> bool {
		let ij = self.to_pixel_coordinates(&*xy);
		let p = self.img.get_pixel(ij[1], ij[0]);

		p[0] == 255
	}

	pub fn draw_path(&mut self, path: Vec<[f64;2]>) {
		for i in 1..path.len() {
			self.draw_line(path[i-1], path[i], 0);
		}
	}

	pub fn draw_tree(&mut self, root: NodeRef<2>) {
		for c in &root.borrow().children {
			self.draw_line(root.borrow().state, c.borrow().state, 180);
			self.draw_tree(c.clone());
		}
	}

	fn to_pixel_coordinates(&self, xy: &[f64; 2]) -> [u32; 2] {
		let i: u32 = ((self.img.height() as f64) - (xy[1] - self.low[1]) * self.ppm) as u32;
		let j: u32 = ((xy[0] - self.low[0]) * self.ppm) as u32;

		[i, j]
	}

	fn build(img: image::GrayImage, low: [f64; 2], up: [f64; 2])-> Map {
		let ppm = (img.width() as f64) / (up[0] - low[0]);

		Map{img, low, /*up,*/ ppm}
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
}