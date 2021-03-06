#![allow(dead_code, unused_imports)]
#![feature(slice_group_by)]

pub mod common;
pub mod sample_space;
pub mod map_io;
pub mod nearest_neighbor;
pub mod rrt;
pub mod prm;
pub mod prm_graph;
pub mod prm_reachability;
pub mod prm_belief_graph;

use crate::prm::*;
use crate::prm_graph::*;
use crate::sample_space::*;
use crate::map_io::*;

use orbtk::prelude::*;
use std::cell::Cell;

use euc::{buffer::Buffer2d, rasterizer, Pipeline};

#[derive(Clone, PartialEq, Pipeline)]
struct PRMPipeline {
    step: Cell<f64>,
    image: Image
}

impl PRMPipeline {
    pub fn new() -> Self {
        PRMPipeline{
            step: Cell::new(0.0),
            image: Image::from_path(std::path::Path::new("data/map2.png")).unwrap()
        }
    }
}

impl RenderPipeline for PRMPipeline {
    fn draw(&self, render_target: &mut RenderTarget) {
        let mut render_context =
        RenderContext2D::new(render_target.width(), render_target.height());

        //let image = ;
        render_context.draw_image(&self.image, 0.0, 0.0);

        render_context.set_stroke_style(utils::Brush::SolidColor(Color::from("#00AA00")));
        render_context.set_line_width(2.0);
        render_context.begin_path();
        render_context.move_to(self.step.get(), self.step.get());
        render_context.line_to(100.0, 100.0);
        render_context.stroke();

        render_context.restore();
        render_target.draw(render_context.data());
    }
}

#[derive(Default, AsAny)]
pub struct MainViewState {
    step: f64,
}

impl MainViewState {
    fn step(&mut self) {
        self.step += 32.0;
    }
}

impl State for MainViewState {
    fn update(&mut self, _: &mut Registry, ctx: &mut Context) {
        if let Some(cube) = ctx
            .widget()
            .get_mut::<DefaultRenderPipeline>("render_pipeline")
            .0
            .as_any()
            .downcast_ref::<PRMPipeline>()
        {
            cube.step.set(self.step);
        }
    }
}

widget!(
    MainView<MainViewState> {
         render_pipeline: DefaultRenderPipeline
    }
);

impl Template for MainView {
    fn template(self, id: Entity, ctx: &mut BuildContext) -> Self {
        self.name("MainView")
            .render_pipeline(DefaultRenderPipeline(Box::new(PRMPipeline::new())))
            .child(
                Grid::new()
                    .rows("*, auto")
                    .child(
                        Canvas::new()
                            .attach(Grid::row(0))
                            .render_pipeline(id)
                            .build(ctx),
                    )
                    .child(
                        Button::new()
                            .text("step")
                            .v_align("end")
                            .attach(Grid::row(1))
                            .margin(4.0)
                            .on_click(move |states, _| {
                                states.get_mut::<MainViewState>(id).step();
                                true
                            })
                            .build(ctx),
                    )
                    .build(ctx),
            )
    }
}

fn display(){
   // use this only if you want to run it as web application.
   orbtk::initialize();

   Application::new()
       .window(|ctx| {
           Window::new()
               .title("OrbTk - canvas example")
               .position((100.0, 100.0))
               .size(1000.0, 1000.0)
               .resizeable(false)
               .child(MainView::new().build(ctx))
               .build(ctx)
       })
       .run();
}

fn main() {
    /*let mut m = Map::open("data/map2.pgm", [-1.0, -1.0], [1.0, 1.0]);
	m.add_zones("data/map2_zone_ids.pgm");

	fn goal(state: &[f64; 2]) -> bool {
		(state[0] - 0.0).abs() < 0.05 && (state[1] - 0.9).abs() < 0.05
	}

	let mut prm = PRM::new(ContinuousSampler::new([-1.0, -1.0], [1.0, 1.0]),
						   DiscreteSampler::new(),
						   &m);

	prm.grow_graph(&[0.55, -0.8], goal, 0.05, 5.0, 2000, 100000).unwrap();
	prm.plan().unwrap();
    prm.react(&[0.0, -0.8], &vec![0.25, 0.25, 0.25, 0.25], 0.2).unwrap();

    prm.print_summary();*/

    display();
}