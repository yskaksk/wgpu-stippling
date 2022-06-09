use image::{self, DynamicImage, GenericImageView};
use pollster::FutureExt;
use rand::{
    distributions::{Distribution, Uniform},
    SeedableRng,
};
use std::mem;
use wgpu::{BufferUsages, TextureFormat, TextureUsages};
use winit::{event::*, window::Window};

use crate::util::{
    compute_work_group_count, create_buffer, create_compute_pipeline, create_render_pipeline,
    create_texture, padded_bytes_per_row, save_gif, BindGroupBuilder, BindGroupLayoutBuilder,
};

const PARTICLES_PER_GROUP: u32 = 64;
const DOT_SIZE: f32 = 0.005;

const NUM_PARTICLES: u32 = 16384;
const Q_CHARGE: f32 = 0.05;
const BLANK_LEVEL: f32 = 0.95;
const TIME_DELTA: f32 = 0.001;
const D_MAX: f32 = 0.005;
const SUSTAIN: f32 = 0.95;

pub struct Model {
    surface: wgpu::Surface,
    device: wgpu::Device,
    queue: wgpu::Queue,
    particle_bind_groups: Vec<wgpu::BindGroup>,
    particle_buffers: Vec<wgpu::Buffer>,
    vertices_buffer: wgpu::Buffer,
    texture_bind_group: wgpu::BindGroup,
    compute_pipeline: wgpu::ComputePipeline,
    render_pipeline: wgpu::RenderPipeline,
    parameter_bind_group_layout: wgpu::BindGroupLayout,
    work_group_count: u32,
    texture_size: wgpu::Extent3d,
    frames: Vec<Vec<u8>>,
    frame_num: usize,
}

impl Model {
    pub async fn new(window: &Window) -> Self {
        let size = window.inner_size();
        let instance = wgpu::Instance::new(wgpu::Backends::all());
        let surface = unsafe { instance.create_surface(window) };
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .unwrap();
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    features: wgpu::Features::empty(),
                    limits: wgpu::Limits::default(),
                },
                None,
            )
            .await
            .unwrap();
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface.get_preferred_format(&adapter).unwrap(),
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::Fifo,
        };
        surface.configure(&device, &config);
        let work_group_count =
            ((NUM_PARTICLES as f32) / (PARTICLES_PER_GROUP as f32)).ceil() as u32;

        let draw_shader = device.create_shader_module(&wgpu::include_wgsl!("draw.wgsl"));
        let gray_scale_shader = device.create_shader_module(&wgpu::include_wgsl!("grayscale.wgsl"));
        let cache_shader = device.create_shader_module(&wgpu::include_wgsl!("cache.wgsl"));
        let compute_shader = device.create_shader_module(&wgpu::include_wgsl!("compute.wgsl"));

        let img_bytes = include_bytes!("../assets/cat_3_square.png");
        let img = image::load_from_memory(img_bytes).unwrap();

        let (w, h) = img.dimensions();
        let texture_size = wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        };

        let gray_texture =
            create_gray_texture(&device, &queue, &img, &gray_scale_shader, texture_size);
        let cache_texture =
            create_cache_texture(&device, &queue, &gray_texture, &cache_shader, texture_size);

        let img_bytes2 = include_bytes!("../assets/cat_face.png");
        let img2 = image::load_from_memory(img_bytes2).unwrap();
        let (w, h) = img2.dimensions();
        let texture_size2 = wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        };
        let gray_texture2 =
            create_gray_texture(&device, &queue, &img2, &gray_scale_shader, texture_size2);
        let cache_texture2 = create_cache_texture(
            &device,
            &queue,
            &gray_texture2,
            &cache_shader,
            texture_size2,
        );

        let param_data = create_param_data(0.0);
        let parameter_bind_group_layout = BindGroupLayoutBuilder::new()
            .add_uniform_buffer(wgpu::BufferSize::new(
                (param_data.len() * mem::size_of::<f32>()) as _,
            ))
            .create_bind_group_layout(&device, None);

        let texture_bind_group_layout = BindGroupLayoutBuilder::new()
            .add_texture()
            .add_texture()
            .create_bind_group_layout(&device, Some("texture bind group layout"));
        let texture_bind_group = BindGroupBuilder::new()
            .add_resource(wgpu::BindingResource::TextureView(
                &cache_texture.create_view(&wgpu::TextureViewDescriptor::default()),
            ))
            .add_resource(wgpu::BindingResource::TextureView(
                &cache_texture2.create_view(&wgpu::TextureViewDescriptor::default()),
            ))
            .create_bind_group(
                &device,
                Some("texture bind group"),
                &texture_bind_group_layout,
            );

        let compute_bind_group_layout = BindGroupLayoutBuilder::new()
            .add_storage_buffer(wgpu::BufferSize::new((NUM_PARTICLES * 16) as _), true)
            .add_storage_buffer(wgpu::BufferSize::new((NUM_PARTICLES * 16) as _), false)
            .create_bind_group_layout(&device, None);
        let compute_pipeline = create_compute_pipeline(
            &device,
            &[
                &parameter_bind_group_layout,
                &compute_bind_group_layout,
                &texture_bind_group_layout,
            ],
            &compute_shader,
        );

        let vertex_buffer_layouts = &[
            wgpu::VertexBufferLayout {
                array_stride: 6 * 4,
                step_mode: wgpu::VertexStepMode::Instance,
                attributes: &wgpu::vertex_attr_array![0 => Float32x2, 1 => Float32x2, 2 => Float32x2],
            },
            wgpu::VertexBufferLayout {
                array_stride: 2 * 4,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &wgpu::vertex_attr_array![3 => Float32x2],
            },
        ];
        let render_pipeline =
            create_render_pipeline(&device, &config, vertex_buffer_layouts, &draw_shader);

        let vertices_buffer = create_vertices_buffer(&device);
        let particle_buffers = create_particle_buffers(&device);
        let particle_bind_groups = Vec::from_iter((0..2).map(|i| {
            BindGroupBuilder::new()
                .add_resource(particle_buffers[i].as_entire_binding())
                .add_resource(particle_buffers[(i + 1) % 2].as_entire_binding())
                .create_bind_group(&device, None, &compute_bind_group_layout)
        }));
        let frames = Vec::new();

        Model {
            surface,
            device,
            queue,
            particle_bind_groups,
            particle_buffers,
            vertices_buffer,
            texture_bind_group,
            compute_pipeline,
            render_pipeline,
            parameter_bind_group_layout,
            work_group_count,
            texture_size,
            frames,
            frame_num: 0,
        }
    }

    pub fn input(&mut self, _: &WindowEvent) -> bool {
        false
    }

    pub fn update(&mut self) {}

    fn render_frame(&mut self) -> anyhow::Result<()> {
        let mut command_encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        let target_texture = create_texture(
            &self.device,
            self.texture_size,
            TextureFormat::Bgra8UnormSrgb,
            TextureUsages::COPY_SRC | TextureUsages::RENDER_ATTACHMENT,
        );
        let color_attachments = [wgpu::RenderPassColorAttachment {
            view: &target_texture.create_view(&wgpu::TextureViewDescriptor::default()),
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color {
                    r: 1.0,
                    g: 1.0,
                    b: 1.0,
                    a: 1.0,
                }),
                store: true,
            },
        }];
        let render_pass_descriptor = wgpu::RenderPassDescriptor {
            label: None,
            color_attachments: &color_attachments,
            depth_stencil_attachment: None,
        };
        {
            let mut render_pass = command_encoder.begin_render_pass(&render_pass_descriptor);
            render_pass.set_pipeline(&self.render_pipeline);
            render_pass
                .set_vertex_buffer(0, self.particle_buffers[(self.frame_num + 1) % 2].slice(..));
            render_pass.set_vertex_buffer(1, self.vertices_buffer.slice(..));
            render_pass.draw(0..24, 0..NUM_PARTICLES);
        }

        let padded_bytes_per_row = padded_bytes_per_row(self.texture_size.width);
        let unpadded_bytes_per_row = self.texture_size.width as usize * 4;
        let output_buffer_size = padded_bytes_per_row as u64
            * self.texture_size.height as u64
            * std::mem::size_of::<u8>() as u64;
        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: output_buffer_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        command_encoder.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                aspect: wgpu::TextureAspect::All,
                texture: &target_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
            },
            wgpu::ImageCopyBuffer {
                buffer: &output_buffer,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: std::num::NonZeroU32::new(padded_bytes_per_row as u32),
                    rows_per_image: std::num::NonZeroU32::new(self.texture_size.height),
                },
            },
            self.texture_size,
        );
        self.queue.submit(Some(command_encoder.finish()));

        let buffer_slice = output_buffer.slice(..);
        let mapping = buffer_slice.map_async(wgpu::MapMode::Read);
        self.device.poll(wgpu::Maintain::Wait);
        mapping.block_on().unwrap();
        let padded_data = buffer_slice.get_mapped_range();

        let data = padded_data
            .chunks(padded_bytes_per_row as _)
            .map(|chunk| &chunk[..unpadded_bytes_per_row as _])
            .flatten()
            .map(|x| *x)
            .collect::<Vec<_>>();
        drop(padded_data);
        output_buffer.unmap();
        self.frames.push(data);
        Ok(())
    }

    pub fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let parameter_data = create_param_data(self.frame_num as f32);
        let parameter_buffer = create_buffer(
            &self.device,
            bytemuck::cast_slice(&parameter_data),
            BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            Some("Parameter Buffer"),
        );
        let parameter_bind_group = BindGroupBuilder::new()
            .add_resource(parameter_buffer.as_entire_binding())
            .create_bind_group(&self.device, None, &self.parameter_bind_group_layout);
        let mut command_encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut compute_pass =
                command_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });
            compute_pass.set_pipeline(&self.compute_pipeline);
            compute_pass.set_bind_group(0, &parameter_bind_group, &[]);
            compute_pass.set_bind_group(1, &self.particle_bind_groups[self.frame_num % 2], &[]);
            compute_pass.set_bind_group(2, &self.texture_bind_group, &[]);
            compute_pass.dispatch(self.work_group_count, 1, 1);
        }

        let frame = self.surface.get_current_texture()?;
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let color_attachments = [wgpu::RenderPassColorAttachment {
            view: &view,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color {
                    r: 1.0,
                    g: 1.0,
                    b: 1.0,
                    a: 1.0,
                }),
                store: true,
            },
        }];
        let render_pass_descriptor = wgpu::RenderPassDescriptor {
            label: None,
            color_attachments: &color_attachments,
            depth_stencil_attachment: None,
        };
        {
            let mut render_pass = command_encoder.begin_render_pass(&render_pass_descriptor);
            render_pass.set_pipeline(&self.render_pipeline);
            render_pass
                .set_vertex_buffer(0, self.particle_buffers[(self.frame_num + 1) % 2].slice(..));
            render_pass.set_vertex_buffer(1, self.vertices_buffer.slice(..));
            render_pass.draw(0..24, 0..NUM_PARTICLES);
        }
        self.queue.submit(Some(command_encoder.finish()));
        frame.present();

        //if self.frame_num % 6 == 0 {
        //    self.render_frame().unwrap();
        //}
        self.frame_num += 1;

        //if self.frame_num == 270 {
        //    println!("saving...");
        //    save_gif(
        //        "output.gif",
        //        &mut self.frames,
        //        1,
        //        self.texture_size.width as u16,
        //    )
        //    .unwrap();
        //    println!("saved!!!");
        //}

        Ok(())
    }
}

fn create_param_data(frame_count: f32) -> Vec<f32> {
    let dt = TIME_DELTA;
    let dt_2 = 0.5 * dt;
    let dtt = dt * dt;
    let v_max = D_MAX / dt;
    let pi = std::f32::consts::PI;
    return vec![
        Q_CHARGE,
        BLANK_LEVEL,
        TIME_DELTA,
        D_MAX,
        SUSTAIN,
        dt,
        dt_2,
        dtt,
        v_max,
        frame_count,
        pi,
    ];
}

fn create_vertices_buffer(device: &wgpu::Device) -> wgpu::Buffer {
    // 8 * 6
    let mut vertex_buffer_data: [f32; 48] = [0.0; 48];
    let theta = 2.0 * std::f32::consts::PI / 8.0;
    for i in 0..8 {
        vertex_buffer_data[6 * i] = 0.0;
        vertex_buffer_data[6 * i + 1] = 0.0;
        vertex_buffer_data[6 * i + 2] = DOT_SIZE * (i as f32 * theta).cos();
        vertex_buffer_data[6 * i + 3] = DOT_SIZE * (i as f32 * theta).sin();
        vertex_buffer_data[6 * i + 4] = DOT_SIZE * ((i as f32 + 1.0) * theta).cos();
        vertex_buffer_data[6 * i + 5] = DOT_SIZE * ((i as f32 + 1.0) * theta).sin();
    }
    create_buffer(
        device,
        bytemuck::bytes_of(&vertex_buffer_data),
        BufferUsages::VERTEX | BufferUsages::COPY_DST,
        Some("Vertex Buffer"),
    )
}

fn create_particle_buffers(device: &wgpu::Device) -> Vec<wgpu::Buffer> {
    let mut initial_particle_data: Vec<f32> = vec![0.0; (6 * NUM_PARTICLES) as usize];
    let mut rng = rand::rngs::StdRng::seed_from_u64(333);
    let unif = Uniform::new_inclusive(-1.0, 1.0);
    for particle_instance_chunk in initial_particle_data.chunks_mut(6) {
        particle_instance_chunk[0] = unif.sample(&mut rng);
        particle_instance_chunk[1] = unif.sample(&mut rng);
        particle_instance_chunk[2] = 0.0;
        particle_instance_chunk[3] = 0.0;
        particle_instance_chunk[4] = 0.0;
        particle_instance_chunk[5] = 0.0;
    }

    let mut particle_buffers = Vec::<wgpu::Buffer>::new();
    let usage = BufferUsages::VERTEX
        | BufferUsages::STORAGE
        | BufferUsages::COPY_DST
        | BufferUsages::MAP_READ;
    for i in 0..2 {
        particle_buffers.push(create_buffer(
            device,
            bytemuck::cast_slice(&initial_particle_data),
            usage,
            Some(&format!("Particle Bufeer {}", i)),
        ));
    }
    return particle_buffers;
}

fn create_gray_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    input_image: &DynamicImage,
    shader: &wgpu::ShaderModule,
    texture_size: wgpu::Extent3d,
) -> wgpu::Texture {
    let input_texture = create_texture(
        device,
        texture_size,
        TextureFormat::Rgba8UnormSrgb,
        TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
    );
    queue.write_texture(
        input_texture.as_image_copy(),
        &input_image.to_rgba8(),
        wgpu::ImageDataLayout {
            offset: 0,
            bytes_per_row: std::num::NonZeroU32::new(4 * texture_size.width),
            rows_per_image: std::num::NonZeroU32::new(texture_size.height),
        },
        texture_size,
    );
    let output_texture = create_texture(
        device,
        texture_size,
        TextureFormat::Rgba32Float,
        TextureUsages::TEXTURE_BINDING | TextureUsages::STORAGE_BINDING,
    );
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("grayscale pipeline"),
        layout: None,
        module: &shader,
        entry_point: "main",
    });
    let bind_group = BindGroupBuilder::new()
        .add_resource(wgpu::BindingResource::TextureView(
            &input_texture.create_view(&wgpu::TextureViewDescriptor::default()),
        ))
        .add_resource(wgpu::BindingResource::TextureView(
            &output_texture.create_view(&wgpu::TextureViewDescriptor::default()),
        ))
        .create_bind_group(
            &device,
            Some("gray scale bind group"),
            &pipeline.get_bind_group_layout(0),
        );
    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let (dispatch_width, dispatch_height) =
            compute_work_group_count((texture_size.width, texture_size.height), (16, 16));
        let mut compute_pass =
            encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });
        compute_pass.set_pipeline(&pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);
        compute_pass.dispatch(dispatch_width, dispatch_height, 1);
    }
    queue.submit(Some(encoder.finish()));

    return output_texture;
}

fn create_cache_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    input_texture: &wgpu::Texture,
    shader: &wgpu::ShaderModule,
    texture_size: wgpu::Extent3d,
) -> wgpu::Texture {
    let output_texture = create_texture(
        device,
        texture_size,
        TextureFormat::Rgba32Float,
        TextureUsages::TEXTURE_BINDING | TextureUsages::STORAGE_BINDING,
    );
    let param = vec![BLANK_LEVEL, NUM_PARTICLES as f32 * Q_CHARGE];
    let param_buffer = create_buffer(
        device,
        bytemuck::cast_slice(&param),
        BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        Some("cache param buffer"),
    );
    let bind_group_layout = BindGroupLayoutBuilder::new()
        .add_uniform_buffer(wgpu::BufferSize::new(
            (param.len() * mem::size_of::<f32>()) as _,
        ))
        .add_texture()
        .add_storage_texture()
        .create_bind_group_layout(device, Some("cache bind group layout"));
    let bind_group = BindGroupBuilder::new()
        .add_resource(param_buffer.as_entire_binding())
        .add_resource(wgpu::BindingResource::TextureView(
            &input_texture.create_view(&wgpu::TextureViewDescriptor::default()),
        ))
        .add_resource(wgpu::BindingResource::TextureView(
            &output_texture.create_view(&wgpu::TextureViewDescriptor::default()),
        ))
        .create_bind_group(device, Some("cache scale bind group"), &bind_group_layout);
    let pipeline = create_compute_pipeline(device, &[&bind_group_layout], &shader);
    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let (dispatch_width, dispatch_height) =
            compute_work_group_count((texture_size.width, texture_size.height), (16, 16));
        let mut compute_pass =
            encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });
        compute_pass.set_pipeline(&pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);
        compute_pass.dispatch(dispatch_width, dispatch_height, 1);
    }
    queue.submit(Some(encoder.finish()));
    return output_texture;
}
