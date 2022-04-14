use std::mem;
use rand::{distributions::{Distribution, Uniform}, SeedableRng};
use winit::{
    event::*,
    event_loop::{ControlFlow, EventLoop},
    window::{Window, WindowBuilder},
    dpi::PhysicalSize
};
use wgpu::util::DeviceExt;
use image::{self, GenericImageView, DynamicImage};

const PARTICLES_PER_GROUP: u32 = 64;

const NUM_PARTICLES: u32 = 8192;
const Q_CHARGE: f32 = 0.2;
const BLANK_LEVEL: f32 = 0.95;
const TIME_DELTA: f32 = 0.001;
const D_MAX: f32 = 0.005;
const SUSTAIN: f32 = 0.95;

struct Model {
    surface: wgpu::Surface,
    device: wgpu::Device,
    queue: wgpu::Queue,
    particle_bind_groups: Vec<wgpu::BindGroup>,
    particle_buffers: Vec<wgpu::Buffer>,
    vertices_buffer: wgpu::Buffer,
    texture_bind_group: wgpu::BindGroup,
    compute_pipeline: wgpu::ComputePipeline,
    render_pipeline: wgpu::RenderPipeline,
    work_group_count: u32,
    frame_num: usize
}

impl Model {
    async fn new(window: &Window) -> Self {
        let size = window.inner_size();
        let instance = wgpu::Instance::new(wgpu::Backends::all());
        let surface = unsafe { instance.create_surface(window) };
        let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::default(),
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        }).await.unwrap();
        let (device, queue) = adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                features: wgpu::Features::empty(),
                limits: wgpu::Limits::default(),
            },
            None,
        ).await.unwrap();
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface.get_preferred_format(&adapter).unwrap(),
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::Fifo,
        };
        surface.configure(&device, &config);

        let compute_shader = device.create_shader_module(&wgpu::include_wgsl!("compute.wgsl"));
        let draw_shader = device.create_shader_module(&wgpu::include_wgsl!("draw.wgsl"));

        let img_bytes = include_bytes!("../assets/black_cat.png");
        let img = image::load_from_memory(img_bytes).unwrap();
        let texture = create_and_write_texture(&device, &queue, &img);
        let texture_bind_group_layout = create_texture_bind_group_layout(&device);
        let texture_bind_group = create_texture_bind_group(&device, &texture_bind_group_layout, &texture);

        let param_data = create_param_data(&img);
        let param_buffer = create_parameter_buffer(&device, &param_data);

        let compute_bind_group_layout = create_compute_bind_group_layout(&device, param_data.len());
        let compute_pipeline_layout = create_compute_pipeline_layout(&device, &compute_bind_group_layout, &texture_bind_group_layout);
        let compute_pipeline = create_compute_pipeline(&device, &compute_pipeline_layout, &compute_shader);

        let render_pipeline_layout = create_render_pipeline_layout(&device);
        let render_pipeline = create_render_pipeline(&device, &draw_shader, &config, &render_pipeline_layout);

        let vertices_buffer = create_vertices_buffer(&device);
        let particle_buffers = create_particle_buffers(&device);
        let particle_bind_groups = create_particle_bind_groups(&device, &compute_bind_group_layout, &particle_buffers, &param_buffer);

        let work_group_count = ((NUM_PARTICLES as f32) / (PARTICLES_PER_GROUP as f32)).ceil() as u32;

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
            work_group_count,
            frame_num: 0,
        }
    }

    fn input(&mut self, _: &WindowEvent) -> bool {
        false
    }

    fn update(&mut self) {}

    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let frame = self.surface.get_current_texture()?;
        let view = frame.texture.create_view(&wgpu::TextureViewDescriptor::default());

        let color_attachments = [wgpu::RenderPassColorAttachment {
            view: &view,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color {
                    r: 1.0,
                    g: 1.0,
                    b: 1.0,
                    a: 1.0
                }),
                store: true,
            },
        }];
        let render_pass_descriptor = wgpu::RenderPassDescriptor {
            label: None,
            color_attachments: &color_attachments,
            depth_stencil_attachment: None,
        };
        let mut command_encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut compute_pass = command_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });
            compute_pass.set_pipeline(&self.compute_pipeline);
            compute_pass.set_bind_group(0, &self.particle_bind_groups[self.frame_num % 2], &[]);
            compute_pass.set_bind_group(1, &self.texture_bind_group, &[]);
            compute_pass.dispatch(self.work_group_count, 1, 1);
        }

        {
            let mut render_pass = command_encoder.begin_render_pass(&render_pass_descriptor);
            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.set_vertex_buffer(0, self.particle_buffers[(self.frame_num + 1) % 2].slice(..));
            render_pass.set_vertex_buffer(1, self.vertices_buffer.slice(..));
            render_pass.draw(0..3, 0..NUM_PARTICLES);
        }
        self.frame_num += 1;

        self.queue.submit(Some(command_encoder.finish()));

        frame.present();
        Ok(())
    }
}

fn main() {
    env_logger::init();
    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_inner_size(PhysicalSize::new(1800.0, 1800.0))
        .build(&event_loop).unwrap();

    let mut model: Model = pollster::block_on(Model::new(&window));

    event_loop.run(move |event, _, control_flow| match event {
        Event::WindowEvent {
            ref event,
            window_id,
        } if window_id == window.id() => {
            if !model.input(event) {
                match event {
                    WindowEvent::CloseRequested
                    | WindowEvent::KeyboardInput {
                        input:
                            KeyboardInput {
                                state: ElementState::Pressed,
                                virtual_keycode: Some(VirtualKeyCode::Escape),
                                ..
                            },
                        ..
                    } => *control_flow = ControlFlow::Exit,
                    _ => {}
                }
            }
        }
        Event::RedrawRequested(_) => {
            model.update();
            match model.render() {
                Ok(_) => {}
                Err(wgpu::SurfaceError::OutOfMemory) => *control_flow = ControlFlow::Exit,
                //Err(wgpu::SurfaceError::Lost) => state.resize(state.size),
                Err(e) => eprintln!("{:?}", e),
            }
        }
        Event::RedrawEventsCleared => {
            window.request_redraw();
        }
        _ => {}
    });
}

fn create_param_data(img: &DynamicImage) -> Vec<f32> {
    let img_rgba = img.to_rgba8();
    let mut total_charge = 0.0;
    for (_, _, p) in img_rgba.enumerate_pixels() {
        total_charge += BLANK_LEVEL - (0.299 * p[0] as f32 + 0.578 * p[1] as f32 + 0.11 * p[2] as f32) / 255.0;
    }
    let dot_charge_total = NUM_PARTICLES as f32 * Q_CHARGE;

    let dt = TIME_DELTA;
    let dt_2 = 0.5 * dt;
    let dtt = dt * dt;
    let v_max = D_MAX / dt;
    let scale = dot_charge_total / total_charge;
    return vec![Q_CHARGE, BLANK_LEVEL, TIME_DELTA, D_MAX, SUSTAIN, dt, dt_2, dtt, v_max, scale]
}


fn create_and_write_texture(device: &wgpu::Device, queue: &wgpu::Queue, img: &DynamicImage) -> wgpu::Texture {
    let (width, height) = img.dimensions();
    let size = wgpu::Extent3d {
        width, height,
        depth_or_array_layers: 1
    };
    let texture = device.create_texture(
        &wgpu::TextureDescriptor {
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            label: Some("texture")
        }
    );
    queue.write_texture(
        wgpu::ImageCopyTexture {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &img.to_rgba8(),
        wgpu::ImageDataLayout {
            offset: 0,
            bytes_per_row: std::num::NonZeroU32::new(4 * width),
            rows_per_image: std::num::NonZeroU32::new(height)
        },
        size
    );
    return texture
}

fn create_texture_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    return device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Texture {
                    multisampled: false,
                    view_dimension: wgpu::TextureViewDimension::D2,
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                },
                count: None,
            },
        ],
        label: Some("texture_bind_group_layout"),
    })
}

fn create_texture_bind_group(
    device: &wgpu::Device,
    texture_bind_group_layout: &wgpu::BindGroupLayout,
    texture: &wgpu::Texture
) -> wgpu::BindGroup {
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    return device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout: texture_bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&view),
            },
        ],
        label: Some("diffuse_bind_group"),
    })
}

fn create_compute_bind_group_layout(device: &wgpu::Device, parameter_size: usize) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: wgpu::BufferSize::new(
                        (parameter_size * mem::size_of::<f32>()) as _,
                    ),
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: wgpu::BufferSize::new((NUM_PARTICLES * 16) as _),
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: wgpu::BufferSize::new((NUM_PARTICLES * 16) as _),
                },
                count: None,
            },
        ],
        label: None,
    })
}

fn create_compute_pipeline_layout(
    device: &wgpu::Device,
    compute_bind_group_layout: &wgpu::BindGroupLayout,
    texture_bind_group_layout: &wgpu::BindGroupLayout
) -> wgpu::PipelineLayout {
    device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("compute"),
        bind_group_layouts: &[compute_bind_group_layout, texture_bind_group_layout],
        push_constant_ranges: &[],
    })
}

fn create_compute_pipeline(
    device: &wgpu::Device,
    compute_pipeline_layout: &wgpu::PipelineLayout,
    compute_shader: &wgpu::ShaderModule
) -> wgpu::ComputePipeline {
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Compute Pipeline"),
        layout: Some(&compute_pipeline_layout),
        module: compute_shader,
        entry_point: "main",
    })
}

fn create_render_pipeline_layout(device: &wgpu::Device) -> wgpu::PipelineLayout {
    device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("render"),
        bind_group_layouts: &[],
        push_constant_ranges: &[],
    })
}

fn create_render_pipeline(
    device: &wgpu::Device,
    draw_shader: &wgpu::ShaderModule,
    config: &wgpu::SurfaceConfiguration,
    render_pipeline_layout: &wgpu::PipelineLayout
) -> wgpu::RenderPipeline {
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: None,
        layout: Some(&render_pipeline_layout),
        vertex: wgpu::VertexState {
            module: &draw_shader,
            entry_point: "main_vs",
            buffers: &[
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
            ],
        },
        fragment: Some(wgpu::FragmentState {
            module: &draw_shader,
            entry_point: "main_fs",
            targets: &[config.format.into()],
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
    })
}

fn create_vertices_buffer(device: &wgpu::Device) -> wgpu::Buffer {
    let vertex_buffer_data = [
        0.005f32, 0.00,
        -0.0025, 0.004,
        -0.0025, -0.004,
    ];
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Vertex Buffer"),
        contents: bytemuck::bytes_of(&vertex_buffer_data),
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
    })
}

fn create_particle_buffers(device: &wgpu::Device) -> Vec<wgpu::Buffer> {
    let mut initial_particle_data : Vec<f32> = vec![0.0; (6 * NUM_PARTICLES) as usize];
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
    for i in 0..2 {
        particle_buffers.push(
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("Particle Buffer {}", i)),
                contents: bytemuck::cast_slice(&initial_particle_data),
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST
            }),
        );
    }
    return particle_buffers
}

fn create_parameter_buffer(device: &wgpu::Device, param_data: &Vec<f32>) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Parameter Buffer"),
        contents: bytemuck::cast_slice(&param_data),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    })
}

fn create_particle_bind_groups(
    device: &wgpu::Device,
    compute_bind_group_layout: &wgpu::BindGroupLayout,
    particle_buffers: &Vec<wgpu::Buffer>,
    parameter_buffer: &wgpu::Buffer
) -> Vec<wgpu::BindGroup> {
    let mut particle_bind_groups = Vec::<wgpu::BindGroup>::new();
    for i in 0..2 {
        particle_bind_groups.push(device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &compute_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: parameter_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: particle_buffers[i].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: particle_buffers[(i+1)%2].as_entire_binding(),
                },
            ],
            label: None
        }));
    };
    return particle_bind_groups
}
