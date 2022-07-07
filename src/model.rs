use image;
use wgpu::{Device, Queue, Surface};
use winit::{event::*, window::Window};

use crate::resources::{ImgShaders, Resources};

#[cfg(feature = "savegif")]
use crate::util::{copy_texture_to_buffer, create_texture, padded_bytes_per_row, save_gif};
#[cfg(feature = "savegif")]
use image::GenericImageView;
#[cfg(feature = "savegif")]
use pollster::FutureExt;
#[cfg(feature = "savegif")]
use wgpu::{Extent3d, TextureFormat, TextureUsages};

pub struct Model {
    surface: Surface,
    device: Device,
    queue: Queue,
    resources: Resources,
    frame_num: usize,
    #[cfg(feature = "savegif")]
    texture_size: Extent3d,
    #[cfg(feature = "savegif")]
    frames: Vec<Vec<u8>>,
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
            format: surface.get_supported_formats(&adapter)[0],
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::Fifo,
        };
        surface.configure(&device, &config);

        let gray_scale_shader = device.create_shader_module(wgpu::include_wgsl!("grayscale.wgsl"));
        let cache_shader = device.create_shader_module(wgpu::include_wgsl!("cache.wgsl"));
        let img_shaders = ImgShaders::new(gray_scale_shader, cache_shader);

        let img = image::load_from_memory(include_bytes!("../assets/cat_3_square.png")).unwrap();
        let img2 = image::load_from_memory(include_bytes!("../assets/cat_face.png")).unwrap();

        let compute_shader = device.create_shader_module(wgpu::include_wgsl!("compute.wgsl"));
        let draw_shader = device.create_shader_module(wgpu::include_wgsl!("draw.wgsl"));
        let resources = Resources::new(
            &device,
            &queue,
            &config,
            &img,
            &img2,
            &img_shaders,
            &compute_shader,
            &draw_shader,
        );

        #[cfg(feature = "savegif")]
        let (width, height) = img.dimensions();
        Model {
            surface,
            device,
            queue,
            resources,
            frame_num: 0,
            #[cfg(feature = "savegif")]
            texture_size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            #[cfg(feature = "savegif")]
            frames: Vec::new(),
        }
    }

    pub fn input(&mut self, _: &WindowEvent) -> bool {
        false
    }

    pub fn update(&mut self) {}

    #[cfg(feature = "savegif")]
    fn save_frame(&mut self) -> anyhow::Result<()> {
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
        self.resources
            .render_pass(&mut command_encoder, &color_attachments, self.frame_num);

        let padded_bytes_per_row = padded_bytes_per_row(self.texture_size.width);
        let output_buffer_size = padded_bytes_per_row as u64
            * self.texture_size.height as u64
            * std::mem::size_of::<u8>() as u64;
        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: output_buffer_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        copy_texture_to_buffer(
            &mut command_encoder,
            &target_texture,
            self.texture_size,
            &output_buffer,
        );
        self.queue.submit(Some(command_encoder.finish()));

        let buffer_slice = output_buffer.slice(..);
        let mapping = buffer_slice.map_async(wgpu::MapMode::Read);
        self.device.poll(wgpu::Maintain::Wait);
        mapping.block_on().unwrap();
        let padded_data = buffer_slice.get_mapped_range();

        let unpadded_bytes_per_row = self.texture_size.width as usize * 4;
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
        let mut command_encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        self.resources
            .compute_pass(&self.device, &mut command_encoder, self.frame_num);

        let frame = self.surface.get_current_texture()?;
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let color_attachments = [Some(wgpu::RenderPassColorAttachment {
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
        })];
        self.resources
            .render_pass(&mut command_encoder, &color_attachments, self.frame_num);
        self.queue.submit(Some(command_encoder.finish()));
        frame.present();

        #[cfg(feature = "savegif")]
        {
            if self.frame_num <= 100 {
                self.save_frame().unwrap();
            }
            if self.frame_num == 100 {
                println!("saving...");
                save_gif(
                    "output.gif",
                    &mut self.frames,
                    1,
                    self.texture_size.width as u16,
                )
                .unwrap();
                println!("saved!!!");
            }
        }

        self.frame_num += 1;
        Ok(())
    }
}
