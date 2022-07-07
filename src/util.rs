use wgpu::util::DeviceExt;
use wgpu::{
    BindGroup, BindGroupLayout, BindGroupLayoutEntry, Buffer, ComputePipeline, Device, Extent3d,
    RenderPipeline, ShaderModule, Texture, TextureFormat, TextureUsages,
};

pub fn create_compute_pipeline(
    device: &Device,
    bind_group_layouts: &[&BindGroupLayout],
    shader: &ShaderModule,
) -> ComputePipeline {
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("compute"),
        bind_group_layouts,
        push_constant_ranges: &[],
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Compute Pipeline"),
        layout: Some(&pipeline_layout),
        module: shader,
        entry_point: "main",
    })
}

pub fn create_render_pipeline(
    device: &Device,
    config: &wgpu::SurfaceConfiguration,
    vertex_buffer_layouts: &[wgpu::VertexBufferLayout],
    shader: &ShaderModule,
) -> RenderPipeline {
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("render"),
        bind_group_layouts: &[],
        push_constant_ranges: &[],
    });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: None,
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: "main_vs",
            buffers: vertex_buffer_layouts,
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: "main_fs",
            targets: &[Some(config.format.into())],
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
    })
}

pub struct BindGroupLayoutBuilder {
    entries: Vec<BindGroupLayoutEntry>,
    binding: u32,
}

impl BindGroupLayoutBuilder {
    pub fn new() -> Self {
        BindGroupLayoutBuilder {
            entries: vec![],
            binding: 0,
        }
    }

    pub fn add_texture(&self) -> Self {
        let mut entries = self.entries.clone();
        entries.push(wgpu::BindGroupLayoutEntry {
            binding: self.binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Texture {
                multisampled: false,
                view_dimension: wgpu::TextureViewDimension::D2,
                sample_type: wgpu::TextureSampleType::Float { filterable: false },
            },
            count: None,
        });
        BindGroupLayoutBuilder {
            entries,
            binding: self.binding + 1,
        }
    }

    pub fn add_storage_texture(&self) -> Self {
        let mut entries = self.entries.clone();
        entries.push(wgpu::BindGroupLayoutEntry {
            binding: 2,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::StorageTexture {
                access: wgpu::StorageTextureAccess::WriteOnly,
                view_dimension: wgpu::TextureViewDimension::D2,
                format: wgpu::TextureFormat::Rgba32Float,
            },
            count: None,
        });
        BindGroupLayoutBuilder {
            entries,
            binding: self.binding + 1,
        }
    }

    pub fn add_uniform_buffer(&self, min_binding_size: Option<wgpu::BufferSize>) -> Self {
        let mut entries = self.entries.clone();
        entries.push(wgpu::BindGroupLayoutEntry {
            binding: self.binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size,
            },
            count: None,
        });
        BindGroupLayoutBuilder {
            entries,
            binding: self.binding + 1,
        }
    }

    pub fn add_storage_buffer(
        &self,
        min_binding_size: Option<wgpu::BufferSize>,
        read_only: bool,
    ) -> Self {
        let mut entries = self.entries.clone();
        entries.push(wgpu::BindGroupLayoutEntry {
            binding: self.binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size,
            },
            count: None,
        });
        BindGroupLayoutBuilder {
            entries,
            binding: self.binding + 1,
        }
    }

    pub fn create_bind_group_layout(
        &self,
        device: &Device,
        label: Option<&str>,
    ) -> BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &self.entries,
            label,
        })
    }
}

pub struct BindGroupBuilder<'a> {
    entries: Vec<wgpu::BindGroupEntry<'a>>,
    binding: u32,
}

impl<'a> BindGroupBuilder<'a> {
    pub fn new() -> Self {
        BindGroupBuilder {
            entries: Vec::new(),
            binding: 0,
        }
    }

    pub fn add_resource(&self, resource: wgpu::BindingResource<'a>) -> Self {
        let mut entries = self.entries.clone();
        entries.push(wgpu::BindGroupEntry {
            binding: self.binding,
            resource,
        });
        BindGroupBuilder {
            entries,
            binding: self.binding + 1,
        }
    }

    pub fn create_bind_group(
        &self,
        device: &Device,
        label: Option<&str>,
        layout: &BindGroupLayout,
    ) -> BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label,
            layout,
            entries: &self.entries,
        })
    }
}

pub fn create_buffer<'a>(
    device: &Device,
    contents: &'a [u8],
    usage: wgpu::BufferUsages,
    label: Option<&str>,
) -> Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label,
        contents,
        usage,
    })
}

pub fn create_texture(
    device: &Device,
    size: Extent3d,
    format: TextureFormat,
    usage: TextureUsages,
) -> Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage,
        label: None,
    })
}

pub fn compute_work_group_count(
    (width, height): (u32, u32),
    (workgroup_width, workgroup_height): (u32, u32),
) -> (u32, u32) {
    let x = (width + workgroup_width - 1) / workgroup_width;
    let y = (height + workgroup_height - 1) / workgroup_height;
    return (x, y);
}

#[cfg(feature = "savegif")]
pub fn copy_texture_to_buffer(
    command_encoder: &mut wgpu::CommandEncoder,
    texture: &wgpu::Texture,
    texture_size: Extent3d,
    buffer: &wgpu::Buffer,
) {
    let padded_bytes_per_row = padded_bytes_per_row(texture_size.width);
    command_encoder.copy_texture_to_buffer(
        wgpu::ImageCopyTexture {
            aspect: wgpu::TextureAspect::All,
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
        },
        wgpu::ImageCopyBuffer {
            buffer: &buffer,
            layout: wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: std::num::NonZeroU32::new(padded_bytes_per_row as u32),
                rows_per_image: std::num::NonZeroU32::new(texture_size.height),
            },
        },
        texture_size,
    );
}

#[cfg(feature = "savegif")]
pub fn save_gif(
    path: &str,
    frames: &mut Vec<Vec<u8>>,
    speed: i32,
    size: u16,
) -> anyhow::Result<()> {
    use gif::{Encoder, Frame, Repeat};

    let mut image = std::fs::File::create(path)?;
    let mut encoder = Encoder::new(&mut image, size, size, &[])?;
    encoder.set_repeat(Repeat::Infinite)?;

    for mut frame in frames {
        encoder.write_frame(&Frame::from_rgba_speed(size, size, &mut frame, speed))?;
    }

    Ok(())
}

#[cfg(feature = "savegif")]
pub fn padded_bytes_per_row(width: u32) -> usize {
    let bytes_per_row = width as usize * 4;
    let padding = (256 - bytes_per_row % 256) % 256;
    bytes_per_row + padding
}
