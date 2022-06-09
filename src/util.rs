use wgpu::{
    BindGroup, BindGroupLayout, BindGroupLayoutEntry, ComputePipeline, Device, RenderPipeline,
    ShaderModule,
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
            targets: &[config.format.into()],
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
