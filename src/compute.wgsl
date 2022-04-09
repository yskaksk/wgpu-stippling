struct Particle {
  pos : vec2<f32>,
  vel : vec2<f32>,
};
struct SimParams {
  deltaT : f32,
  beta : f32,
};

@group(0) @binding(0) var<uniform> params : SimParams;
@group(0) @binding(1) var<storage, read> particleSrc : array<Particle>;
@group(0) @binding(2) var<storage, read_write> particleDst : array<Particle>;
@group(0) @binding(3) var<uniform> image : array<array<f32,128>,128>;

@stage(compute)
@workgroup_size(64)
fn main(@builtin(global_invocation_id) global_invocation_id: vec3<u32>) {
}
