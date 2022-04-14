@stage(vertex)
fn main_vs(
    @location(0) particle_pos: vec2<f32>,
    @location(3) position: vec2<f32>,
) -> @builtin(position) vec4<f32> {
    return vec4<f32>(position + particle_pos, 0.0, 1.0);
}

@stage(fragment)
fn main_fs() -> @location(0) vec4<f32> {
    return vec4<f32>(0.0, 0.0, 0.0, 1.0);
}
