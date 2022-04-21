struct Particle {
  pos : vec2<f32>,
  vel : vec2<f32>,
  acc : vec2<f32>,
};

struct SimParams {
    q_charge: f32,
    blank_level: f32,
    time_delta: f32,
    d_max: f32,
    sustain: f32,
    dt: f32,
    dt_2: f32,
    dtt: f32,
    v_max: f32,
    fc: f32,
    pi: f32
};

@group(0) @binding(0) var<uniform> params : SimParams;

@group(1) @binding(0) var<storage, read> particlesSrc : array<Particle>;
@group(1) @binding(1) var<storage, read_write> particlesDst : array<Particle>;

@group(2) @binding(0) var t: texture_2d<f32>;
@group(2) @binding(1) var t2: texture_2d<f32>;

fn rotate(v: f32) -> f32 {
    if (v > 1.0) {
        return v - 2.0;
    } else if (v < -1.0) {
        return v + 2.0;
    } else {
        return v;
    }
}

fn get_texel(texture: texture_2d<f32>, x: f32, y: f32) -> vec4<f32> {
    let texture_size = textureDimensions(texture);
    let width = texture_size[0];
    let height = texture_size[1];
    let ix = i32(floor((x + 1.0) * f32(width) / 2.0));
    let iy = i32(floor((1.0 - y) * f32(height) / 2.0));
    return textureLoad(texture, vec2<i32>(ix, iy), 0);
}

@stage(compute)
@workgroup_size(64)
fn main(@builtin(global_invocation_id) global_invocation_id: vec3<u32>) {
  let total = arrayLength(&particlesSrc);
  let index = global_invocation_id.x;
  if (index >= total) {
    return;
  }

  var vPos : vec2<f32> = particlesSrc[index].pos;
  var vVel : vec2<f32> = particlesSrc[index].vel;
  var vAcc : vec2<f32> = particlesSrc[index].acc;

  let Epsilon = 0.0000001;

  let texel1 = get_texel(t, vPos.x, vPos.y);
  let texel2 = get_texel(t2, vPos.x, vPos.y);

  var acc : vec2<f32> = vec2<f32>(0.0, 0.0);
  let theta = params.fc * params.pi / 360.0;
  if (vPos.x > cos(theta)) {
    acc = vec2<f32>(texel1[0], texel1[1]);
  } else {
    acc = vec2<f32>(texel2[0], texel2[1]);
  }


  var i: u32 = 0u;
  loop {
      if (i >= total) {
          break;
      }
      if (i == index) {
          continue;
      }
      let dotPos = particlesSrc[i].pos;
      var dp : vec2<f32> = dotPos - vPos;
      dp.x = rotate(dp.x);
      dp.y = rotate(dp.y);

      let d2 = dot(dp, dp) + 0.00003;
      let d = sqrt(d2);
      if (abs(d) > Epsilon) {
          let q = params.q_charge / d2;
          acc -= q * dp / d;
      }
      continuing {
          i = i + 1u;
      }
  }

  acc *= params.q_charge;

  var vel : vec2<f32> = (params.sustain * vVel) + params.dt_2 * (vAcc + acc);
  let vlen = length(vel);
  if (vlen > params.v_max) {
      vel *= params.v_max / vlen;
  }

  var dp : vec2<f32> = (vel * params.dt) + (0.5 * acc * params.dtt);
  let dlen = length(dp);
  if (dlen > params.d_max) {
      dp *= params.d_max / dlen;
  }
  var pos : vec2<f32> = vPos + dp;

  pos.x = rotate(pos.x);
  pos.y = rotate(pos.y);

  particlesDst[index] = Particle(pos, vel, acc);
}
