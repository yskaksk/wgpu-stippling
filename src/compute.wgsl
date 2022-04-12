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
    scale: f32
};

@group(0) @binding(0) var<uniform> params : SimParams;
@group(0) @binding(1) var<storage, read> particlesSrc : array<Particle>;
@group(0) @binding(2) var<storage, read_write> particlesDst : array<Particle>;

@group(1) @binding(0) var t: texture_2d<f32>;

fn rotate(v: f32) -> f32 {
    if (v > 1.0) {
        return v - 2.0;
    } else if (v < -1.0) {
        return v + 2.0;
    } else {
        return v;
    }
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
  var acc : vec2<f32> = vec2<f32>(0.0, 0.0);

  let texture_size = textureDimensions(t);
  let texture_w = texture_size[0];
  let texture_h = texture_size[1];

  var ix : i32 = 0;
  loop {
    if (ix >= texture_w) {
      break;
    }
    var iy : i32 = 0;
    loop {
      if (iy >= texture_h) {
        break;
      }

      let texel = textureLoad(t, vec2<i32>(ix, iy), 0);
      let bmpQ = (params.blank_level - (texel[0] * 0.299 + texel[1] * 0.587 + texel[2] * 0.114)) * params.scale;

      let texpos = vec2<f32>(
        2.0 * f32(ix) / f32(texture_w) - 1.0,
        1.0 - 2.0 * f32(iy) / f32(texture_h)
      );

      var dp : vec2<f32> = texpos - vPos;
      dp[0] = rotate(dp[0]);
      dp[1] = rotate(dp[1]);
      let d2 = dot(dp, dp) + 0.00003;
      let d = sqrt(d2);

      if (abs(d) > Epsilon) {
          let q = bmpQ / d2;
          acc += q * dp / d;
      }

      continuing {
        iy = iy + 1;
      }
    }
    continuing {
      ix = ix + 1;
    }
  }

  var i : u32 = 0u;
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
  var vel : vec2<f32> = vVel;
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

  // Write back
  particlesDst[index] = Particle(pos, vel, acc);
}
