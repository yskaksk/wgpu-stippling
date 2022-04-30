struct Params {
    blank_level: f32,
    dot_total_charge: f32
};

@group(0) @binding(0) var<uniform> params : Params;
@group(0) @binding(1) var input_texture : texture_2d<f32>;
@group(0) @binding(2) var output_texture : texture_storage_2d<rgba32float, write>;

fn rotate(v: f32) -> f32 {
    if (v > 1.0) {
        return v - 2.0;
    } else if (v < -1.0) {
        return v + 2.0;
    } else {
        return v;
    }
}

fn coord_to_pos(x: i32, y: i32, width: i32, height: i32) -> vec2<f32> {
    let pos = vec2<f32>(
        2.0 * f32(x) / f32(width) - 1.0,
        1.0 - 2.0 * f32(y) / f32(height)
    );
    return pos;
}

@stage(compute)
@workgroup_size(16, 16)
fn main(
    @builtin(global_invocation_id) global_invocation_id: vec3<u32>
) {
    let dimensions = textureDimensions(input_texture);
    let width = dimensions.x;
    let height = dimensions.y;
    let coords = vec2<i32>(global_invocation_id.xy);
    let Epsilon = 0.0000001;

    if (coords.x >= width || coords.y >= height) {
        return;
    }

    let pos = coord_to_pos(coords.x, coords.y, width, height);

    var acc : vec2<f32> = vec2<f32>(0.0, 0.0);
    var total_q : f32 = 0.0;
    var ix: i32 = 0;
    loop {
        if (ix >= width) {
            break;
        }
        var iy: i32 = 0;
        loop {
            if (iy >= height) {
                break;
            }
            let texel = textureLoad(input_texture, vec2<i32>(ix, iy), 0);
            let gray = texel[0];
            let bmpQ = params.blank_level - gray;
            total_q += bmpQ;

            let texpos = coord_to_pos(ix, iy, width, height);
            var dp : vec2<f32> = texpos - pos;
            dp.x = rotate(dp.x);
            dp.y = rotate(dp.y);

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
    acc *= params.dot_total_charge / total_q;
    textureStore(output_texture, coords.xy, vec4<f32>(acc, f32(coords.x), f32(coords.y)));
}
