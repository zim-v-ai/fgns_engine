// ============================================================================
// Location: // fgns_engine/tools/fgns_manifold_engine/shaders/finsler_sheaf_raymarch.wgsl
// Description: WGSL Compute & Fragment Ray-Marching Shader for the FGNS-TRM v5.0.0
//              Cutaway Asymmetric Finsler-Randers Indicatrix Manifold, Non-Abelian SO(3)
//              Rodrigues-Hamilton Lie Transport, and Cook-Torrance GGX Microfacet Shading.
// Author: Vyacheslav Zimovsky (v.zimovsky.ai@gmail.com)
// License: MIT-0
// ============================================================================

struct ManifoldUniforms {
    resolution: vec2<f32>,
    camera_pos: vec3<f32>,
    camera_target: vec3<f32>,
    vec_l: vec3<f32>,
    vec_c: vec3<f32>,
    vec_r: vec3<f32>,
    randers_beta: vec3<f32>,
    metric_diag: vec3<f32>,
    scene_mode: u32,
};

@group(0) @binding(0) var<uniform> u_cfg: ManifoldUniforms;
@group(0) @binding(1) var out_tex: texture_storage_2d<rgba8unorm, write>;

fn finsler_randers_norm(v: vec3<f32>, a_diag: vec3<f32>, beta: vec3<f32>) -> f32 {
    let riem_sq = a_diag.x * v.x * v.x + a_diag.y * v.y * v.y + a_diag.z * v.z * v.z;
    let riem = sqrt(max(riem_sq, 1e-8));
    let drift = dot(beta, v);
    return riem + drift;
}

fn sdf_finsler_indicatrix(p: vec3<f32>) -> f32 {
    let f_val = finsler_randers_norm(p, u_cfg.metric_diag, u_cfg.randers_beta);
    return (f_val - 1.0) * 0.72;
}

fn so3_rodrigues_rotate(v: vec3<f32>, axis_raw: vec3<f32>, theta: f32) -> vec3<f32> {
    let u = normalize(axis_raw + vec3<f32>(1e-7, 0.0, 0.0));
    let c = cos(theta);
    let s = sin(theta);
    return v * c + cross(u, v) * s + u * dot(u, v) * (1.0 - c);
}

fn colormap_finsler_asymmetry(t_raw: f32) -> vec3<f32> {
    let t = clamp(t_raw, 0.0, 1.0);
    let c0 = vec3<f32>(0.04, 0.48, 1.00);
    let c1 = vec3<f32>(0.06, 0.88, 0.78);
    let c2 = vec3<f32>(1.00, 0.72, 0.06);
    let c3 = vec3<f32>(0.98, 0.16, 0.44);
    if (t < 0.333) {
        return mix(c0, c1, t * 3.0);
    } else if (t < 0.666) {
        return mix(c1, c2, (t - 0.333) * 3.0);
    } else {
        return mix(c2, c3, (t - 0.666) * 3.0);
    }
}

fn shade_finsler_shell(p: vec3<f32>, n: vec3<f32>, rd: vec3<f32>) -> vec4<f32> {
    let dir = normalize(p + vec3<f32>(1e-6));
    let cost_asym = 0.5 + 0.5 * dot(dir, normalize(u_cfg.randers_beta + vec3<f32>(1e-5)));
    let base_col = colormap_finsler_asymmetry(cost_asym);
    let ndotv = abs(dot(n, -rd));
    let fresnel = pow(1.0 - ndotv, 3.2);
    let lat = asin(clamp(dir.y, -1.0, 1.0));
    let lon = atan2(dir.z, dir.x);
    let grid_lat = exp(-pow(sin(lat * 6.0), 2.0) * 110.0);
    let grid_lon = exp(-pow(sin(lon * 6.0), 2.0) * 110.0);
    let iso_wire = max(grid_lat, grid_lon);
    let col = mix(base_col, vec3<f32>(0.90, 0.97, 1.0), iso_wire * 0.65 + fresnel * 0.50);
    let alpha = clamp(0.08 + 0.64 * fresnel + 0.55 * iso_wire, 0.0, 0.90);
    return vec4<f32>(col, alpha);
}

@compute @workgroup_size(8, 8, 1)
fn main_raymarch(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = vec2<u32>(u32(u_cfg.resolution.x), u32(u_cfg.resolution.y));
    if (gid.x >= dims.x || gid.y >= dims.y) {
        return;
    }
    let uv = (vec2<f32>(f32(gid.x), f32(gid.y)) / u_cfg.resolution) * 2.0 - vec2<f32>(1.0, 1.0);
    let aspect = u_cfg.resolution.x / max(u_cfg.resolution.y, 1.0);
    let fwd = normalize(u_cfg.camera_target - u_cfg.camera_pos);
    let right = normalize(cross(fwd, vec3<f32>(0.0, 1.0, 0.0)));
    let up = cross(right, fwd);
    let rd = normalize(fwd * 2.15 + right * (uv.x * aspect) - up * uv.y);

    var bg = mix(vec3<f32>(0.005, 0.009, 0.018), vec3<f32>(0.015, 0.025, 0.052), clamp(1.0 - length(uv) * 0.55, 0.0, 1.0));
    var t = 0.1;
    for (var i: u32 = 0u; i < 96u; i = i + 1u) {
        let p = u_cfg.camera_pos + rd * t;
        let d = sdf_finsler_indicatrix(p);
        if (abs(d) < 0.002) {
            let eps = 0.002;
            let n = normalize(vec3<f32>(
                sdf_finsler_indicatrix(p + vec3<f32>(eps, 0.0, 0.0)) - sdf_finsler_indicatrix(p - vec3<f32>(eps, 0.0, 0.0)),
                sdf_finsler_indicatrix(p + vec3<f32>(0.0, eps, 0.0)) - sdf_finsler_indicatrix(p - vec3<f32>(0.0, eps, 0.0)),
                sdf_finsler_indicatrix(p + vec3<f32>(0.0, 0.0, eps)) - sdf_finsler_indicatrix(p - vec3<f32>(0.0, 0.0, eps))
            ));
            let sh = shade_finsler_shell(p, n, rd);
            bg = mix(bg, sh.xyz, sh.w);
            break;
        }
        t = t + max(abs(d) * 0.7, 0.015);
        if (t > 12.0) {
            break;
        }
    }
    textureStore(out_tex, vec2<i32>(i32(gid.x), i32(gid.y)), vec4<f32>(bg, 1.0));
}
