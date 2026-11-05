//! Location: // fgns_engine/tools/fgns_manifold_engine/src/main.rs
//! Description: FGNS-TRM v5.0.0 Multi-Module Rust 3D Differential-Geometric Ray-Tracing
//!              & WGSL Shader Engine (1600x1200, 9x Stratified Supersampling):
//!              (1) NA3D Asymmetric Finsler-Randers Indicatrix & SO(3) Wilson Holonomy
//!              (2) Orthogonal Fiber Bundle H_color^(11) (+) H_geom^(117) & Smooth Gauge Tori
//!              (3) Cellular Sheaf Stiefel Lattice, Perona-Malik Shields & OctoWave Bounce
//! Author: Vyacheslav Zimovsky (v.zimovsky.ai@gmail.com)
//! License: MIT-0

pub mod math;
pub mod pbr;
pub mod scene_bundle;
pub mod scene_na3d;
pub mod scene_sheaf;

use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::Path;

use crate::math::Vec3;
use crate::scene_bundle::build_scene_bundle;
use crate::scene_na3d::{build_na3d_metric, build_scene_na3d, trace_scene_na3d};
use crate::scene_sheaf::build_scene_sheaf;

pub const WGSL_SHADER_SOURCE: &str = include_str!("../shaders/finsler_sheaf_raymarch.wgsl");

/// High-Contrast ACES Filmic Tone Mapping + sRGB Gamma Correction
#[inline]
fn aces_tonemap_srgb(hdr: Vec3) -> (u8, u8, u8) {
    let map_ch = |x: f64| -> u8 {
        let v = (x * 1.05).max(0.0).powf(1.06);
        let mapped = (v * (2.51 * v + 0.03)) / (v * (2.43 * v + 0.59) + 0.14);
        let gamma = mapped.clamp(0.0, 1.0).powf(1.0 / 2.2);
        (gamma * 255.0 + 0.5).clamp(0.0, 255.0) as u8
    };
    (map_ch(hdr.x), map_ch(hdr.y), map_ch(hdr.z))
}

fn write_ppm(path: &Path, width: usize, height: usize, pixels: &[u8]) -> std::io::Result<()> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);
    write!(writer, "P6\n{} {}\n255\n", width, height)?;
    writer.write_all(pixels)?;
    writer.flush()
}

fn render_camera<F>(
    width: usize,
    height: usize,
    cam_pos: Vec3,
    look_at: Vec3,
    fov_scale: f64,
    trace_fn: F,
) -> Vec<u8>
where
    F: Fn(Vec3, Vec3) -> Vec3 + Sync,
{
    let fwd = (look_at - cam_pos).normalized();
    let right = fwd.cross(Vec3::new(0.0, 1.0, 0.0)).normalized();
    let up = right.cross(fwd).normalized();
    let aspect = (width as f64) / (height as f64);

    // 9x Stratified 3x3 Supersampling Grid for Razor-Sharp Vector Edges
    let sub_offsets = [
        (0.1667, 0.1667),
        (0.5000, 0.1667),
        (0.8333, 0.1667),
        (0.1667, 0.5000),
        (0.5000, 0.5000),
        (0.8333, 0.5000),
        (0.1667, 0.8333),
        (0.5000, 0.8333),
        (0.8333, 0.8333),
    ];
    let inv_samples = 1.0 / (sub_offsets.len() as f64);

    let mut pixels = vec![0u8; width * height * 3];
    let n_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(8)
        .max(1);
    let rows_per_chunk = (height + n_threads - 1) / n_threads;
    let row_bytes = width * 3;

    std::thread::scope(|s| {
        for (chunk_idx, chunk_slice) in pixels.chunks_mut(rows_per_chunk * row_bytes).enumerate() {
            let y_start = chunk_idx * rows_per_chunk;
            let trace_ref = &trace_fn;
            s.spawn(move || {
                let chunk_rows = chunk_slice.len() / row_bytes;
                for local_y in 0..chunk_rows {
                    let y = y_start + local_y;
                    for x in 0..width {
                        let mut accum = Vec3::new(0.0, 0.0, 0.0);
                        for &(ox, oy) in &sub_offsets {
                            let u = ((x as f64 + ox) / (width as f64) * 2.0 - 1.0)
                                * aspect
                                * fov_scale;
                            let v = (1.0 - (y as f64 + oy) / (height as f64) * 2.0) * fov_scale;
                            let rd = (fwd + right * u + up * v).normalized();
                            accum = accum + trace_ref(cam_pos, rd);
                        }
                        let hdr = accum * inv_samples;
                        let (r, g, b) = aces_tonemap_srgb(hdr);
                        let idx = (local_y * width + x) * 3;
                        chunk_slice[idx] = r;
                        chunk_slice[idx + 1] = g;
                        chunk_slice[idx + 2] = b;
                    }
                }
            });
        }
    });

    pixels
}

fn main() -> std::io::Result<()> {
    let out_dir = Path::new("../../artifacts/rust_renders");
    fs::create_dir_all(out_dir)?;

    println!(
        "FGNS Manifold Engine v5.0 | Cook-Torrance GGX + Exact Analytical Intersectors ({} bytes WGSL)",
        WGSL_SHADER_SOURCE.len()
    );

    let width = 1600;
    let height = 1200;

    let ppm1 = out_dir.join("scene1_na3d_finsler.ppm");
    if !ppm1.exists() {
        println!(
            "[1/4] Ray-Tracing Scene 1: Cutaway NA3D Finsler-Randers Indicatrix & SO(3) Wilson Plaquette ({}x{}, 9x SSAA)...",
            width, height
        );
        let metric = build_na3d_metric();
        let scene1 = build_scene_na3d(&metric);
        let cam1 = Vec3::new(1.85, 1.18, 3.72);
        let look1 = Vec3::new(-0.06, -0.02, 0.0);
        let buf1 = render_camera(width, height, cam1, look1, 0.44, |ro, rd| {
            trace_scene_na3d(ro, rd, &scene1, &metric)
        });
        write_ppm(&ppm1, width, height, &buf1)?;
    }

    let ppm2 = out_dir.join("scene2_orthogonal_bundle.ppm");
    if !ppm2.exists() {
        println!(
            "[2/4] Ray-Tracing Scene 2: 3D Orthogonal Fiber Bundle H_color^(11) (+) H_geom^(117) & Smooth Gauge Tori ({}x{}, 9x SSAA)...",
            width, height
        );
        let scene2 = build_scene_bundle();
        let cam2 = Vec3::new(2.58, 1.56, 3.58);
        let look2 = Vec3::new(0.0, -0.06, 0.0);
        let buf2 = render_camera(width, height, cam2, look2, 0.44, |ro, rd| {
            scene2.evaluate_ray(ro, rd)
        });
        write_ppm(&ppm2, width, height, &buf2)?;
    }

    let ppm3 = out_dir.join("scene3_sheaf_lattice.ppm");
    if !ppm3.exists() {
        println!(
            "[3/3] Ray-Tracing Scene 3: 5x5 Cellular Sheaf Stiefel Lattice, Perona-Malik Shields & OctoWave Bounce ({}x{}, 9x SSAA)...",
            width, height
        );
        let scene3 = build_scene_sheaf();
        let cam3 = Vec3::new(2.32, 2.38, 3.38);
        let look3 = Vec3::new(0.0, -0.20, 0.04);
        let buf3 = render_camera(width, height, cam3, look3, 0.44, |ro, rd| {
            scene3.evaluate_ray(ro, rd)
        });
        write_ppm(&ppm3, width, height, &buf3)?;
    }

    println!("All 3 HDR GGX Ray-Traced manifold renders written to artifacts/rust_renders/");
    Ok(())
}
