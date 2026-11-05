//! Location: // fgns_engine/tools/fgns_manifold_engine/src/scene_sheaf.rs
//! Description: Scene 3 — High-Contrast 5x5 Cellular Sheaf Stiefel Lattice V_32(R^128),
//!              Perona-Malik Edge-Stopping Barrier Shields (c_uv -> 0), Parallel-Transport
//!              Bridges (c_uv ~ 1), and OctoWave 8-Axis Laser Wavefront with Householder
//!              Specular Reflection (R_n = I - 2 n n^T).
//! Author: Vyacheslav Zimovsky (v.zimovsky.ai@gmail.com)
//! License: MIT-0

use crate::math::{arc_palette_color, TriadSO3, Vec3};
use crate::pbr::{Material, Scene};

pub fn build_scene_sheaf() -> Scene {
    let mut scene = Scene::new(-1.52, 0.68);

    // 5x5 High-Contrast 4-Chamber ARC Lattice Separated by Central Titanium Cross Wall (5)
    let grid: [[usize; 5]; 5] = [
        [3, 3, 5, 1, 1],
        [3, 3, 5, 1, 1],
        [5, 5, 5, 5, 5],
        [4, 4, 5, 8, 8],
        [4, 4, 5, 8, 8],
    ];

    let spacing = 0.68;
    let offset = 2.0 * spacing;
    let base_y = -1.18;

    let mut stalk_centers = [[Vec3::default(); 5]; 5];

    // 1. Sculpted 3D ARC Voxel Relief Base + Elevated Stiefel Stalks F(v) = V_32(R^128)
    for r in 0..5 {
        for c in 0..5 {
            let sym = grid[r][c];
            let col = arc_palette_color(sym);
            let x = (c as f64) * spacing - offset;
            let z = (r as f64) * spacing - offset;

            let is_wall = sym == 5;
            let tile_h = if is_wall { 0.36 } else { 0.22 };

            let base_col = if is_wall {
                Vec3::new(0.22, 0.26, 0.34)
            } else {
                col * 0.82
            };

            // Bevelled Voxel Plinth + Luminous Top Cap
            scene.add_box(
                Vec3::new(x - 0.30, base_y, z - 0.30),
                Vec3::new(x + 0.30, base_y + tile_h, z + 0.30),
                Material::solid(base_col, 0.72, 0.18),
            );
            scene.add_box(
                Vec3::new(x - 0.25, base_y + tile_h, z - 0.25),
                Vec3::new(x + 0.25, base_y + tile_h + 0.045, z + 0.25),
                Material::glowing(
                    if is_wall {
                        Vec3::new(0.45, 0.52, 0.64)
                    } else {
                        col
                    },
                    if is_wall { 0.10 } else { 0.36 },
                ),
            );

            let stalk_y = if is_wall { 0.04 } else { 0.44 };
            let stalk_pt = Vec3::new(x, stalk_y, z);
            stalk_centers[r][c] = stalk_pt;

            // Vertical Fiber Riser from Voxel to Stiefel Stalk
            let m_riser = Material::glowing(
                col.lerp(Vec3::new(0.70, 0.90, 1.0), 0.40),
                if is_wall { 0.08 } else { 0.28 },
            );
            scene.add_capsule(
                Vec3::new(x, base_y + tile_h + 0.045, z),
                stalk_pt,
                0.014,
                m_riser,
            );

            // Synchronized Stiefel Frame Triad O(3) inside each homogeneous chamber
            if !is_wall {
                let chamber_rot = match sym {
                    3 => TriadSO3::canonical().rotated(Vec3::new(0.2, 1.0, 0.1), 0.28),
                    1 => TriadSO3::canonical().rotated(Vec3::new(-0.3, 1.0, 0.4), -0.78),
                    4 => TriadSO3::canonical().rotated(Vec3::new(0.5, 0.8, -0.2), 0.96),
                    _ => TriadSO3::canonical().rotated(Vec3::new(-0.4, 0.6, 0.7), -1.15),
                };
                scene.add_triad(stalk_pt, chamber_rot, 0.22, 0.012);
            } else {
                scene.add_sphere(
                    stalk_pt,
                    0.046,
                    Material::solid(Vec3::new(0.68, 0.75, 0.86), 0.85, 0.16),
                );
            }
        }
    }

    // 2. Intra-Chamber Parallel Transport Bridges (c_uv ~ 1) vs. Edge-Stopping Barrier Shields (c_uv -> 0)
    let m_bridge = Material::glowing(Vec3::new(0.06, 0.88, 1.00), 0.62);
    let m_shield_glass = Material::glass(Vec3::new(1.00, 0.22, 0.06), 0.24, 0.34);
    let m_shield_rim = Material::glowing(Vec3::new(1.00, 0.54, 0.08), 0.78);

    for r in 0..5 {
        for c in 0..5 {
            let p = stalk_centers[r][c];
            let sym_p = grid[r][c];

            if c + 1 < 5 {
                let q = stalk_centers[r][c + 1];
                let sym_q = grid[r][c + 1];
                if sym_p == sym_q && sym_p != 5 {
                    scene.add_capsule(p, q, 0.022, m_bridge);
                } else if (sym_p == 5) != (sym_q == 5) && r != 2 {
                    let mid = (p + q) * 0.5;
                    scene.add_box(
                        Vec3::new(mid.x - 0.016, base_y + 0.26, mid.z - 0.27),
                        Vec3::new(mid.x + 0.016, 0.50, mid.z + 0.27),
                        m_shield_glass,
                    );
                    scene.add_capsule(
                        Vec3::new(mid.x, 0.50, mid.z - 0.27),
                        Vec3::new(mid.x, 0.50, mid.z + 0.27),
                        0.016,
                        m_shield_rim,
                    );
                }
            }

            if r + 1 < 5 {
                let q = stalk_centers[r + 1][c];
                let sym_q = grid[r + 1][c];
                if sym_p == sym_q && sym_p != 5 {
                    scene.add_capsule(p, q, 0.022, m_bridge);
                } else if (sym_p == 5) != (sym_q == 5) && c != 2 {
                    let mid = (p + q) * 0.5;
                    scene.add_box(
                        Vec3::new(mid.x - 0.27, base_y + 0.26, mid.z - 0.016),
                        Vec3::new(mid.x + 0.27, 0.50, mid.z + 0.016),
                        m_shield_glass,
                    );
                    scene.add_capsule(
                        Vec3::new(mid.x - 0.27, 0.50, mid.z),
                        Vec3::new(mid.x + 0.27, 0.50, mid.z),
                        0.016,
                        m_shield_rim,
                    );
                }
            }
        }
    }

    // 3. OctoWave 8-Axis Laser Wavefront + 45-Degree Householder Specular Reflection (R_n = I - 2 n n^T)
    let wave_y = 0.72;
    let p_seed = Vec3::new(stalk_centers[0][4].x, wave_y, stalk_centers[0][4].z);
    let p_hit = Vec3::new(
        (stalk_centers[1][2].x + stalk_centers[1][3].x) * 0.5 + 0.03,
        wave_y,
        (stalk_centers[0][3].z + stalk_centers[1][3].z) * 0.5 + 0.10,
    );
    let p_refl = Vec3::new(
        stalk_centers[1][4].x + 0.08,
        wave_y,
        stalk_centers[1][4].z + 0.22,
    );

    let m_laser_in = Material::glowing(Vec3::new(1.00, 0.86, 0.10), 0.90);
    let m_laser_out = Material::glowing(Vec3::new(0.12, 0.98, 0.72), 0.90);
    let m_octo_sub = Material::glowing(Vec3::new(1.00, 0.64, 0.14), 0.55);

    // 8-Axis OctoWave Starburst at Seed
    for k in 0..8 {
        let ang = (k as f64) * std::f64::consts::FRAC_PI_4;
        let dir = Vec3::new(ang.cos(), 0.0, ang.sin());
        scene.add_arrow(p_seed, p_seed + dir * 0.34, 0.013, 0.032, 0.095, m_octo_sub);
    }
    scene.add_sphere(p_seed, 0.078, m_laser_in);

    // Incident 45-deg Diagonal Wavefront Ray + Specularly Reflected Ray
    scene.add_arrow(p_seed, p_hit, 0.032, 0.078, 0.22, m_laser_in);
    scene.add_arrow(p_hit, p_refl, 0.032, 0.078, 0.22, m_laser_out);

    // Householder Boundary Normal Vector n and Specular Bounce Torus Halo at p_hit
    scene.add_sphere(p_hit, 0.072, Material::glowing(Vec3::new(1.0, 0.97, 0.85), 1.05));
    scene.add_torus_y(
        p_hit,
        0.14,
        0.016,
        Material::glowing(Vec3::new(1.0, 0.78, 0.18), 0.75),
    );
    let normal_vec = Vec3::new(1.0, 0.0, 0.0);
    scene.add_arrow(
        p_hit,
        p_hit + normal_vec * 0.48,
        0.020,
        0.052,
        0.15,
        Material::glowing(Vec3::new(1.00, 0.26, 0.74), 0.82),
    );

    // Secondary Coherent Wavefront in Chamber A (Top-Left Emerald Chamber)
    let p_a0 = Vec3::new(stalk_centers[0][0].x, wave_y, stalk_centers[0][0].z);
    let p_a1 = Vec3::new(stalk_centers[1][1].x, wave_y, stalk_centers[1][1].z);
    scene.add_sphere(p_a0, 0.064, m_laser_in);
    scene.add_arrow(p_a0, p_a1, 0.026, 0.064, 0.18, m_laser_out);

    scene
}
