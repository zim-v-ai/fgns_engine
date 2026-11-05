//! Location: // fgns_engine/tools/fgns_manifold_engine/src/scene_bundle.rs
//! Description: Scene 2 — 3D Orthogonal Color-Geometry Fiber Bundle R^128 = H_color^(11) (+) H_geom^(117)
//!              with Precision Optical Base Platform, 11 Gram-Schmidt ARC Basis Spheres,
//!              Smooth Analytical SO(3)^8 Gauge Tori, and Contrasting Unsplit SO(3) Leakage Helix.
//! Author: Vyacheslav Zimovsky (v.zimovsky.ai@gmail.com)
//! License: MIT-0

use std::f64::consts::PI;

use crate::math::{arc_palette_color, TriadSO3, Vec3};
use crate::pbr::{Material, Scene};

pub fn build_scene_bundle() -> Scene {
    let mut scene = Scene::new(-1.55, 0.72);
    let base_y = -1.06;
    let ring_r = 1.50;

    // 1. Precision Dark-Sapphire Optical Platform for Color Subspace H_color^(11) (P_color = E^T E)
    scene.add_disc_y(
        Vec3::new(0.0, base_y - 0.03, 0.0),
        0.0,
        ring_r * 1.20,
        Material::solid(Vec3::new(0.04, 0.09, 0.18), 0.75, 0.18),
    );
    scene.add_disc_y(
        Vec3::new(0.0, base_y, 0.0),
        0.14,
        ring_r * 1.18,
        Material::glass(Vec3::new(0.08, 0.48, 0.96), 0.28, 0.14),
    );

    // Smooth Analytical Concentric Reference Rings on Base Platform
    scene.add_torus_y(
        Vec3::new(0.0, base_y + 0.005, 0.0),
        ring_r * 0.55,
        0.011,
        Material::glowing(Vec3::new(0.18, 0.74, 1.00), 0.35),
    );
    scene.add_torus_y(
        Vec3::new(0.0, base_y + 0.005, 0.0),
        ring_r,
        0.020,
        Material::glowing(Vec3::new(0.24, 0.84, 1.00), 0.45),
    );
    scene.add_torus_y(
        Vec3::new(0.0, base_y - 0.01, 0.0),
        ring_r * 1.19,
        0.024,
        Material::solid(Vec3::new(0.88, 0.72, 0.24), 0.88, 0.15),
    );

    // Central Gram-Schmidt Orthogonal Projector Hub & Vertical Normal Axis P_geom = I - E^T E
    let center_hub = Vec3::new(0.0, base_y, 0.0);
    scene.add_sphere(
        center_hub,
        0.145,
        Material::solid(Vec3::new(0.95, 0.97, 1.0), 0.92, 0.09),
    );
    scene.add_arrow(
        center_hub,
        Vec3::new(0.0, 1.48, 0.0),
        0.026,
        0.066,
        0.20,
        Material::glowing(Vec3::new(0.28, 0.88, 1.0), 0.52),
    );

    // 2. 11 Orthonormal ARC Palette Basis Spheres + Vertical Geometric Fiber Columns H_geom^(117)
    for k in 0..11 {
        let phi = (k as f64 / 11.0) * 2.0 * PI + 0.18;
        let col = arc_palette_color(k);
        let base_pt = Vec3::new(ring_r * phi.cos(), base_y, ring_r * phi.sin());
        let top_pt = Vec3::new(ring_r * phi.cos(), 1.24, ring_r * phi.sin());

        // Radial Gram-Schmidt Spoke from Origin Hub to Basis Sphere e_c
        let m_spoke = Material::glowing(col.lerp(Vec3::new(0.65, 0.88, 1.0), 0.35), 0.32);
        scene.add_capsule(center_hub, base_pt, 0.014, m_spoke);

        // Crystalline ARC Palette Basis Sphere on H_color^(11) + Golden Pedestal Ring
        scene.add_sphere(base_pt, 0.135, Material::glowing(col, 0.42));
        scene.add_torus_y(
            Vec3::new(base_pt.x, base_y + 0.01, base_pt.z),
            0.165,
            0.014,
            Material::solid(Vec3::new(0.95, 0.80, 0.25), 0.88, 0.14),
        );

        // Vertical Geometric Fiber Column H_geom^(117) Strictly Orthogonal to Base
        let m_fiber = Material::solid(
            col.lerp(Vec3::new(0.84, 0.92, 1.0), 0.50),
            0.78,
            0.16,
        );
        scene.add_capsule(base_pt, top_pt, 0.022, m_fiber);

        // Top Fiber Cap Node
        scene.add_sphere(top_pt, 0.058, Material::glowing(col, 0.38));
    }

    // 3. Three Smooth Analytical Horizontal SO(3)^8 Gauge Transport Tori (Zero Color Drift: dy = 0)
    let gauge_levels = [-0.34, 0.36, 1.04];
    let gauge_colors = [
        Vec3::new(0.06, 0.84, 1.00),
        Vec3::new(0.10, 0.94, 0.58),
        Vec3::new(0.32, 0.70, 1.00),
    ];

    for (idx, &gy) in gauge_levels.iter().enumerate() {
        let m_torus = Material::glowing(gauge_colors[idx], 0.52);
        scene.add_torus_y(Vec3::new(0.0, gy, 0.0), ring_r, 0.028, m_torus);

        // Directional Golden Gauge Flow Cones & Transported 3D Orthonormal Triads
        for t_idx in 0..4 {
            let ang = (t_idx as f64 / 4.0) * 2.0 * PI + (idx as f64) * 0.48;
            let pos = Vec3::new(ring_r * ang.cos(), gy, ring_r * ang.sin());
            let tangent = Vec3::new(-ang.sin(), 0.0, ang.cos());

            scene.add_cone(
                pos - tangent * 0.07,
                pos + tangent * 0.11,
                0.064,
                Material::glowing(Vec3::new(1.00, 0.84, 0.16), 0.65),
            );

            let triad = TriadSO3::canonical()
                .rotated(Vec3::new(0.0, 1.0, 0.0), ang)
                .rotated(Vec3::new(1.0, 0.2, 0.4), (idx as f64 + 1.0) * 0.52);
            scene.add_triad(pos + tangent * 0.24, triad, 0.24, 0.013);
        }
    }

    // 4. Contrasting Unsplit SO(3) Color-Leakage Trajectory (Crimson Tilted Orbit Piercing Levels)
    let m_leak = Material::glowing(Vec3::new(1.00, 0.14, 0.24), 0.72);
    let m_warn = Material::glowing(Vec3::new(1.00, 0.76, 0.08), 0.78);
    let n_leak = 72;
    for i in 0..n_leak {
        let f0 = i as f64 / n_leak as f64;
        let f1 = (i + 1) as f64 / n_leak as f64;
        let a0 = f0 * 2.0 * PI;
        let a1 = f1 * 2.0 * PI;
        let r_l = ring_r * 0.66;
        let y0 = 0.20 + 0.76 * a0.sin() + 0.16 * (a0 * 3.0).cos();
        let y1 = 0.20 + 0.76 * a1.sin() + 0.16 * (a1 * 3.0).cos();
        let p0 = Vec3::new(r_l * a0.cos(), y0, r_l * a0.sin());
        let p1 = Vec3::new(r_l * a1.cos(), y1, r_l * a1.sin());
        scene.add_capsule(p0, p1, 0.028, m_leak);

        if i % 18 == 9 {
            scene.add_sphere(p0, 0.068, m_warn);
        }
    }

    scene
}
