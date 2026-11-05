//! Location: // fgns_engine/tools/fgns_manifold_engine/src/scene_na3d.rs
//! Description: Scene 1 — High-Contrast Cutaway Finsler-Randers Indicatrix Manifold,
//!              Inner Riemannian Reference Equator, Volumetric 3D Chiral Simplex (L, C, R),
//!              Randers Drift Wavefront Rings, and Non-Abelian SO(3) Wilson Plaquette Holonomy.
//! Author: Vyacheslav Zimovsky (v.zimovsky.ai@gmail.com)
//! License: MIT-0

use std::f64::consts::PI;

use crate::math::{colormap_finsler_cost, FinslerRandersMetric, TriadSO3, Vec3};
use crate::pbr::{Material, Scene};

pub fn build_na3d_metric() -> FinslerRandersMetric {
    FinslerRandersMetric {
        a_diag: Vec3::new(1.04, 0.92, 1.08),
        off_xy: 0.09,
        off_yz: -0.05,
        beta: Vec3::new(0.36, 0.18, -0.12),
    }
}

pub fn build_scene_na3d(metric: &FinslerRandersMetric) -> Scene {
    let mut scene = Scene::new(-1.58, 0.72);
    let shell_scale = 1.24;

    // 1. Inner Riemannian Reference Equator Ring F_0(v) = sqrt(v^T A v) = 1 (beta = 0)
    let m_riem_cage = Material::glowing(Vec3::new(0.55, 0.78, 1.00), 0.28);
    for &lat in &[-0.28 * PI, 0.0, 0.28 * PI] {
        let n_seg = 48;
        for i in 0..n_seg {
            if i % 2 == 1 {
                continue;
            }
            let p0 = (i as f64 / n_seg as f64) * 2.0 * PI;
            let p1 = ((i + 1) as f64 / n_seg as f64) * 2.0 * PI;
            let d0 = Vec3::new(lat.cos() * p0.cos(), lat.sin(), lat.cos() * p0.sin());
            let d1 = Vec3::new(lat.cos() * p1.cos(), lat.sin(), lat.cos() * p1.sin());
            let q0 = d0 * (shell_scale * 0.74 / metric.riemannian_only(d0));
            let q1 = d1 * (shell_scale * 0.74 / metric.riemannian_only(d1));
            scene.add_capsule(q0, q1, 0.011, m_riem_cage);
        }
    }

    // 2. Central Platinum-Titanium Origin Hub
    let origin = Vec3::new(0.0, 0.0, 0.0);
    scene.add_sphere(
        origin,
        0.095,
        Material::solid(Vec3::new(0.96, 0.98, 1.0), 0.90, 0.10),
    );

    // 3. Well-Separated Chiral 3-Vector Simplex (L, C, R) & Randers Drift Vector (beta)
    let v_l = Vec3::new(-1.02, 0.58, 0.38);
    let v_c = Vec3::new(-0.12, 1.10, 0.44);
    let v_r = Vec3::new(0.94, -0.34, 0.56);
    let v_beta = Vec3::new(-1.26, -0.56, 0.34);

    let m_l = Material::glowing(Vec3::new(0.06, 0.52, 1.00), 0.52);
    let m_c = Material::glowing(Vec3::new(0.06, 0.92, 0.46), 0.52);
    let m_r = Material::glowing(Vec3::new(1.00, 0.16, 0.30), 0.52);
    let m_beta = Material::glowing(Vec3::new(1.00, 0.78, 0.06), 0.64);

    scene.add_arrow(origin, v_l, 0.040, 0.094, 0.25, m_l);
    scene.add_arrow(origin, v_c, 0.040, 0.094, 0.25, m_c);
    scene.add_arrow(origin, v_r, 0.040, 0.094, 0.25, m_r);
    scene.add_arrow(origin, v_beta, 0.044, 0.105, 0.27, m_beta);

    // Concentric Wavefront Rings around the Randers Drift Axis beta
    let beta_triad = TriadSO3::from_normal(v_beta);
    for &(frac, r_ring) in &[(0.42, 0.16), (0.66, 0.24)] {
        let ring_c = v_beta * frac;
        let n_seg = 32;
        for i in 0..n_seg {
            let a0 = (i as f64 / n_seg as f64) * 2.0 * PI;
            let a1 = ((i + 1) as f64 / n_seg as f64) * 2.0 * PI;
            let p0 = ring_c + beta_triad.e1 * (r_ring * a0.cos()) + beta_triad.e3 * (r_ring * a0.sin());
            let p1 = ring_c + beta_triad.e1 * (r_ring * a1.cos()) + beta_triad.e3 * (r_ring * a1.sin());
            scene.add_capsule(p0, p1, 0.010, m_beta);
        }
    }

    // Translucent Chiral Simplex Facets V = det[L, C, R] + Glowing Golden Simplex Frame
    let m_simplex_glass = Material::glass(Vec3::new(0.10, 0.92, 0.72), 0.34, 0.22);
    let m_simplex_base = Material::glass(Vec3::new(0.16, 0.64, 1.00), 0.24, 0.14);
    scene.add_triangle(v_l, v_c, v_r, m_simplex_glass);
    scene.add_triangle(origin, v_l, v_c, m_simplex_base);
    scene.add_triangle(origin, v_c, v_r, m_simplex_base);
    scene.add_triangle(origin, v_r, v_l, m_simplex_base);

    let m_edge = Material::glowing(Vec3::new(1.00, 0.86, 0.22), 0.48);
    scene.add_capsule(v_l, v_c, 0.020, m_edge);
    scene.add_capsule(v_c, v_r, 0.020, m_edge);
    scene.add_capsule(v_r, v_l, 0.020, m_edge);
    scene.add_sphere(v_l, 0.058, m_l);
    scene.add_sphere(v_c, 0.058, m_c);
    scene.add_sphere(v_r, 0.058, m_r);

    // 4. Outer Finsler-Randers Indicatrix Geodesic Wireframe Ribs
    let lats = [-0.56 * PI * 0.5, -0.28 * PI * 0.5, 0.0, 0.28 * PI * 0.5, 0.56 * PI * 0.5];
    for &lat in &lats {
        let n_seg = 64;
        for i in 0..n_seg {
            let p0 = (i as f64 / n_seg as f64) * 2.0 * PI;
            let p1 = ((i + 1) as f64 / n_seg as f64) * 2.0 * PI;
            let d0 = Vec3::new(lat.cos() * p0.cos(), lat.sin(), lat.cos() * p0.sin());
            let d1 = Vec3::new(lat.cos() * p1.cos(), lat.sin(), lat.cos() * p1.sin());
            let q0 = metric.indicatrix_point(d0, shell_scale);
            let q1 = metric.indicatrix_point(d1, shell_scale);
            let cost_t = ((metric.evaluate(d0) - 0.65) / 0.82).clamp(0.0, 1.0);
            let col = colormap_finsler_cost(cost_t);
            let r_rib = if lat.abs() < 1e-3 { 0.014 } else { 0.0095 };
            scene.add_capsule(q0, q1, r_rib, Material::glowing(col, 0.28));
        }
    }

    let n_meridians = 10;
    for m in 0..n_meridians {
        let phi = (m as f64 / n_meridians as f64) * 2.0 * PI;
        let n_seg = 48;
        for i in 0..n_seg {
            let t0 = -0.46 * PI + (i as f64 / n_seg as f64) * 0.92 * PI;
            let t1 = -0.46 * PI + ((i + 1) as f64 / n_seg as f64) * 0.92 * PI;
            let d0 = Vec3::new(t0.cos() * phi.cos(), t0.sin(), t0.cos() * phi.sin());
            let d1 = Vec3::new(t1.cos() * phi.cos(), t1.sin(), t1.cos() * phi.sin());
            let q0 = metric.indicatrix_point(d0, shell_scale);
            let q1 = metric.indicatrix_point(d1, shell_scale);
            let cost_t = ((metric.evaluate(d0) - 0.65) / 0.82).clamp(0.0, 1.0);
            let col = colormap_finsler_cost(cost_t);
            scene.add_capsule(q0, q1, 0.009, Material::glowing(col, 0.22));
        }
    }

    // 5. Non-Abelian SO(3) Wilson Plaquette Loop Positioned Cleanly in Top-Right Dome Sector
    let plaq_dirs = [
        Vec3::new(0.52, 0.36, 0.77).normalized(),
        Vec3::new(0.84, 0.32, 0.44).normalized(),
        Vec3::new(0.78, 0.60, 0.16).normalized(),
        Vec3::new(0.44, 0.66, 0.61).normalized(),
        Vec3::new(0.50, 0.48, 0.72).normalized(),
    ];

    let mut plaq_pts = Vec::with_capacity(5);
    for &d in &plaq_dirs {
        plaq_pts.push(metric.indicatrix_point(d, shell_scale * 1.045));
    }

    let m_plaq = Material::glowing(Vec3::new(1.00, 0.84, 0.12), 0.68);
    for k in 0..4 {
        let a = plaq_pts[k];
        let b = plaq_pts[k + 1];
        let sub = 12;
        for s in 0..sub {
            let f0 = s as f64 / sub as f64;
            let f1 = (s + 1) as f64 / sub as f64;
            let d0 = a.lerp(b, f0).normalized();
            let d1 = a.lerp(b, f1).normalized();
            let p0 = metric.indicatrix_point(d0, shell_scale * 1.045);
            let p1 = metric.indicatrix_point(d1, shell_scale * 1.045);
            scene.add_capsule(p0, p1, 0.025, m_plaq);
        }
    }

    // Transport 3D Orthonormal Triads around the Wilson Plaquette via exact SO(3) Rodrigues map
    let mut triad = TriadSO3::canonical().rotated(Vec3::new(0.2, 1.0, -0.3), 0.35);
    let rot_axes = [
        Vec3::new(0.0, 1.0, 0.2),
        Vec3::new(1.0, 0.1, -0.3),
        Vec3::new(-0.2, -1.0, 0.3),
        Vec3::new(-0.9, 0.2, 0.4),
    ];
    let rot_angles = [0.60, 0.66, 0.58, 0.64];

    for k in 0..5 {
        scene.add_triad(plaq_pts[k], triad, 0.25, 0.013);
        if k < 4 {
            triad = triad.rotated(rot_axes[k], rot_angles[k]);
        }
    }

    // Glowing Magenta Commutator Gap Vector [Omega_y, Omega_x] != 0 closing the plaquette
    let m_comm = Material::glowing(Vec3::new(0.98, 0.18, 0.90), 0.85);
    scene.add_arrow(plaq_pts[4], plaq_pts[0], 0.028, 0.068, 0.14, m_comm);

    scene
}

fn intersect_finsler_shell(
    ro: Vec3,
    rd: Vec3,
    metric: &FinslerRandersMetric,
    shell_scale: f64,
) -> Option<(f64, Vec3, f64)> {
    let b_bound = ro.dot(rd);
    let c_bound = ro.dot(ro) - 2.45 * 2.45;
    if c_bound > 0.0 && (b_bound > 0.0 || b_bound * b_bound < c_bound) {
        return None;
    }
    let s_bound = (b_bound * b_bound - c_bound).max(0.0).sqrt();
    let mut t = (-b_bound - s_bound).max(0.25);
    let t_max = (-b_bound + s_bound).min(9.5);
    if t >= t_max {
        return None;
    }
    let mut prev_val = metric.evaluate(ro + rd * t) - shell_scale;

    for _ in 0..96 {
        let p = ro + rd * t;
        let val = metric.evaluate(p) - shell_scale;
        if prev_val * val <= 0.0 {
            let mut a = t - 0.07;
            let mut b = t;
            for _ in 0..8 {
                let mid = 0.5 * (a + b);
                let v_mid = metric.evaluate(ro + rd * mid) - shell_scale;
                if prev_val * v_mid <= 0.0 {
                    b = mid;
                } else {
                    a = mid;
                    prev_val = v_mid;
                }
            }
            let t_hit = 0.5 * (a + b);
            let p_hit = ro + rd * t_hit;
            let eps = 0.002;
            let gx = metric.evaluate(p_hit + Vec3::new(eps, 0.0, 0.0))
                - metric.evaluate(p_hit - Vec3::new(eps, 0.0, 0.0));
            let gy = metric.evaluate(p_hit + Vec3::new(0.0, eps, 0.0))
                - metric.evaluate(p_hit - Vec3::new(0.0, eps, 0.0));
            let gz = metric.evaluate(p_hit + Vec3::new(0.0, 0.0, eps))
                - metric.evaluate(p_hit - Vec3::new(0.0, 0.0, eps));
            let n = Vec3::new(gx, gy, gz).normalized();
            let dir_cost = metric.evaluate(p_hit.normalized());
            return Some((t_hit, n, dir_cost));
        }
        prev_val = val;
        t += 0.07;
        if t > t_max {
            break;
        }
    }
    None
}

pub fn trace_scene_na3d(
    ro: Vec3,
    rd: Vec3,
    scene: &Scene,
    metric: &FinslerRandersMetric,
) -> Vec3 {
    let base_col = scene.evaluate_ray(ro, rd);
    let shell_scale = 1.24;

    if let Some((t_shell, n_shell, dir_cost)) = intersect_finsler_shell(ro, rd, metric, shell_scale)
    {
        let opaque_t = scene.trace_opaque(ro, rd).map(|h| h.t).unwrap_or(f64::INFINITY);
        let p_shell = ro + rd * t_shell;
        let l_key = Vec3::new(0.54, 0.78, 0.46).normalized();
        let v = -rd;
        let h = (l_key + v).normalized();

        let cost_t = ((dir_cost - 0.65) / 0.82).clamp(0.0, 1.0);
        let shell_tint = colormap_finsler_cost(cost_t);

        let ndotv = n_shell.dot(v).abs().clamp(0.0, 1.0);
        let fresnel = (1.0 - ndotv).powf(3.4);
        let spec = n_shell.dot(h).max(0.0).powf(140.0);

        let lat_angle = (p_shell.y / p_shell.norm().max(1e-6)).asin();
        let lon_angle = p_shell.z.atan2(p_shell.x);
        let grid_wave = ((lat_angle * 16.0).cos().abs().powf(24.0)
            + (lon_angle * 16.0).cos().abs().powf(24.0))
        .clamp(0.0, 1.0);

        let shell_col = shell_tint * 0.45
            + Vec3::new(0.35, 0.86, 1.0) * (fresnel * 0.85 + grid_wave * 0.35)
            + Vec3::splat(spec * 0.90);

        // Front-window cutaway transparency: keeps interior simplex 100% sharp and vivid!
        let alpha = if t_shell < opaque_t {
            (0.04 + 0.66 * fresnel + 0.16 * grid_wave).clamp(0.04, 0.82)
        } else {
            0.05
        };
        return base_col.lerp(shell_col.clamp01(), alpha);
    }

    base_col
}
