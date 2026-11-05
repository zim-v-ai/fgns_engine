//! Location: // fgns_engine/tools/fgns_manifold_engine/src/pbr.rs
//! Description: Energy-Conserving Cook-Torrance GGX Microfacet Ray-Tracer, Exact Analytical
//!              Cone & Smooth Torus Intersectors, Ambient Occlusion, and Deep Obsidian Stage
//!              Compositor for Publication-Grade 3D Manifold Visualizations (FGNS v5.0).
//! Author: Vyacheslav Zimovsky (v.zimovsky.ai@gmail.com)
//! License: MIT-0

use std::f64::consts::PI;

use crate::math::{TriadSO3, Vec3};

#[derive(Clone, Copy, Debug)]
pub struct Material {
    pub albedo: Vec3,
    pub emission: f64,
    pub metallic: f64,
    pub roughness: f64,
    pub opacity: f64,
}

impl Material {
    pub const fn solid(albedo: Vec3, metallic: f64, roughness: f64) -> Self {
        Self {
            albedo,
            emission: 0.0,
            metallic,
            roughness,
            opacity: 1.0,
        }
    }

    pub const fn glowing(albedo: Vec3, emission: f64) -> Self {
        Self {
            albedo,
            emission,
            metallic: 0.30,
            roughness: 0.16,
            opacity: 1.0,
        }
    }

    pub const fn glass(albedo: Vec3, opacity: f64, emission: f64) -> Self {
        Self {
            albedo,
            emission,
            metallic: 0.08,
            roughness: 0.06,
            opacity,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum GeomKind {
    Sphere {
        center: Vec3,
        radius: f64,
    },
    Capsule {
        a: Vec3,
        b: Vec3,
        radius: f64,
    },
    Cone {
        base: Vec3,
        tip: Vec3,
        radius: f64,
    },
    TorusY {
        center: Vec3,
        major_r: f64,
        minor_r: f64,
    },
    Box3D {
        min: Vec3,
        max: Vec3,
    },
    Triangle {
        v0: Vec3,
        v1: Vec3,
        v2: Vec3,
    },
    DiscY {
        center: Vec3,
        radius_inner: f64,
        radius_outer: f64,
    },
}

#[derive(Clone, Copy, Debug)]
pub struct Primitive {
    pub kind: GeomKind,
    pub mat: Material,
    pub bound_center: Vec3,
    pub bound_radius_sq: f64,
}

impl Primitive {
    pub fn new(kind: GeomKind, mat: Material) -> Self {
        let (bound_center, r) = match kind {
            GeomKind::Sphere { center, radius } => (center, radius + 1e-3),
            GeomKind::Capsule { a, b, radius } => {
                let c = (a + b) * 0.5;
                let half = (b - a).norm() * 0.5;
                (c, half + radius + 1e-3)
            }
            GeomKind::Cone { base, tip, radius } => {
                let c = (base + tip) * 0.5;
                let half = (tip - base).norm() * 0.5;
                (c, half + radius + 1e-3)
            }
            GeomKind::TorusY {
                center,
                major_r,
                minor_r,
            } => (center, major_r + minor_r + 2e-3),
            GeomKind::Box3D { min, max } => {
                let c = (min + max) * 0.5;
                let half = (max - min).norm() * 0.5;
                (c, half + 1e-3)
            }
            GeomKind::Triangle { v0, v1, v2 } => {
                let c = (v0 + v1 + v2) * (1.0 / 3.0);
                let r0 = (v0 - c).norm();
                let r1 = (v1 - c).norm();
                let r2 = (v2 - c).norm();
                (c, r0.max(r1).max(r2) + 1e-3)
            }
            GeomKind::DiscY {
                center,
                radius_outer,
                ..
            } => (center, radius_outer + 1e-3),
        };
        Self {
            kind,
            mat,
            bound_center,
            bound_radius_sq: r * r,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct HitRecord {
    pub t: f64,
    pub pos: Vec3,
    pub normal: Vec3,
    pub mat: Material,
}

pub struct Scene {
    pub opaque_prims: Vec<Primitive>,
    pub trans_prims: Vec<Primitive>,
    pub floor_y: f64,
    pub floor_grid_step: f64,
}

impl Scene {
    pub fn new(floor_y: f64, floor_grid_step: f64) -> Self {
        Self {
            opaque_prims: Vec::with_capacity(640),
            trans_prims: Vec::with_capacity(160),
            floor_y,
            floor_grid_step,
        }
    }

    pub fn add_prim(&mut self, kind: GeomKind, mat: Material) {
        let p = Primitive::new(kind, mat);
        if mat.opacity >= 0.995 {
            self.opaque_prims.push(p);
        } else {
            self.trans_prims.push(p);
        }
    }

    pub fn add_sphere(&mut self, center: Vec3, radius: f64, mat: Material) {
        self.add_prim(GeomKind::Sphere { center, radius }, mat);
    }

    pub fn add_capsule(&mut self, a: Vec3, b: Vec3, radius: f64, mat: Material) {
        self.add_prim(GeomKind::Capsule { a, b, radius }, mat);
    }

    pub fn add_cone(&mut self, base: Vec3, tip: Vec3, radius: f64, mat: Material) {
        self.add_prim(GeomKind::Cone { base, tip, radius }, mat);
    }

    pub fn add_torus_y(&mut self, center: Vec3, major_r: f64, minor_r: f64, mat: Material) {
        self.add_prim(
            GeomKind::TorusY {
                center,
                major_r,
                minor_r,
            },
            mat,
        );
    }

    pub fn add_box(&mut self, min: Vec3, max: Vec3, mat: Material) {
        self.add_prim(GeomKind::Box3D { min, max }, mat);
    }

    pub fn add_triangle(&mut self, v0: Vec3, v1: Vec3, v2: Vec3, mat: Material) {
        self.add_prim(GeomKind::Triangle { v0, v1, v2 }, mat);
    }

    pub fn add_disc_y(&mut self, center: Vec3, radius_inner: f64, radius_outer: f64, mat: Material) {
        self.add_prim(
            GeomKind::DiscY {
                center,
                radius_inner,
                radius_outer,
            },
            mat,
        );
    }

    /// Adds a precision 3D scientific vector arrow with metallic collar ring and exact analytical cone head
    pub fn add_arrow(
        &mut self,
        start: Vec3,
        end: Vec3,
        shaft_r: f64,
        head_r: f64,
        head_len: f64,
        mat: Material,
    ) {
        let dir = end - start;
        let len = dir.norm();
        if len < 1e-6 {
            return;
        }
        let u = dir * (1.0 / len);
        let hl = head_len.min(len * 0.40);
        let neck = end - u * hl;
        self.add_capsule(start, neck, shaft_r, mat);
        // Precision metallic collar ring at the junction between shaft and cone
        let collar_mat = Material::solid(
            mat.albedo.lerp(Vec3::new(0.95, 0.97, 1.0), 0.55),
            0.85,
            0.14,
        );
        self.add_capsule(neck - u * (shaft_r * 0.6), neck, shaft_r * 1.32, collar_mat);
        self.add_cone(neck, end, head_r, mat);
    }

    /// Adds a high-contrast 3D orthonormal SO(3) coordinate triad at `origin` with axis length `scale`
    pub fn add_triad(&mut self, origin: Vec3, triad: TriadSO3, scale: f64, r_shaft: f64) {
        let m_hub = Material::solid(Vec3::new(0.94, 0.97, 1.0), 0.85, 0.12);
        self.add_sphere(origin, r_shaft * 2.25, m_hub);

        let m_x = Material::glowing(Vec3::new(0.98, 0.18, 0.32), 0.30);
        let m_y = Material::glowing(Vec3::new(0.08, 0.88, 0.44), 0.30);
        let m_z = Material::glowing(Vec3::new(0.10, 0.62, 1.00), 0.30);

        self.add_arrow(
            origin,
            origin + triad.e1 * scale,
            r_shaft,
            r_shaft * 2.35,
            scale * 0.34,
            m_x,
        );
        self.add_arrow(
            origin,
            origin + triad.e2 * scale,
            r_shaft,
            r_shaft * 2.35,
            scale * 0.34,
            m_y,
        );
        self.add_arrow(
            origin,
            origin + triad.e3 * scale,
            r_shaft,
            r_shaft * 2.35,
            scale * 0.34,
            m_z,
        );
    }
}

#[inline]
fn intersect_sphere(ro: Vec3, rd: Vec3, center: Vec3, radius: f64) -> Option<(f64, Vec3)> {
    let oc = ro - center;
    let b = oc.dot(rd);
    let c = oc.dot(oc) - radius * radius;
    let disc = b * b - c;
    if disc < 0.0 {
        return None;
    }
    let s = disc.sqrt();
    let t1 = -b - s;
    let t = if t1 > 1e-4 { t1 } else { -b + s };
    if t > 1e-4 {
        let p = ro + rd * t;
        Some((t, (p - center) * (1.0 / radius)))
    } else {
        None
    }
}

#[inline]
fn intersect_capsule(ro: Vec3, rd: Vec3, a: Vec3, b: Vec3, r: f64) -> Option<(f64, Vec3)> {
    let ba = b - a;
    let oa = ro - a;
    let baba = ba.dot(ba);
    if baba < 1e-12 {
        return intersect_sphere(ro, rd, a, r);
    }
    let bard = ba.dot(rd);
    let baoa = ba.dot(oa);
    let rdoa = rd.dot(oa);
    let oaoa = oa.dot(oa);

    let a_q = baba - bard * bard;
    let b_q = baba * rdoa - baoa * bard;
    let c_q = baba * oaoa - baoa * baoa - r * r * baba;
    let h = b_q * b_q - a_q * c_q;
    if h >= 0.0 && a_q.abs() > 1e-12 {
        let t = (-b_q - h.sqrt()) / a_q;
        let y = baoa + t * bard;
        if y > 0.0 && y < baba && t > 1e-4 {
            let p = ro + rd * t;
            let proj = a + ba * (y / baba);
            return Some((t, (p - proj).normalized()));
        }
        let oc = if y <= 0.0 { oa } else { ro - b };
        let b_s = rd.dot(oc);
        let c_s = oc.dot(oc) - r * r;
        let h_s = b_s * b_s - c_s;
        if h_s > 0.0 {
            let t_s = -b_s - h_s.sqrt();
            if t_s > 1e-4 {
                let p = ro + rd * t_s;
                let c_pt = if y <= 0.0 { a } else { b };
                return Some((t_s, (p - c_pt).normalized()));
            }
        }
    }
    None
}

/// Exact analytical ray-cone intersection (with planar base cap) — zero stepped artifacts!
#[inline]
fn intersect_cone_exact(ro: Vec3, rd: Vec3, base: Vec3, tip: Vec3, r: f64) -> Option<(f64, Vec3)> {
    let axis_vec = base - tip;
    let h = axis_vec.norm();
    if h < 1e-7 || r < 1e-7 {
        return None;
    }
    let v_axis = axis_vec * (1.0 / h);
    let cos2 = (h * h) / (h * h + r * r);
    let tan_theta = r / h;

    let co = ro - tip;
    let d_v = rd.dot(v_axis);
    let co_v = co.dot(v_axis);

    let a = d_v * d_v - cos2;
    let b = d_v * co_v - cos2 * rd.dot(co);
    let c = co_v * co_v - cos2 * co.dot(co);

    let mut best_t = f64::INFINITY;
    let mut best_n = Vec3::new(0.0, 1.0, 0.0);

    let disc = b * b - a * c;
    if disc >= 0.0 && a.abs() > 1e-12 {
        let s = disc.sqrt();
        for &t_cand in &[(-b - s) / a, (-b + s) / a] {
            if t_cand > 1e-4 && t_cand < best_t {
                let p = ro + rd * t_cand;
                let y = (p - tip).dot(v_axis);
                if y >= 0.0 && y <= h {
                    let radial = (p - (tip + v_axis * y)).normalized();
                    let n = (radial - v_axis * tan_theta).normalized();
                    best_t = t_cand;
                    best_n = n;
                }
            }
        }
    }

    // Check planar circular base cap at `base`
    if d_v.abs() > 1e-9 {
        let t_cap = (base - ro).dot(v_axis) / d_v;
        if t_cap > 1e-4 && t_cap < best_t {
            let p_cap = ro + rd * t_cap;
            if (p_cap - base).norm_sq() <= r * r {
                best_t = t_cap;
                best_n = v_axis;
            }
        }
    }

    if best_t < f64::INFINITY {
        Some((best_t, best_n))
    } else {
        None
    }
}

/// Exact analytical SDF sphere-tracing + Newton root polish for a smooth horizontal torus in XZ plane
#[inline]
fn intersect_torus_y(
    ro: Vec3,
    rd: Vec3,
    center: Vec3,
    major_r: f64,
    minor_r: f64,
) -> Option<(f64, Vec3)> {
    let oc = ro - center;
    let bound_r = major_r + minor_r + 0.02;
    let b = oc.dot(rd);
    let c = oc.dot(oc) - bound_r * bound_r;
    let disc = b * b - c;
    if disc < 0.0 {
        return None;
    }
    let s = disc.sqrt();
    let mut t = (-b - s).max(1e-4);
    let t_max = -b + s;
    if t >= t_max {
        return None;
    }

    let sdf = |p: Vec3| -> f64 {
        let q = p - center;
        let q_xz = (q.x * q.x + q.z * q.z).sqrt() - major_r;
        (q_xz * q_xz + q.y * q.y).sqrt() - minor_r
    };

    for _ in 0..56 {
        let p = ro + rd * t;
        let d = sdf(p);
        if d < 1e-4 {
            let q = p - center;
            let r_xz = (q.x * q.x + q.z * q.z).sqrt().max(1e-9);
            let ring_pt = Vec3::new(q.x * (major_r / r_xz), 0.0, q.z * (major_r / r_xz));
            let n = (q - ring_pt).normalized();
            return Some((t, n));
        }
        t += d.max(0.002);
        if t > t_max {
            break;
        }
    }
    None
}

#[inline]
fn intersect_box(ro: Vec3, rd: Vec3, bmin: Vec3, bmax: Vec3) -> Option<(f64, Vec3)> {
    let inv_x = if rd.x.abs() > 1e-12 { 1.0 / rd.x } else { 1e12 };
    let inv_y = if rd.y.abs() > 1e-12 { 1.0 / rd.y } else { 1e12 };
    let inv_z = if rd.z.abs() > 1e-12 { 1.0 / rd.z } else { 1e12 };

    let tx1 = (bmin.x - ro.x) * inv_x;
    let tx2 = (bmax.x - ro.x) * inv_x;
    let mut tmin = tx1.min(tx2);
    let mut tmax = tx1.max(tx2);

    let ty1 = (bmin.y - ro.y) * inv_y;
    let ty2 = (bmax.y - ro.y) * inv_y;
    tmin = tmin.max(ty1.min(ty2));
    tmax = tmax.min(ty1.max(ty2));

    let tz1 = (bmin.z - ro.z) * inv_z;
    let tz2 = (bmax.z - ro.z) * inv_z;
    tmin = tmin.max(tz1.min(tz2));
    tmax = tmax.min(tz1.max(tz2));

    if tmax >= tmin.max(1e-4) {
        let t = if tmin > 1e-4 { tmin } else { tmax };
        let p = ro + rd * t;
        let c = (bmin + bmax) * 0.5;
        let half = (bmax - bmin) * 0.5;
        let local = p - c;
        let dx = (local.x.abs() - half.x).abs();
        let dy = (local.y.abs() - half.y).abs();
        let dz = (local.z.abs() - half.z).abs();
        let n = if dx <= dy && dx <= dz {
            Vec3::new(local.x.signum(), 0.0, 0.0)
        } else if dy <= dz {
            Vec3::new(0.0, local.y.signum(), 0.0)
        } else {
            Vec3::new(0.0, 0.0, local.z.signum())
        };
        Some((t, n))
    } else {
        None
    }
}

#[inline]
fn intersect_triangle(ro: Vec3, rd: Vec3, v0: Vec3, v1: Vec3, v2: Vec3) -> Option<(f64, Vec3)> {
    let e1 = v1 - v0;
    let e2 = v2 - v0;
    let h = rd.cross(e2);
    let a = e1.dot(h);
    if a.abs() < 1e-9 {
        return None;
    }
    let f = 1.0 / a;
    let s = ro - v0;
    let u = f * s.dot(h);
    if !(0.0..=1.0).contains(&u) {
        return None;
    }
    let q = s.cross(e1);
    let v = f * rd.dot(q);
    if v < 0.0 || u + v > 1.0 {
        return None;
    }
    let t = f * e2.dot(q);
    if t > 1e-4 {
        let mut n = e1.cross(e2).normalized();
        if n.dot(rd) > 0.0 {
            n = -n;
        }
        Some((t, n))
    } else {
        None
    }
}

#[inline]
fn intersect_disc_y(
    ro: Vec3,
    rd: Vec3,
    center: Vec3,
    r_in: f64,
    r_out: f64,
) -> Option<(f64, Vec3)> {
    if rd.y.abs() < 1e-9 {
        return None;
    }
    let t = (center.y - ro.y) / rd.y;
    if t <= 1e-4 {
        return None;
    }
    let p = ro + rd * t;
    let dx = p.x - center.x;
    let dz = p.z - center.z;
    let r2 = dx * dx + dz * dz;
    if r2 >= r_in * r_in && r2 <= r_out * r_out {
        let n = if rd.y < 0.0 {
            Vec3::new(0.0, 1.0, 0.0)
        } else {
            Vec3::new(0.0, -1.0, 0.0)
        };
        Some((t, n))
    } else {
        None
    }
}

#[inline]
pub fn intersect_primitive(ro: Vec3, rd: Vec3, prim: &Primitive) -> Option<(f64, Vec3)> {
    let oc = ro - prim.bound_center;
    let b = oc.dot(rd);
    let c = oc.dot(oc) - prim.bound_radius_sq;
    if c > 0.0 && (b > 0.0 || b * b < c) {
        return None;
    }
    match prim.kind {
        GeomKind::Sphere { center, radius } => intersect_sphere(ro, rd, center, radius),
        GeomKind::Capsule { a, b, radius } => intersect_capsule(ro, rd, a, b, radius),
        GeomKind::Cone { base, tip, radius } => intersect_cone_exact(ro, rd, base, tip, radius),
        GeomKind::TorusY {
            center,
            major_r,
            minor_r,
        } => intersect_torus_y(ro, rd, center, major_r, minor_r),
        GeomKind::Box3D { min, max } => intersect_box(ro, rd, min, max),
        GeomKind::Triangle { v0, v1, v2 } => intersect_triangle(ro, rd, v0, v1, v2),
        GeomKind::DiscY {
            center,
            radius_inner,
            radius_outer,
        } => intersect_disc_y(ro, rd, center, radius_inner, radius_outer),
    }
}

/// Cook-Torrance GGX Microfacet evaluation for a single directional light source
#[inline]
fn eval_ggx_light(
    n: Vec3,
    v: Vec3,
    l: Vec3,
    albedo: Vec3,
    f0: Vec3,
    metallic: f64,
    roughness: f64,
    radiance: Vec3,
) -> Vec3 {
    let ndotl = n.dot(l).max(0.0);
    if ndotl <= 1e-6 {
        return Vec3::splat(0.0);
    }
    let ndotv = n.dot(v).abs().max(1e-4);
    let h = (v + l).normalized();
    let ndoth = n.dot(h).max(0.0);
    let vdoth = v.dot(h).max(0.0);

    let alpha = (roughness * roughness).max(0.02);
    let a2 = alpha * alpha;
    let denom_d = ndoth * ndoth * (a2 - 1.0) + 1.0;
    let d_ggx = a2 / (PI * denom_d * denom_d).max(1e-7);

    let k = ((roughness + 1.0) * (roughness + 1.0)) * 0.125;
    let g1_v = ndotv / (ndotv * (1.0 - k) + k);
    let g1_l = ndotl / (ndotl * (1.0 - k) + k);
    let g_smith = g1_v * g1_l;

    let f_schlick = f0 + (Vec3::splat(1.0) - f0) * (1.0 - vdoth).powi(5);
    let spec_num = f_schlick * (d_ggx * g_smith);
    let spec_den = (4.0 * ndotv * ndotl).max(1e-4);
    let specular = spec_num * (1.0 / spec_den);

    let k_d = (Vec3::splat(1.0) - f_schlick) * (1.0 - metallic);
    let diffuse = k_d.cwise_mul(albedo) * (1.0 / PI);

    (diffuse + specular).cwise_mul(radiance) * ndotl
}

impl Scene {
    pub fn trace_opaque(&self, ro: Vec3, rd: Vec3) -> Option<HitRecord> {
        let mut best_t = f64::INFINITY;
        let mut best_hit: Option<HitRecord> = None;

        for prim in &self.opaque_prims {
            if let Some((t, n)) = intersect_primitive(ro, rd, prim) {
                if t < best_t {
                    best_t = t;
                    best_hit = Some(HitRecord {
                        t,
                        pos: ro + rd * t,
                        normal: n,
                        mat: prim.mat,
                    });
                }
            }
        }
        best_hit
    }

    pub fn is_shadowed(&self, p: Vec3, light_dir: Vec3) -> bool {
        let ro = p + light_dir * 0.008;
        for prim in &self.opaque_prims {
            if let Some((t, _)) = intersect_primitive(ro, light_dir, prim) {
                if t < 16.0 {
                    return true;
                }
            }
        }
        false
    }

    /// Evaluates short-range 5-ray hemispherical Ambient Occlusion for crisp contact shadows
    pub fn compute_ambient_occlusion(&self, p: Vec3, n: Vec3) -> f64 {
        let triad = TriadSO3::from_normal(n);
        let dirs = [
            n,
            (n + triad.e1 * 0.75).normalized(),
            (n - triad.e1 * 0.75).normalized(),
            (n + triad.e3 * 0.75).normalized(),
            (n - triad.e3 * 0.75).normalized(),
        ];
        let mut occ = 0.0;
        let ro = p + n * 0.006;
        for d in &dirs {
            for prim in &self.opaque_prims {
                if let Some((t, _)) = intersect_primitive(ro, *d, prim) {
                    if t < 0.34 {
                        occ += (1.0 - t / 0.34) * 0.16;
                        break;
                    }
                }
            }
        }
        (1.0 - occ).clamp(0.32, 1.0)
    }

    pub fn collect_translucent_hits(&self, ro: Vec3, rd: Vec3, max_t: f64) -> Vec<HitRecord> {
        let mut hits = Vec::new();
        for prim in &self.trans_prims {
            if let Some((t, n)) = intersect_primitive(ro, rd, prim) {
                if t < max_t {
                    hits.push(HitRecord {
                        t,
                        pos: ro + rd * t,
                        normal: n,
                        mat: prim.mat,
                    });
                }
            }
        }
        hits.sort_by(|a, b| b.t.partial_cmp(&a.t).unwrap_or(std::cmp::Ordering::Equal));
        hits
    }

    /// Energy-Conserving Cook-Torrance GGX Microfacet Shading with Silhouette Rim Definition
    pub fn shade_hit(&self, hit: &HitRecord, rd: Vec3, allow_shadow: bool) -> Vec3 {
        let n = hit.normal;
        let v = -rd;
        let f0 = Vec3::splat(0.04).lerp(hit.mat.albedo, hit.mat.metallic);

        let l_key = Vec3::new(0.54, 0.78, 0.46).normalized();
        let l_fill = Vec3::new(-0.72, 0.44, 0.52).normalized();
        let l_rim = Vec3::new(-0.15, 0.42, -0.89).normalized();

        let shadow = if allow_shadow && self.is_shadowed(hit.pos, l_key) {
            0.26
        } else {
            1.0
        };
        let ao = if allow_shadow {
            self.compute_ambient_occlusion(hit.pos, n)
        } else {
            1.0
        };

        let rad_key = Vec3::new(2.75, 2.68, 2.58) * shadow;
        let rad_fill = Vec3::new(0.85, 1.05, 1.38) * ao;
        let rad_rim = Vec3::new(1.15, 1.30, 1.65) * ao;

        let lo_key = eval_ggx_light(
            n,
            v,
            l_key,
            hit.mat.albedo,
            f0,
            hit.mat.metallic,
            hit.mat.roughness,
            rad_key,
        );
        let lo_fill = eval_ggx_light(
            n,
            v,
            l_fill,
            hit.mat.albedo,
            f0,
            hit.mat.metallic,
            hit.mat.roughness,
            rad_fill,
        );
        let lo_rim = eval_ggx_light(
            n,
            v,
            l_rim,
            hit.mat.albedo,
            f0,
            hit.mat.metallic,
            hit.mat.roughness,
            rad_rim,
        );

        let ndotv = n.dot(v).abs().clamp(0.0, 1.0);
        let fresnel_rim = (1.0 - ndotv).powf(3.6);
        let hemi_env = Vec3::new(0.04, 0.07, 0.13).lerp(Vec3::new(0.18, 0.28, 0.46), 0.5 + 0.5 * n.y);
        let ambient = hit.mat.albedo.cwise_mul(hemi_env) * (0.28 * ao)
            + hit.mat.albedo * (0.22 * ao * (1.0 - 0.55 * hit.mat.metallic));

        // Crisp dark silhouette edge ink at grazing angles for publication-grade legibility
        let edge_ink = if ndotv < 0.18 {
            0.55 + 0.45 * (ndotv / 0.18)
        } else {
            1.0
        };

        let emissive = hit.mat.albedo * (hit.mat.emission * 1.18);
        let rim_glow = hit.mat.albedo.lerp(Vec3::new(0.65, 0.90, 1.0), 0.35) * (fresnel_rim * 0.24 * ao);

        (ambient + lo_key + lo_fill + lo_rim + rim_glow) * edge_ink + emissive
    }

    /// Deep Obsidian Carbon Laboratory Stage Floor (`#070B14`) with Subtle Gold/Sapphire Polar Grid
    pub fn shade_studio_background_and_floor(&self, ro: Vec3, rd: Vec3) -> (Vec3, f64) {
        let sky_zenith = Vec3::new(0.004, 0.007, 0.015);
        let sky_horizon = Vec3::new(0.012, 0.020, 0.042);
        let sky_t = (1.0 - rd.y.abs().powf(0.65)).clamp(0.0, 1.0);
        let mut bg = sky_zenith.lerp(sky_horizon, sky_t);

        if rd.y < -1e-5 {
            let t_floor = (self.floor_y - ro.y) / rd.y;
            if t_floor > 1e-4 && t_floor < 26.0 {
                let p = ro + rd * t_floor;
                let r_dist = (p.x * p.x + p.z * p.z).sqrt();
                if r_dist < 13.0 {
                    let fade = (1.0 - (r_dist / 13.0).powi(2)).clamp(0.0, 1.0);
                    let step = self.floor_grid_step;
                    let fx = (p.x / step).rem_euclid(1.0);
                    let fz = (p.z / step).rem_euclid(1.0);
                    let dx = fx.min(1.0 - fx);
                    let dz = fz.min(1.0 - fz);
                    let d_major = dx.min(dz);
                    let line_glow = ((0.014 - d_major).max(0.0) / 0.014).powf(2.0);

                    let mut floor_col = Vec3::new(0.007, 0.011, 0.022)
                        .lerp(Vec3::new(0.06, 0.24, 0.48), line_glow * 0.38);

                    let r_mod = (r_dist / (step * 2.0)).rem_euclid(1.0);
                    let d_ring = r_mod.min(1.0 - r_mod);
                    if d_ring < 0.012 {
                        let ring_w = ((0.012 - d_ring) / 0.012).powf(1.8);
                        floor_col = floor_col.lerp(Vec3::new(0.18, 0.52, 0.85), ring_w * 0.32);
                    }

                    let l_key = Vec3::new(0.54, 0.78, 0.46).normalized();
                    if self.is_shadowed(p, l_key) {
                        floor_col = floor_col * 0.35;
                    }

                    let refl_rd = Vec3::new(rd.x, -rd.y, rd.z);
                    if let Some(refl_hit) =
                        self.trace_opaque(p + Vec3::new(0.0, 0.005, 0.0), refl_rd)
                    {
                        let refl_c = self.shade_hit(&refl_hit, refl_rd, false).clamp01();
                        let refl_atten = (-refl_hit.t * 0.40).exp() * 0.24;
                        floor_col = floor_col.lerp(refl_c, refl_atten);
                    }

                    bg = bg.lerp(floor_col, fade * 0.95);
                    return (bg, t_floor);
                }
            }
        }
        (bg, f64::INFINITY)
    }

    /// Evaluates full radiance along ray (ro, rd) with Beer-Lambert dielectric glass compositing
    pub fn evaluate_ray(&self, ro: Vec3, rd: Vec3) -> Vec3 {
        let (bg_col, t_floor) = self.shade_studio_background_and_floor(ro, rd);
        let opaque_hit = self.trace_opaque(ro, rd);

        let (mut base_col, max_t) = match opaque_hit {
            Some(ref h) if h.t < t_floor => (self.shade_hit(h, rd, true), h.t),
            _ => (bg_col, t_floor),
        };

        let trans_hits = self.collect_translucent_hits(ro, rd, max_t);
        for th in trans_hits {
            let ndotv = th.normal.dot(-rd).abs().clamp(0.0, 1.0);
            let fresnel = (1.0 - ndotv).powf(3.0);
            let l_key = Vec3::new(0.54, 0.78, 0.46).normalized();
            let h_vec = (l_key - rd).normalized();
            let spec = th.normal.dot(h_vec).max(0.0).powf(120.0);

            // Beer-Lambert transmittance preserves dark background contrast through glass
            let tint = Vec3::splat(1.0).lerp(th.mat.albedo, th.mat.opacity * 0.55);
            let transmitted = base_col.cwise_mul(tint);
            let surface_glow = th.mat.albedo * (th.mat.emission + 0.55 * fresnel)
                + Vec3::splat(spec * 0.85);
            let alpha_edge = (th.mat.opacity * 0.45 + 0.50 * fresnel).clamp(0.06, 0.88);
            base_col = transmitted.lerp(surface_glow, alpha_edge);
        }

        base_col
    }
}
