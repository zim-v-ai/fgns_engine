//! Location: // fgns_engine/tools/fgns_manifold_engine/src/math.rs
//! Description: Exact Differential Geometry, SO(3) Rodrigues-Hamilton Lie Algebra
//!              Exponential Map, Orthonormal Frame Construction, Asymmetric Finsler-Randers
//!              Metric Indicatrix, and High-Contrast Scientific Colormaps (FGNS v5.0).
//! Author: Vyacheslav Zimovsky (v.zimovsky.ai@gmail.com)
//! License: MIT-0

use std::ops::{Add, Mul, Neg, Sub};

#[derive(Clone, Copy, Debug, Default)]
pub struct Vec3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Vec3 {
    #[inline]
    pub const fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    #[inline]
    pub const fn splat(v: f64) -> Self {
        Self { x: v, y: v, z: v }
    }

    #[inline]
    pub fn dot(self, o: Self) -> f64 {
        self.x * o.x + self.y * o.y + self.z * o.z
    }

    #[inline]
    pub fn cross(self, o: Self) -> Self {
        Self::new(
            self.y * o.z - self.z * o.y,
            self.z * o.x - self.x * o.z,
            self.x * o.y - self.y * o.x,
        )
    }

    #[inline]
    pub fn norm_sq(self) -> f64 {
        self.dot(self)
    }

    #[inline]
    pub fn norm(self) -> f64 {
        self.norm_sq().sqrt()
    }

    #[inline]
    pub fn normalized(self) -> Self {
        let n = self.norm();
        if n > 1e-12 {
            self * (1.0 / n)
        } else {
            Self::new(0.0, 1.0, 0.0)
        }
    }

    #[inline]
    pub fn cwise_mul(self, o: Self) -> Self {
        Self::new(self.x * o.x, self.y * o.y, self.z * o.z)
    }

    #[inline]
    pub fn lerp(self, o: Self, t: f64) -> Self {
        self * (1.0 - t) + o * t
    }

    #[inline]
    pub fn clamp01(self) -> Self {
        Self::new(
            self.x.clamp(0.0, 1.0),
            self.y.clamp(0.0, 1.0),
            self.z.clamp(0.0, 1.0),
        )
    }

    #[inline]
    pub fn max_comp(self, o: Self) -> Self {
        Self::new(self.x.max(o.x), self.y.max(o.y), self.z.max(o.z))
    }

    #[inline]
    pub fn abs(self) -> Self {
        Self::new(self.x.abs(), self.y.abs(), self.z.abs())
    }
}

impl Add for Vec3 {
    type Output = Self;
    #[inline]
    fn add(self, o: Self) -> Self {
        Self::new(self.x + o.x, self.y + o.y, self.z + o.z)
    }
}

impl Sub for Vec3 {
    type Output = Self;
    #[inline]
    fn sub(self, o: Self) -> Self {
        Self::new(self.x - o.x, self.y - o.y, self.z - o.z)
    }
}

impl Mul<f64> for Vec3 {
    type Output = Self;
    #[inline]
    fn mul(self, s: f64) -> Self {
        Self::new(self.x * s, self.y * s, self.z * s)
    }
}

impl Neg for Vec3 {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self::new(-self.x, -self.y, -self.z)
    }
}

/// Exact SO(3) Rodrigues-Hamilton Exponential Map:
/// U(theta, u) v = v cos(theta) + (u x v) sin(theta) + u (u . v) (1 - cos(theta))
#[inline]
pub fn so3_rodrigues_rotate(v: Vec3, axis: Vec3, theta: f64) -> Vec3 {
    let u = axis.normalized();
    let c = theta.cos();
    let s = theta.sin();
    v * c + u.cross(v) * s + u * (u.dot(v) * (1.0 - c))
}

/// Orthonormal 3D Frame Triad (e1, e2, e3) in SO(3)
#[derive(Clone, Copy, Debug)]
pub struct TriadSO3 {
    pub e1: Vec3,
    pub e2: Vec3,
    pub e3: Vec3,
}

impl TriadSO3 {
    pub fn canonical() -> Self {
        Self {
            e1: Vec3::new(1.0, 0.0, 0.0),
            e2: Vec3::new(0.0, 1.0, 0.0),
            e3: Vec3::new(0.0, 0.0, 1.0),
        }
    }

    pub fn rotated(self, axis: Vec3, theta: f64) -> Self {
        Self {
            e1: so3_rodrigues_rotate(self.e1, axis, theta),
            e2: so3_rodrigues_rotate(self.e2, axis, theta),
            e3: so3_rodrigues_rotate(self.e3, axis, theta),
        }
    }

    /// Constructs a singularity-free Duff-Pixar orthonormal basis aligned with unit normal `n`
    pub fn from_normal(n_raw: Vec3) -> Self {
        let n = n_raw.normalized();
        let up = if n.y.abs() < 0.92 {
            Vec3::new(0.0, 1.0, 0.0)
        } else {
            Vec3::new(1.0, 0.0, 0.0)
        };
        let t1 = up.cross(n).normalized();
        let t2 = n.cross(t1).normalized();
        Self {
            e1: t1,
            e2: n,
            e3: t2,
        }
    }
}

/// Asymmetric Finsler-Randers Metric F(x, v) = sqrt(v^T A(x) v) + <beta(x), v>
#[derive(Clone, Copy, Debug)]
pub struct FinslerRandersMetric {
    pub a_diag: Vec3,
    pub off_xy: f64,
    pub off_yz: f64,
    pub beta: Vec3,
}

impl FinslerRandersMetric {
    #[inline]
    pub fn evaluate(&self, v: Vec3) -> f64 {
        let q = self.a_diag.x * v.x * v.x
            + self.a_diag.y * v.y * v.y
            + self.a_diag.z * v.z * v.z
            + 2.0 * self.off_xy * v.x * v.y
            + 2.0 * self.off_yz * v.y * v.z;
        let alpha = q.max(1e-12).sqrt();
        let drift = self.beta.dot(v);
        alpha + drift
    }

    #[inline]
    pub fn riemannian_only(&self, v: Vec3) -> f64 {
        let q = self.a_diag.x * v.x * v.x
            + self.a_diag.y * v.y * v.y
            + self.a_diag.z * v.z * v.z
            + 2.0 * self.off_xy * v.x * v.y
            + 2.0 * self.off_yz * v.y * v.z;
        q.max(1e-12).sqrt()
    }

    /// Exact boundary point on the Finsler-Randers indicatrix F(x, r * dir) = scale
    #[inline]
    pub fn indicatrix_point(&self, dir_unit: Vec3, scale: f64) -> Vec3 {
        let f_dir = self.evaluate(dir_unit).max(0.15);
        dir_unit * (scale / f_dir)
    }
}

/// High-Contrast Scientific Colormap for Finsler Directional Cost Asymmetry
/// Deep Electric Sapphire -> Vivid Cyan-Teal -> Solar Amber -> Radiant Crimson-Magenta
#[inline]
pub fn colormap_finsler_cost(t_raw: f64) -> Vec3 {
    let t = t_raw.clamp(0.0, 1.0);
    let c0 = Vec3::new(0.04, 0.48, 1.00); // Low-cost forward cone (Electric Sapphire)
    let c1 = Vec3::new(0.06, 0.88, 0.78); // Isometric transition (Vivid Cyan-Emerald)
    let c2 = Vec3::new(1.00, 0.72, 0.06); // Anisotropic drift zone (Solar Gold)
    let c3 = Vec3::new(0.98, 0.16, 0.44); // High-cost upstream barrier (Ruby-Magenta)
    if t < 0.3333 {
        c0.lerp(c1, t * 3.0)
    } else if t < 0.6666 {
        c1.lerp(c2, (t - 0.3333) * 3.0)
    } else {
        c2.lerp(c3, (t - 0.6666) * 3.0)
    }
}

/// Official ARC-AGI 11-Symbol Palette Calibrated for Linear High-Contrast PBR Rendering
pub fn arc_palette_color(symbol: usize) -> Vec3 {
    match symbol % 11 {
        0 => Vec3::new(0.11, 0.15, 0.23), // 0: Deep Obsidian Slate
        1 => Vec3::new(0.05, 0.44, 0.98), // 1: Royal Cobalt Blue
        2 => Vec3::new(0.96, 0.14, 0.22), // 2: Pure Crimson Red
        3 => Vec3::new(0.06, 0.80, 0.36), // 3: Vivid Emerald Green
        4 => Vec3::new(1.00, 0.80, 0.04), // 4: Solar Gold Yellow
        5 => Vec3::new(0.54, 0.60, 0.70), // 5: Brushed Titanium Grey
        6 => Vec3::new(0.94, 0.16, 0.74), // 6: Electric Magenta
        7 => Vec3::new(1.00, 0.46, 0.04), // 7: Tangerine Amber
        8 => Vec3::new(0.08, 0.84, 0.98), // 8: Crystalline Cyan
        9 => Vec3::new(0.68, 0.08, 0.22), // 9: Imperial Burgundy
        _ => Vec3::new(0.58, 0.30, 0.98), // 10: Hyperbolic Violet (PAD/Boundary)
    }
}
