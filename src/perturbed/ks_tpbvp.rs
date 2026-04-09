//! MCPI-KS-TPBVP: Two-Point Boundary Value Problem solver using
//! Kustaanheimo-Stiefel (KS) regularization for medium-arc perturbed
//! Lambert transfers (transfer angle 2π/3 ≤ θ < ~1.8π).
//!
//! The KS transformation regularizes the two-body equations of motion by:
//!   1. Embedding 3D position into 4D spinor space
//!   2. Introducing fictitious time `s` via `dt = ||u||² ds`
//!   3. Linearizing the unperturbed problem into a harmonic oscillator
//!
//! This extends the TPBVP convergence domain from ~1/3 orbit to ~90%.

use nalgebra::Vector3;

use crate::force_models::ForceModel;
use crate::perturbed::chebyshev::{
    cgl_nodes, chebyshev_t_all, coefficients_from_nodes,
    coefficients_from_nodes_3d, integrate_chebyshev_coeffs, integrate_chebyshev_coeffs_3d,
};

// =========================================================================
// KS transformation utilities
// =========================================================================

/// A 4D KS state vector.
pub type KsVec4 = [f64; 4];

/// Convert a 3D Cartesian position to a 4D KS spinor.
///
/// The inverse mapping is not unique — there is a one-parameter family of
/// spinors for each position. We choose u4 = 0 when r.x ≥ 0 (the "standard"
/// branch).
pub fn cartesian_to_ks(r: &Vector3<f64>) -> KsVec4 {
    let r_mag = r.norm();
    if r_mag < 1e-30 {
        return [0.0; 4];
    }

    if r.x >= 0.0 {
        // Standard branch: u4 = 0
        let u1 = ((r_mag + r.x) / 2.0).sqrt();
        let denom = 2.0 * u1;
        let u2 = r.y / denom;
        let u3 = r.z / denom;
        [u1, u2, u3, 0.0]
    } else {
        // Alternate branch when r.x < 0: u3 = 0
        let u2 = ((r_mag - r.x) / 2.0).sqrt();
        let denom = 2.0 * u2;
        let u1 = r.y / denom;
        let u4 = -r.z / denom;
        [u1, u2, 0.0, u4] // u3 = 0
    }
}

/// Convert a 4D KS spinor to a 3D Cartesian position: r = L(u)·u.
pub fn ks_to_cartesian(u: &KsVec4) -> Vector3<f64> {
    Vector3::new(
        u[0] * u[0] - u[1] * u[1] - u[2] * u[2] + u[3] * u[3],
        2.0 * (u[0] * u[1] - u[2] * u[3]),
        2.0 * (u[0] * u[2] + u[1] * u[3]),
    )
}

/// Compute ||u||² = r_mag.
#[inline]
pub fn ks_norm_sq(u: &KsVec4) -> f64 {
    u[0] * u[0] + u[1] * u[1] + u[2] * u[2] + u[3] * u[3]
}

/// Convert 3D Cartesian velocity to KS velocity (du/ds).
///
/// Given the physical velocity v = dr/dt and the spinor u, the
/// KS velocity is du/ds = r · (du/dt) where:
///   du/dt = ½ L(u)ᵀ · v / ||u||²
/// So: du/ds = ||u||² · du/dt = ½ L(u)ᵀ · v
pub fn cartesian_vel_to_ks(u: &KsVec4, v: &Vector3<f64>) -> KsVec4 {
    // ½ L(u)ᵀ v
    [
        0.5 * (u[0] * v.x + u[1] * v.y + u[2] * v.z),
        0.5 * (-u[1] * v.x + u[0] * v.y + u[3] * v.z),
        0.5 * (-u[2] * v.x - u[3] * v.y + u[0] * v.z),
        0.5 * (u[3] * v.x - u[2] * v.y + u[1] * v.z),
    ]
}

/// Convert KS velocity (du/ds) back to 3D Cartesian velocity.
///
/// v = dr/dt = (1/r) dr/ds = (1/||u||²) · 2 L(u) u'
pub fn ks_vel_to_cartesian(u: &KsVec4, u_prime: &KsVec4) -> Vector3<f64> {
    let r = ks_norm_sq(u);
    let factor = 2.0 / r;
    // L(u) · u' (the first three components of the 4D product)
    let rx = u[0] * u_prime[0] - u[1] * u_prime[1] - u[2] * u_prime[2] + u[3] * u_prime[3];
    let ry = u[1] * u_prime[0] + u[0] * u_prime[1] - u[3] * u_prime[2] - u[2] * u_prime[3];
    let rz = u[2] * u_prime[0] + u[3] * u_prime[1] + u[0] * u_prime[2] + u[1] * u_prime[3];
    Vector3::new(factor * rx, factor * ry, factor * rz)
}

// =========================================================================
// KS-regularized equations of motion
// =========================================================================

/// Compute the KS second derivative u'' from the regularized EOM.
///
/// The regularized EOM in fictitious time s is:
///   u'' = (h/2) u + (r/2) P(u)
/// where:
///   h = −μ/(2a) = v²/2 − μ/r  (total orbital energy, conserved for two-body)
///   P(u) = ½ L(u)ᵀ · a_perturb  (perturbing acceleration in KS coords)
///   r = ||u||²
///
/// For the TPBVP we evolve the full EOM including the energy equation:
///   h' = 2 u' · P(u) / r    (energy rate for perturbed motion)
fn ks_accel(
    u: &KsVec4,
    u_prime: &KsVec4,
    h: f64,
    mu: f64,
    force_model: &dyn ForceModel,
    t: f64,
) -> (KsVec4, f64) {
    let r = ks_norm_sq(u);
    let r_vec = ks_to_cartesian(u);
    let v_vec = ks_vel_to_cartesian(u, u_prime);

    // Full acceleration
    let a_full = force_model.acceleration(t, &r_vec, &v_vec);
    // Two-body acceleration
    let a_twobody = -mu / (r * r.sqrt()) * &r_vec;
    // Perturbing acceleration
    let a_perturb = a_full - a_twobody;

    // P(u) = ½ L(u)ᵀ · a_perturb
    let p = [
        0.5 * (u[0] * a_perturb.x + u[1] * a_perturb.y + u[2] * a_perturb.z),
        0.5 * (-u[1] * a_perturb.x + u[0] * a_perturb.y + u[3] * a_perturb.z),
        0.5 * (-u[2] * a_perturb.x - u[3] * a_perturb.y + u[0] * a_perturb.z),
        0.5 * (u[3] * a_perturb.x - u[2] * a_perturb.y + u[1] * a_perturb.z),
    ];

    // u'' = (h/2) u + (r/2) P(u)
    let u_ddot = [
        0.5 * h * u[0] + 0.5 * r * p[0],
        0.5 * h * u[1] + 0.5 * r * p[1],
        0.5 * h * u[2] + 0.5 * r * p[2],
        0.5 * h * u[3] + 0.5 * r * p[3],
    ];

    // h' = 2 dot(u', P) (Note: some formulations divide by r, but with
    // dt = r ds the chain rule cancels: dh/ds = r · dh/dt)
    let h_dot = 2.0 * (u_prime[0] * p[0] + u_prime[1] * p[1] + u_prime[2] * p[2] + u_prime[3] * p[3]);

    (u_ddot, h_dot)
}

// =========================================================================
// Fictitious time mapping
// =========================================================================

/// Estimate the total fictitious time Δs for a Keplerian transfer.
///
/// For a Keplerian orbit, ds = dt / r, so Δs = ∫ dt/r. For a rough
/// estimate, use the mean-motion relation:
///   Δs ≈ Δt / ā   where ā ~ (r1 + r2)/2
/// A better estimate uses the eccentric anomaly: Δs = ΔE / √(μ/a³) · a
/// For simplicity we use an IVP propagation to map physical times to
/// fictitious times at CGL nodes.
fn estimate_fictitious_time(
    r1: &Vector3<f64>,
    r2: &Vector3<f64>,
    dt: f64,
    mu: f64,
    a: f64,
) -> f64 {
    // For a Keplerian ellipse: mean motion n = sqrt(mu/a^3)
    // ds = dt/r, and on average r ~ a (for moderate eccentricity)
    // So Δs ≈ Δt · n = Δt · sqrt(mu/a^3) ... but that's the mean anomaly.
    // Actually for KS: s maps to half the eccentric anomaly difference.
    // A safe estimate: Δs = Δt / r_avg
    let r_avg = (r1.norm() + r2.norm()) / 2.0;
    let _ = (mu, a); // available for refinement
    dt / r_avg
}

// =========================================================================
// Configuration
// =========================================================================

/// Configuration for the MCPI-KS-TPBVP solver.
#[derive(Debug, Clone)]
pub struct KsTpbvpConfig {
    /// Chebyshev polynomial degree (N CGL intervals, N+1 nodes).
    pub poly_degree: usize,
    /// Maximum number of Picard iterations.
    pub max_iterations: usize,
    /// Convergence tolerance on KS position (km^{1/2}, typ. 1e-10).
    pub tolerance: f64,
}

impl Default for KsTpbvpConfig {
    fn default() -> Self {
        Self {
            poly_degree: 80,
            max_iterations: 50,
            tolerance: 1e-10,
        }
    }
}

// =========================================================================
// Result
// =========================================================================

/// Result of the MCPI-KS-TPBVP solver.
#[derive(Debug, Clone)]
pub struct KsTpbvpResult {
    /// Departure velocity at r1 (km/s).
    pub v1: Vector3<f64>,
    /// Arrival velocity at r2 (km/s).
    pub v2: Vector3<f64>,
    /// Whether the Picard iteration converged.
    pub converged: bool,
    /// Number of Picard iterations actually performed.
    pub iterations_used: usize,
    /// Boundary position error at endpoints (km).
    pub boundary_error: f64,
}

// =========================================================================
// Solver
// =========================================================================

/// Solve the perturbed Lambert problem using MCPI-KS-TPBVP.
///
/// This extends the Cartesian TPBVP to medium arcs (θ up to ~1.8π) by
/// working in KS regularized coordinates. The two-body part of the EOM
/// becomes a harmonic oscillator, vastly improving MCPI convergence for
/// long arcs and high-eccentricity transfers.
///
/// # Arguments
/// * `r1`, `r2` — endpoint positions (km)
/// * `t0`, `tf` — physical time interval (s)
/// * `v0_guess` — initial velocity estimate (typically from Keplerian solver)
/// * `a_guess` — semi-major axis estimate from Keplerian solver (km)
/// * `force_model` — full perturbed force model
/// * `config` — solver configuration
pub fn solve_ks_tpbvp(
    r1: &Vector3<f64>,
    r2: &Vector3<f64>,
    t0: f64,
    tf: f64,
    v0_guess: &Vector3<f64>,
    a_guess: f64,
    force_model: &dyn ForceModel,
    config: &KsTpbvpConfig,
) -> KsTpbvpResult {
    let mu = compute_mu(r1, v0_guess);
    let dt = tf - t0;
    let n = config.poly_degree;

    // ------------------------------------------------------------------
    // Step 1: Convert endpoints to KS coordinates
    // ------------------------------------------------------------------
    let u1 = cartesian_to_ks(r1);
    let u2 = cartesian_to_ks(r2);

    // Initial KS velocity from Keplerian guess
    let u1_prime_guess = cartesian_vel_to_ks(&u1, v0_guess);

    // Orbital energy from vis-viva
    let r1_mag = r1.norm();
    let v0_sq = v0_guess.norm_squared();
    let h0 = v0_sq / 2.0 - mu / r1_mag;

    // ------------------------------------------------------------------
    // Step 2: Estimate fictitious time interval [0, Δs]
    // ------------------------------------------------------------------
    let ds_total = estimate_fictitious_time(r1, r2, dt, mu, a_guess);
    let w = ds_total / 2.0; // half-interval in fictitious time

    // ------------------------------------------------------------------
    // Step 3: Set up CGL nodes in fictitious time
    // ------------------------------------------------------------------
    let tau_nodes = cgl_nodes(n);

    // ------------------------------------------------------------------
    // Step 4: Initial guess — linear interpolation in KS space + IVP warm start
    // ------------------------------------------------------------------
    // We'll use a two-body IVP propagation in KS space for warm start.
    // First, build an initial trajectory estimate via linear interpolation
    // from u1 to u2, then iteratively refine.
    let mut u_nodes: Vec<KsVec4> = Vec::with_capacity(n + 1);
    let mut up_nodes: Vec<KsVec4> = Vec::with_capacity(n + 1);
    let mut h_nodes: Vec<f64> = Vec::with_capacity(n + 1);
    let mut t_phys_nodes: Vec<f64> = Vec::with_capacity(n + 1);

    for j in 0..=n {
        let frac = (tau_nodes[j] + 1.0) / 2.0; // 0 at τ=-1 (u1), 1 at τ=+1 (u2)
        let u_j = [
            u1[0] + frac * (u2[0] - u1[0]),
            u1[1] + frac * (u2[1] - u1[1]),
            u1[2] + frac * (u2[2] - u1[2]),
            u1[3] + frac * (u2[3] - u1[3]),
        ];
        u_nodes.push(u_j);
        up_nodes.push(u1_prime_guess);
        h_nodes.push(h0);
        t_phys_nodes.push(t0 + frac * dt);
    }

    // Pre-compute T_k(τ_j)
    let t_all: Vec<Vec<f64>> = tau_nodes.iter().map(|&t| chebyshev_t_all(n, t)).collect();

    let mut converged = false;
    let mut iterations_used = 0;

    // ------------------------------------------------------------------
    // Step 5: Picard iteration in KS space
    // ------------------------------------------------------------------
    for iter in 0..config.max_iterations {
        iterations_used = iter + 1;

        // Step 5a: Compute physical time at each CGL node by integrating
        //   dt/ds = ||u||² over fictitious time
        //   t(τ) = t0 + w ∫_{-1}^{τ} ||u(s)||² dσ
        let r_sq_nodes: Vec<f64> = u_nodes.iter().map(|u| ks_norm_sq(u)).collect();
        let r_sq_coeffs = coefficients_from_nodes(&r_sq_nodes, n);
        let r_sq_int = integrate_chebyshev_coeffs(&r_sq_coeffs);

        // Physical time at each node
        t_phys_nodes.clear();
        for j in 0..=n {
            let int_val: f64 = r_sq_int.iter().enumerate().map(|(k, &d)| d * t_all[j][k]).sum();
            t_phys_nodes.push(t0 + w * int_val);
        }

        // Step 5b: Evaluate KS accelerations at all CGL nodes
        let mut u_ddot_nodes: Vec<KsVec4> = Vec::with_capacity(n + 1);
        let mut h_dot_nodes: Vec<f64> = Vec::with_capacity(n + 1);
        for j in 0..=n {
            let (u_ddot, h_dot) = ks_accel(
                &u_nodes[j],
                &up_nodes[j],
                h_nodes[j],
                mu,
                force_model,
                t_phys_nodes[j],
            );
            u_ddot_nodes.push(u_ddot);
            h_dot_nodes.push(h_dot);
        }

        // Step 5c: Chebyshev coefficients of u'' (4 components)
        let uddot_3d: Vec<[f64; 3]> = u_ddot_nodes.iter().map(|a| [a[0], a[1], a[2]]).collect();
        let uddot_4th: Vec<f64> = u_ddot_nodes.iter().map(|a| a[3]).collect();
        let uddot_coeffs_3d = coefficients_from_nodes_3d(&uddot_3d, n);
        let uddot_coeffs_4th = coefficients_from_nodes(&uddot_4th, n);

        // Chebyshev coefficients of h'
        let hdot_coeffs = coefficients_from_nodes(&h_dot_nodes, n);

        // Step 5d: First integral u''→ u' (4D)
        let up_int_3d = integrate_chebyshev_coeffs_3d(&uddot_coeffs_3d);
        let up_int_4th = integrate_chebyshev_coeffs(&uddot_coeffs_4th);

        // h integral: h(τ) = h0 + w * ∫ h'(σ) dσ
        let h_int = integrate_chebyshev_coeffs(&hdot_coeffs);

        // Step 5e: Second integral u' → position offset (4D)
        let u_int_3d = integrate_chebyshev_coeffs_3d(&up_int_3d);
        let u_int_4th = integrate_chebyshev_coeffs(&up_int_4th);

        // Step 5f: S_B = B(1) = sum of u_int coefficients (T_k(1) = 1)
        let mut s_b = [0.0_f64; 4];
        for k in 0..u_int_3d.len() {
            s_b[0] += u_int_3d[k][0];
            s_b[1] += u_int_3d[k][1];
            s_b[2] += u_int_3d[k][2];
        }
        for k in 0..u_int_4th.len() {
            s_b[3] += u_int_4th[k];
        }

        // Step 5g: Determine initial KS velocity from BCs
        // u'₀ = (u2 - u1 - w² S_B) / (2w)
        let up0 = [
            (u2[0] - u1[0] - w * w * s_b[0]) / (2.0 * w),
            (u2[1] - u1[1] - w * w * s_b[1]) / (2.0 * w),
            (u2[2] - u1[2] - w * w * s_b[2]) / (2.0 * w),
            (u2[3] - u1[3] - w * w * s_b[3]) / (2.0 * w),
        ];

        // Step 5h: Build u' coefficients: up_coeff = up0 + w * A_int
        let mut up_coeffs_3d: Vec<[f64; 3]> = Vec::with_capacity(n + 1);
        up_coeffs_3d.push([
            up0[0] + w * up_int_3d[0][0],
            up0[1] + w * up_int_3d[0][1],
            up0[2] + w * up_int_3d[0][2],
        ]);
        for k in 1..up_int_3d.len() {
            up_coeffs_3d.push([
                w * up_int_3d[k][0],
                w * up_int_3d[k][1],
                w * up_int_3d[k][2],
            ]);
        }
        let mut up_coeffs_4th: Vec<f64> = Vec::with_capacity(n + 1);
        up_coeffs_4th.push(up0[3] + w * up_int_4th[0]);
        for k in 1..up_int_4th.len() {
            up_coeffs_4th.push(w * up_int_4th[k]);
        }

        // h coefficients
        let mut h_coeffs: Vec<f64> = Vec::with_capacity(n + 1);
        h_coeffs.push(h0 + w * h_int[0]);
        for k in 1..h_int.len() {
            h_coeffs.push(w * h_int[k]);
        }

        // Step 5i: Integrate u' → position offset
        let v_int_3d = integrate_chebyshev_coeffs_3d(&up_coeffs_3d);
        let v_int_4th = integrate_chebyshev_coeffs(&up_coeffs_4th);

        // Build u coefficients: u_coeff = u1 + w * V_int
        let mut u_coeffs_3d: Vec<[f64; 3]> = Vec::with_capacity(n + 1);
        u_coeffs_3d.push([
            u1[0] + w * v_int_3d[0][0],
            u1[1] + w * v_int_3d[0][1],
            u1[2] + w * v_int_3d[0][2],
        ]);
        for k in 1..v_int_3d.len() {
            u_coeffs_3d.push([
                w * v_int_3d[k][0],
                w * v_int_3d[k][1],
                w * v_int_3d[k][2],
            ]);
        }
        let mut u_coeffs_4th: Vec<f64> = Vec::with_capacity(n + 1);
        u_coeffs_4th.push(u1[3] + w * v_int_4th[0]);
        for k in 1..v_int_4th.len() {
            u_coeffs_4th.push(w * v_int_4th[k]);
        }

        // Step 5j: Evaluate new state at all CGL nodes
        let mut new_u_nodes: Vec<KsVec4> = Vec::with_capacity(n + 1);
        let mut new_up_nodes: Vec<KsVec4> = Vec::with_capacity(n + 1);
        let mut new_h_nodes: Vec<f64> = Vec::with_capacity(n + 1);

        for j in 0..=n {
            let u_3 = eval_3d_at_node(&u_coeffs_3d, &t_all[j]);
            let u_4: f64 = u_coeffs_4th.iter().enumerate().map(|(k, &c)| c * t_all[j][k]).sum();
            new_u_nodes.push([u_3[0], u_3[1], u_3[2], u_4]);

            let up_3 = eval_3d_at_node(&up_coeffs_3d, &t_all[j]);
            let up_4: f64 = up_coeffs_4th.iter().enumerate().map(|(k, &c)| c * t_all[j][k]).sum();
            new_up_nodes.push([up_3[0], up_3[1], up_3[2], up_4]);

            let h_j: f64 = h_coeffs.iter().enumerate().map(|(k, &c)| c * t_all[j][k]).sum();
            new_h_nodes.push(h_j);
        }

        // Enforce BCs exactly: u(τ=-1) = u1 at j=n, u(τ=+1) = u2 at j=0
        new_u_nodes[n] = u1;
        new_u_nodes[0] = u2;
        new_up_nodes[n] = up0;

        // Step 5k: Convergence check (max KS position difference)
        let mut max_err: f64 = 0.0;
        for j in 0..=n {
            for dim in 0..4 {
                let diff = (new_u_nodes[j][dim] - u_nodes[j][dim]).abs();
                if diff > max_err {
                    max_err = diff;
                }
            }
        }

        u_nodes = new_u_nodes;
        up_nodes = new_up_nodes;
        h_nodes = new_h_nodes;

        if max_err < config.tolerance {
            converged = true;
            break;
        }
    }

    // ------------------------------------------------------------------
    // Step 6: Convert converged solution back to Cartesian
    // ------------------------------------------------------------------
    let v1 = ks_vel_to_cartesian(&u_nodes[n], &up_nodes[n]);
    let v2 = ks_vel_to_cartesian(&u_nodes[0], &up_nodes[0]);

    // Boundary error in Cartesian
    let r1_check = ks_to_cartesian(&u_nodes[n]);
    let r2_check = ks_to_cartesian(&u_nodes[0]);
    let bc_err = (r1_check - r1).norm().max((r2_check - r2).norm());

    KsTpbvpResult {
        v1,
        v2,
        converged,
        iterations_used,
        boundary_error: bc_err,
    }
}

/// Infer mu from position and velocity using vis-viva for v²/2 − μ/r = h:
/// We don't know h, but we know a from the Keplerian solver. However, for
/// simplicity we assume Earth's mu. The caller should ensure consistency.
fn compute_mu(r: &Vector3<f64>, v: &Vector3<f64>) -> f64 {
    // Use standard Earth mu. We could infer from orbital elements, but
    // the caller typically passes consistent (r, v, mu) from the Keplerian
    // solver anyway.
    let _ = (r, v);
    crate::constants::MU_EARTH
}

// =========================================================================
// 4D Chebyshev helper
// =========================================================================

fn eval_3d_at_node(coeffs: &[[f64; 3]], t_k_at_node: &[f64]) -> [f64; 3] {
    let m = coeffs.len().min(t_k_at_node.len());
    let mut result = [0.0; 3];
    for k in 0..m {
        let tk = t_k_at_node[k];
        result[0] += coeffs[k][0] * tk;
        result[1] += coeffs[k][1] * tk;
        result[2] += coeffs[k][2] * tk;
    }
    result
}

/// Extend evaluation for Chebyshev series with 4 components, using
/// 3D evaluation for the first three and a separate scalar for the 4th.
/// (Avoiding re-exporting 4D arrays keeps compatibility with existing 3D tooling.)

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::force_models::two_body::TwoBody;

    #[test]
    fn test_ks_roundtrip_equatorial() {
        let r = Vector3::new(7000.0, 0.0, 0.0);
        let u = cartesian_to_ks(&r);
        let r_back = ks_to_cartesian(&u);
        assert!((r_back - r).norm() < 1e-12, "roundtrip error: {}", (r_back - r).norm());
        assert!((ks_norm_sq(&u) - r.norm()).abs() < 1e-12);
    }

    #[test]
    fn test_ks_roundtrip_3d() {
        let r = Vector3::new(5000.0, 10000.0, 2100.0);
        let u = cartesian_to_ks(&r);
        let r_back = ks_to_cartesian(&u);
        assert!((r_back - r).norm() < 1e-10, "roundtrip error: {}", (r_back - r).norm());
    }

    #[test]
    fn test_ks_roundtrip_negative_x() {
        let r = Vector3::new(-8000.0, 3000.0, 1500.0);
        let u = cartesian_to_ks(&r);
        let r_back = ks_to_cartesian(&u);
        assert!((r_back - r).norm() < 1e-10, "roundtrip error for neg-x: {}", (r_back - r).norm());
    }

    #[test]
    fn test_ks_velocity_roundtrip() {
        let r = Vector3::new(7000.0, 0.0, 0.0);
        let v = Vector3::new(0.0, 7.5, 1.0);
        let u = cartesian_to_ks(&r);
        let up = cartesian_vel_to_ks(&u, &v);
        let v_back = ks_vel_to_cartesian(&u, &up);
        assert!(
            (v_back - v).norm() < 1e-10,
            "velocity roundtrip error: {}",
            (v_back - v).norm()
        );
    }

    #[test]
    fn test_ks_velocity_roundtrip_3d() {
        let r = Vector3::new(5000.0, 10000.0, 2100.0);
        let v = Vector3::new(-2.0, 4.0, 1.5);
        let u = cartesian_to_ks(&r);
        let up = cartesian_vel_to_ks(&u, &v);
        let v_back = ks_vel_to_cartesian(&u, &up);
        assert!(
            (v_back - v).norm() < 1e-10,
            "3D vel roundtrip error: {}",
            (v_back - v).norm()
        );
    }

    /// Under pure two-body, KS-TPBVP should match the Keplerian solver
    /// for a 90° transfer (well within the Cartesian TPBVP domain, verifying
    /// the KS machinery is equivalent).
    #[test]
    fn test_ks_tpbvp_two_body_short_arc() {
        let mu: f64 = 398600.4418;
        let r1 = Vector3::new(7000.0, 0.0, 0.0);
        let r2 = Vector3::new(0.0, 7000.0, 0.0);
        let tof = 2000.0;

        let input = crate::types::LambertInput {
            r1,
            r2,
            tof,
            mu,
            direction: crate::types::Direction::Prograde,
            max_revs: Some(0),
        };
        let sols = crate::keplerian::prussing::solve_prussing(&input).unwrap();
        let v1_kep = sols[0].v1;
        let a_kep = sols[0].a;

        let force = TwoBody::new(mu);
        let config = KsTpbvpConfig {
            poly_degree: 60,
            max_iterations: 30,
            tolerance: 1e-10,
        };
        let result = solve_ks_tpbvp(&r1, &r2, 0.0, tof, &v1_kep, a_kep, &force, &config);

        assert!(
            result.converged,
            "KS-TPBVP should converge for short arc ({} iters)",
            result.iterations_used
        );
        let v1_err = (result.v1 - v1_kep).norm();
        assert!(
            v1_err < 1e-4,
            "v1 error = {v1_err:.6e} km/s"
        );
    }

    /// Two-body medium arc (~150°): should converge where Cartesian TPBVP fails.
    #[test]
    fn test_ks_tpbvp_two_body_medium_arc() {
        let mu: f64 = 398600.4418;
        // 150° transfer angle
        let theta = 150.0_f64.to_radians();
        let r_mag = 7000.0;
        let r1 = Vector3::new(r_mag, 0.0, 0.0);
        let r2 = Vector3::new(r_mag * theta.cos(), r_mag * theta.sin(), 0.0);
        let tof = 3500.0;

        let input = crate::types::LambertInput {
            r1,
            r2,
            tof,
            mu,
            direction: crate::types::Direction::Prograde,
            max_revs: Some(0),
        };
        let sols = crate::keplerian::prussing::solve_prussing(&input).unwrap();
        let v1_kep = sols[0].v1;
        let a_kep = sols[0].a;

        let force = TwoBody::new(mu);
        let config = KsTpbvpConfig {
            poly_degree: 80,
            max_iterations: 50,
            tolerance: 1e-10,
        };
        let result = solve_ks_tpbvp(&r1, &r2, 0.0, tof, &v1_kep, a_kep, &force, &config);

        assert!(
            result.converged,
            "KS-TPBVP should converge for 150° arc ({} iters)",
            result.iterations_used
        );
        let v1_err = (result.v1 - v1_kep).norm();
        assert!(
            v1_err < 1e-3,
            "v1 error at 150° = {v1_err:.6e} km/s"
        );
    }

    /// 3D transfer with the KS-TPBVP solver.
    #[test]
    fn test_ks_tpbvp_two_body_3d() {
        let mu: f64 = 398600.4418;
        let r1 = Vector3::new(5000.0, 10000.0, 2100.0);
        let r2 = Vector3::new(-14600.0, 2500.0, 7000.0);
        let tof = 3600.0;

        let input = crate::types::LambertInput {
            r1,
            r2,
            tof,
            mu,
            direction: crate::types::Direction::Prograde,
            max_revs: Some(0),
        };
        let sols = crate::keplerian::prussing::solve_prussing(&input).unwrap();
        let v1_kep = sols[0].v1;
        let a_kep = sols[0].a;

        let force = TwoBody::new(mu);
        let config = KsTpbvpConfig {
            poly_degree: 80,
            max_iterations: 50,
            tolerance: 1e-10,
        };
        let result = solve_ks_tpbvp(&r1, &r2, 0.0, tof, &v1_kep, a_kep, &force, &config);

        assert!(
            result.converged,
            "KS-TPBVP 3D should converge ({} iters)",
            result.iterations_used
        );
        let v1_err = (result.v1 - v1_kep).norm();
        assert!(
            v1_err < 1e-3,
            "v1 error (3D) = {v1_err:.6e} km/s"
        );
    }
}
