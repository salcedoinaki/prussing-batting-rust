//! MCPI-TPBVP: Two-Point Boundary Value Problem solver for short-arc
//! perturbed Lambert transfers (transfer angle < ~2π/3).
//!
//! The TPBVP formulation encodes both endpoint positions as boundary
//! conditions and determines the unknown initial velocity through Picard
//! iteration.  No shooting or state-transition matrices are required.

use nalgebra::Vector3;

use crate::force_models::ForceModel;
use crate::perturbed::chebyshev::{
    cgl_nodes, chebyshev_t_all, coefficients_from_nodes_3d, integrate_chebyshev_coeffs_3d,
    tau_to_time,
};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the MCPI-TPBVP solver.
#[derive(Debug, Clone)]
pub struct TpbvpConfig {
    /// Chebyshev polynomial degree (N CGL intervals, N+1 nodes).
    pub poly_degree: usize,
    /// Maximum number of Picard iterations.
    pub max_iterations: usize,
    /// Convergence tolerance on position (same units as r, typically km).
    pub tolerance: f64,
}

impl Default for TpbvpConfig {
    fn default() -> Self {
        Self {
            poly_degree: 80,
            max_iterations: 30,
            tolerance: 1e-10,
        }
    }
}

// ---------------------------------------------------------------------------
// Result
// ---------------------------------------------------------------------------

/// Result of the MCPI-TPBVP solver.
#[derive(Debug, Clone)]
pub struct TpbvpResult {
    /// Departure velocity at r1 (km/s).
    pub v1: Vector3<f64>,
    /// Arrival velocity at r2 (km/s).
    pub v2: Vector3<f64>,
    /// Whether the Picard iteration converged.
    pub converged: bool,
    /// Number of Picard iterations actually performed.
    pub iterations_used: usize,
    /// Maximum position error at the boundary nodes (km).
    pub boundary_error: f64,
}

// ---------------------------------------------------------------------------
// Solver
// ---------------------------------------------------------------------------

/// Solve the perturbed Lambert TPBVP using MCPI.
///
/// Given two position vectors `r1`, `r2` and a time of flight `[t0, tf]`,
/// find the departure and arrival velocities under the given force model.
///
/// The `v0_guess` (typically from a Keplerian solver) provides the initial
/// velocity estimate for the Picard iteration.
pub fn solve_tpbvp(
    r1: &Vector3<f64>,
    r2: &Vector3<f64>,
    t0: f64,
    tf: f64,
    v0_guess: &Vector3<f64>,
    force_model: &dyn ForceModel,
    config: &TpbvpConfig,
) -> TpbvpResult {
    let n = config.poly_degree;
    let tau = cgl_nodes(n);
    let w = (tf - t0) / 2.0; // half-interval

    // Physical times at CGL nodes (j=0 => τ=1 => tf, j=n => τ=-1 => t0)
    let times: Vec<f64> = tau.iter().map(|&tk| tau_to_time(tk, t0, tf)).collect();

    let r1v = [r1.x, r1.y, r1.z];
    let r2v = [r2.x, r2.y, r2.z];
    let v0g = [v0_guess.x, v0_guess.y, v0_guess.z];

    // ------------------------------------------------------------------
    // Initial guess: linear interpolation in position, constant velocity
    // ------------------------------------------------------------------
    // CGL ordering: j=0 is τ=1 (endpoint r2), j=n is τ=-1 (endpoint r1)
    let mut pos_nodes: Vec<[f64; 3]> = Vec::with_capacity(n + 1);
    let mut vel_nodes: Vec<[f64; 3]> = Vec::with_capacity(n + 1);
    for j in 0..=n {
        // fraction: 0.0 at τ=-1 (r1), 1.0 at τ=+1 (r2)
        let frac = (tau[j] + 1.0) / 2.0;
        pos_nodes.push([
            r1v[0] + frac * (r2v[0] - r1v[0]),
            r1v[1] + frac * (r2v[1] - r1v[1]),
            r1v[2] + frac * (r2v[2] - r1v[2]),
        ]);
        vel_nodes.push(v0g);
    }

    // Pre-compute T_k(τ_j) for all nodes and degrees
    let t_all: Vec<Vec<f64>> = tau.iter().map(|&t| chebyshev_t_all(n, t)).collect();

    let mut converged = false;
    let mut iterations_used = 0;
    let mut v0 = v0g;

    // ------------------------------------------------------------------
    // Picard iteration
    // ------------------------------------------------------------------
    for iter in 0..config.max_iterations {
        iterations_used = iter + 1;

        // Step 1: Evaluate acceleration at every CGL node
        let mut accel_nodes: Vec<[f64; 3]> = Vec::with_capacity(n + 1);
        for j in 0..=n {
            let rj = Vector3::new(pos_nodes[j][0], pos_nodes[j][1], pos_nodes[j][2]);
            let vj = Vector3::new(vel_nodes[j][0], vel_nodes[j][1], vel_nodes[j][2]);
            let a = force_model.acceleration(times[j], &rj, &vj);
            accel_nodes.push([a.x, a.y, a.z]);
        }

        // Step 2: Chebyshev coefficients of the acceleration
        let accel_coeffs = coefficients_from_nodes_3d(&accel_nodes, n);

        // Step 3: First integral A(τ) = ∫_{-1}^{τ} a(s) ds, with A(-1)=0
        let a_int = integrate_chebyshev_coeffs_3d(&accel_coeffs);

        // Step 4: Second integral B(τ) = ∫_{-1}^{τ} A(s) ds, with B(-1)=0
        let b_int = integrate_chebyshev_coeffs_3d(&a_int);

        // Step 5: S_B = B(1) = Σ B_coeff_k (since T_k(1) = 1 for all k)
        let mut s_b = [0.0_f64; 3];
        for coeff in &b_int {
            s_b[0] += coeff[0];
            s_b[1] += coeff[1];
            s_b[2] += coeff[2];
        }

        // Step 6: Determine initial velocity from boundary constraints
        //   v0 = (r2 - r1 - w² * S_B) / (2w)
        v0 = [
            (r2v[0] - r1v[0] - w * w * s_b[0]) / (2.0 * w),
            (r2v[1] - r1v[1] - w * w * s_b[1]) / (2.0 * w),
            (r2v[2] - r1v[2] - w * w * s_b[2]) / (2.0 * w),
        ];

        // Step 7: Build velocity Chebyshev coefficients
        //   vel_coeff[0] = v0 + w * A_coeff[0]
        //   vel_coeff[k] = w * A_coeff[k]  for k ≥ 1
        let mut vel_coeffs: Vec<[f64; 3]> = Vec::with_capacity(n + 1);
        vel_coeffs.push([
            v0[0] + w * a_int[0][0],
            v0[1] + w * a_int[0][1],
            v0[2] + w * a_int[0][2],
        ]);
        for k in 1..a_int.len() {
            vel_coeffs.push([w * a_int[k][0], w * a_int[k][1], w * a_int[k][2]]);
        }

        // Step 8: Integrate velocity → V(τ) = ∫_{-1}^{τ} v(s) ds
        let v_int = integrate_chebyshev_coeffs_3d(&vel_coeffs);

        // Step 9: Build position Chebyshev coefficients
        //   pos_coeff[0] = r1 + w * V_coeff[0]
        //   pos_coeff[k] = w * V_coeff[k]  for k ≥ 1
        let mut pos_coeffs: Vec<[f64; 3]> = Vec::with_capacity(n + 1);
        pos_coeffs.push([
            r1v[0] + w * v_int[0][0],
            r1v[1] + w * v_int[0][1],
            r1v[2] + w * v_int[0][2],
        ]);
        for k in 1..v_int.len() {
            pos_coeffs.push([w * v_int[k][0], w * v_int[k][1], w * v_int[k][2]]);
        }

        // Step 10: Evaluate new positions and velocities at CGL nodes
        let mut new_pos_nodes: Vec<[f64; 3]> = Vec::with_capacity(n + 1);
        let mut new_vel_nodes: Vec<[f64; 3]> = Vec::with_capacity(n + 1);
        for j in 0..=n {
            new_pos_nodes.push(eval_3d_at_node(&pos_coeffs, &t_all[j]));
            new_vel_nodes.push(eval_3d_at_node(&vel_coeffs, &t_all[j]));
        }

        // Step 11: Enforce boundary conditions exactly
        //   r(τ=-1) = r1 at node j=n
        //   r(τ=+1) = r2 at node j=0
        new_pos_nodes[n] = r1v;
        new_pos_nodes[0] = r2v;
        new_vel_nodes[n] = v0;

        // Step 12: Check convergence (max position change at all nodes)
        let mut max_err: f64 = 0.0;
        for j in 0..=n {
            for dim in 0..3 {
                let diff = (new_pos_nodes[j][dim] - pos_nodes[j][dim]).abs();
                if diff > max_err {
                    max_err = diff;
                }
            }
        }

        pos_nodes = new_pos_nodes;
        vel_nodes = new_vel_nodes;

        if max_err < config.tolerance {
            converged = true;
            break;
        }
    }

    // Compute boundary error for diagnostics
    let bc_err_r1 = ((pos_nodes[n][0] - r1v[0]).powi(2)
        + (pos_nodes[n][1] - r1v[1]).powi(2)
        + (pos_nodes[n][2] - r1v[2]).powi(2))
    .sqrt();
    let bc_err_r2 = ((pos_nodes[0][0] - r2v[0]).powi(2)
        + (pos_nodes[0][1] - r2v[1]).powi(2)
        + (pos_nodes[0][2] - r2v[2]).powi(2))
    .sqrt();
    let boundary_error = bc_err_r1.max(bc_err_r2);

    TpbvpResult {
        v1: Vector3::new(v0[0], v0[1], v0[2]),
        v2: Vector3::new(vel_nodes[0][0], vel_nodes[0][1], vel_nodes[0][2]),
        converged,
        iterations_used,
        boundary_error,
    }
}

// ---------------------------------------------------------------------------
// Helper
// ---------------------------------------------------------------------------

/// Evaluate a 3-component Chebyshev series at a node, using precomputed
/// T_k(τ_j) values.
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

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::force_models::two_body::TwoBody;

    /// Under pure two-body dynamics, the TPBVP should recover the same
    /// departure velocity as the Keplerian Lambert solver.
    #[test]
    fn test_tpbvp_two_body_matches_keplerian() {
        let mu: f64 = 398600.4418;
        let r1 = Vector3::new(7000.0, 0.0, 0.0);
        let r2 = Vector3::new(0.0, 7000.0, 0.0);
        let tof = 2000.0;

        // Get Keplerian solution for the warm start
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

        // Solve TPBVP under two-body — should converge to the same answer
        let force = TwoBody::new(mu);
        let config = TpbvpConfig {
            poly_degree: 60,
            max_iterations: 20,
            tolerance: 1e-10,
        };
        let result = solve_tpbvp(&r1, &r2, 0.0, tof, &v1_kep, &force, &config);

        assert!(
            result.converged,
            "TPBVP should converge under two-body in {} iterations",
            result.iterations_used
        );

        let v1_err = (result.v1 - v1_kep).norm();
        assert!(
            v1_err < 1e-6,
            "v1 error vs Keplerian = {v1_err:.6e} km/s (should be < 1e-6)"
        );
    }

    /// Under pure two-body, TPBVP with a non-coplanar transfer should
    /// also match the Keplerian solution.
    #[test]
    fn test_tpbvp_two_body_3d() {
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

        let force = TwoBody::new(mu);
        let config = TpbvpConfig {
            poly_degree: 80,
            max_iterations: 25,
            tolerance: 1e-10,
        };
        let result = solve_tpbvp(&r1, &r2, 0.0, tof, &v1_kep, &force, &config);

        assert!(result.converged, "TPBVP 3D should converge");

        let v1_err = (result.v1 - v1_kep).norm();
        assert!(
            v1_err < 1e-5,
            "v1 error = {v1_err:.6e} km/s"
        );
    }
}
