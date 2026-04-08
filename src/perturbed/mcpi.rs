//! Modified Chebyshev-Picard Iteration (MCPI) engine for orbit propagation.
//!
//! The MCPI method represents the entire trajectory arc as a truncated
//! Chebyshev polynomial series evaluated at Chebyshev-Gauss-Lobatto (CGL)
//! nodes, then iteratively refines the series via Picard iteration.

use nalgebra::Vector3;

use crate::force_models::ForceModel;
use crate::perturbed::chebyshev::{cgl_nodes, chebyshev_t_all, tau_to_time};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration parameters for the MCPI propagator.
#[derive(Debug, Clone)]
pub struct McpiConfig {
    /// Number of CGL intervals (polynomial degree). N+1 nodes will be used.
    pub poly_degree: usize,
    /// Maximum number of Picard iterations.
    pub max_iterations: usize,
    /// Convergence tolerance on position (same units as `r`).
    pub tolerance: f64,
}

impl Default for McpiConfig {
    fn default() -> Self {
        Self {
            poly_degree: 80,
            max_iterations: 30,
            tolerance: 1e-12,
        }
    }
}

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

/// The converged (or best-effort) MCPI solution at the CGL nodes.
#[derive(Debug, Clone)]
pub struct McpiState {
    /// Position vectors at each CGL node (length N+1).
    pub positions: Vec<Vector3<f64>>,
    /// Velocity vectors at each CGL node (length N+1).
    pub velocities: Vec<Vector3<f64>>,
    /// Chebyshev coefficients for position (each element: [cx, cy, cz]).
    pub chebyshev_coeffs_r: Vec<[f64; 3]>,
    /// Chebyshev coefficients for velocity.
    pub chebyshev_coeffs_v: Vec<[f64; 3]>,
    /// Whether the iteration converged within tolerance.
    pub converged: bool,
    /// Number of Picard iterations actually performed.
    pub iterations_used: usize,
}

// ---------------------------------------------------------------------------
// MCPI IVP propagator
// ---------------------------------------------------------------------------

/// Propagate an initial-value problem (IVP) from `(r0, v0)` at `t0` to `tf`
/// under the given force model using MCPI.
///
/// Returns the full MCPI state, including positions and velocities at every
/// CGL node and the Chebyshev coefficients.
pub fn mcpi_propagate(
    r0: &Vector3<f64>,
    v0: &Vector3<f64>,
    t0: f64,
    tf: f64,
    force_model: &dyn ForceModel,
    config: &McpiConfig,
) -> McpiState {
    let n = config.poly_degree;
    let tau = cgl_nodes(n);
    let half_dt = (tf - t0) / 2.0;

    // Physical times at each CGL node
    let times: Vec<f64> = tau.iter().map(|&tk| tau_to_time(tk, t0, tf)).collect();

    // -----------------------------------------------------------------------
    // Initial guess: linear interpolation in position and constant velocity
    // -----------------------------------------------------------------------
    let r0v = [r0.x, r0.y, r0.z];
    let v0v = [v0.x, v0.y, v0.z];

    let mut pos_nodes: Vec<[f64; 3]> = Vec::with_capacity(n + 1);
    let mut vel_nodes: Vec<[f64; 3]> = Vec::with_capacity(n + 1);
    for j in 0..=n {
        let dt = times[j] - t0;
        pos_nodes.push([
            r0v[0] + v0v[0] * dt,
            r0v[1] + v0v[1] * dt,
            r0v[2] + v0v[2] * dt,
        ]);
        vel_nodes.push(v0v);
    }

    // -----------------------------------------------------------------------
    // Picard iteration
    // -----------------------------------------------------------------------
    let mut converged = false;
    let mut iterations_used = 0;

    // Pre-compute T_k(tau_j) for all nodes and degrees
    let t_all: Vec<Vec<f64>> = tau.iter().map(|&t| chebyshev_t_all(n, t)).collect();

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

        // Step 3: Integrate acceleration -> velocity coefficients
        //         v(tau) = v0 + half_dt * integral_{-1}^{tau} a(s) ds
        let vel_int = integrate_chebyshev_coeffs_3d(&accel_coeffs);
        let mut vel_coeffs_phys: Vec<[f64; 3]> = vel_int
            .iter()
            .map(|c| [c[0] * half_dt, c[1] * half_dt, c[2] * half_dt])
            .collect();
        vel_coeffs_phys[0][0] += v0v[0];
        vel_coeffs_phys[0][1] += v0v[1];
        vel_coeffs_phys[0][2] += v0v[2];

        // Step 4: Integrate velocity -> position coefficients
        //         r(tau) = r0 + half_dt * integral_{-1}^{tau} v(s) ds
        let pos_int = integrate_chebyshev_coeffs_3d(&vel_coeffs_phys);
        let mut pos_coeffs_phys: Vec<[f64; 3]> = pos_int
            .iter()
            .map(|c| [c[0] * half_dt, c[1] * half_dt, c[2] * half_dt])
            .collect();
        pos_coeffs_phys[0][0] += r0v[0];
        pos_coeffs_phys[0][1] += r0v[1];
        pos_coeffs_phys[0][2] += r0v[2];

        // Step 5: Evaluate new positions and velocities at CGL nodes
        let mut new_pos_nodes: Vec<[f64; 3]> = Vec::with_capacity(n + 1);
        let mut new_vel_nodes: Vec<[f64; 3]> = Vec::with_capacity(n + 1);
        for j in 0..=n {
            new_pos_nodes.push(eval_3d_at_node(&pos_coeffs_phys, &t_all[j]));
            new_vel_nodes.push(eval_3d_at_node(&vel_coeffs_phys, &t_all[j]));
        }

        // Step 6: Enforce the initial condition exactly
        // tau=-1 corresponds to j=n in the CGL ordering (descending).
        new_pos_nodes[n] = r0v;
        new_vel_nodes[n] = v0v;

        // Step 7: Check convergence (max position difference at all nodes)
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
            let final_pos_coeffs = coefficients_from_nodes_3d(&pos_nodes, n);
            let final_vel_coeffs = coefficients_from_nodes_3d(&vel_nodes, n);

            return McpiState {
                positions: pos_nodes
                    .iter()
                    .map(|p| Vector3::new(p[0], p[1], p[2]))
                    .collect(),
                velocities: vel_nodes
                    .iter()
                    .map(|v| Vector3::new(v[0], v[1], v[2]))
                    .collect(),
                chebyshev_coeffs_r: final_pos_coeffs,
                chebyshev_coeffs_v: final_vel_coeffs,
                converged,
                iterations_used,
            };
        }
    }

    // Did not converge — return best effort
    let final_pos_coeffs = coefficients_from_nodes_3d(&pos_nodes, n);
    let final_vel_coeffs = coefficients_from_nodes_3d(&vel_nodes, n);

    McpiState {
        positions: pos_nodes
            .iter()
            .map(|p| Vector3::new(p[0], p[1], p[2]))
            .collect(),
        velocities: vel_nodes
            .iter()
            .map(|v| Vector3::new(v[0], v[1], v[2]))
            .collect(),
        chebyshev_coeffs_r: final_pos_coeffs,
        chebyshev_coeffs_v: final_vel_coeffs,
        converged,
        iterations_used,
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Evaluate a 3-component Chebyshev series at a node, using precomputed
/// T_k(tau_j) values.
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

fn coefficients_from_nodes_3d(values: &[[f64; 3]], n: usize) -> Vec<[f64; 3]> {
    crate::perturbed::chebyshev::coefficients_from_nodes_3d(values, n)
}

fn integrate_chebyshev_coeffs_3d(coeffs: &[[f64; 3]]) -> Vec<[f64; 3]> {
    crate::perturbed::chebyshev::integrate_chebyshev_coeffs_3d(coeffs)
}

/// Evaluate a converged MCPI state at an arbitrary time `t ∈ [t0, tf]`
/// using Clenshaw summation on the Chebyshev coefficients.
pub fn evaluate_at(state: &McpiState, t: f64, t0: f64, tf: f64) -> (Vector3<f64>, Vector3<f64>) {
    let tau = crate::perturbed::chebyshev::time_to_tau(t, t0, tf);

    let r = crate::perturbed::chebyshev::clenshaw_3d(&state.chebyshev_coeffs_r, tau);
    let v = crate::perturbed::chebyshev::clenshaw_3d(&state.chebyshev_coeffs_v, tau);

    (
        Vector3::new(r[0], r[1], r[2]),
        Vector3::new(v[0], v[1], v[2]),
    )
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// A simple mu/r^3 two-body force model for testing.
    struct TwoBodyTest {
        mu: f64,
    }
    impl ForceModel for TwoBodyTest {
        fn acceleration(&self, _t: f64, r: &Vector3<f64>, _v: &Vector3<f64>) -> Vector3<f64> {
            let r_mag = r.norm();
            -self.mu / (r_mag * r_mag * r_mag) * r
        }
    }

    /// Propagate a circular orbit for one full period and verify the
    /// satellite returns to the starting position.
    #[test]
    fn test_circular_orbit_full_period() {
        let mu: f64 = 398600.4418;
        let r_orbit: f64 = 7000.0; // km
        let v_circ = (mu / r_orbit).sqrt();
        let period = 2.0 * std::f64::consts::PI * (r_orbit.powi(3) / mu).sqrt();

        let r0 = Vector3::new(r_orbit, 0.0, 0.0);
        let v0 = Vector3::new(0.0, v_circ, 0.0);

        let force = TwoBodyTest { mu };
        let config = McpiConfig {
            poly_degree: 80,
            max_iterations: 30,
            tolerance: 1e-10,
        };

        let state = mcpi_propagate(&r0, &v0, 0.0, period, &force, &config);

        assert!(
            state.converged,
            "MCPI did not converge in {} iterations",
            state.iterations_used
        );

        // Final state is at tau=1 which is CGL node j=0
        let rf = &state.positions[0];
        let err = (rf - r0).norm();
        assert!(
            err < 1e-4,
            "position error after full period = {err:.6e} km (expected < 1e-4)"
        );
    }

    /// Propagate a circular orbit for half a period and check the position
    /// is at the antipodal point.
    #[test]
    fn test_circular_orbit_half_period() {
        let mu: f64 = 398600.4418;
        let r_orbit: f64 = 7000.0;
        let v_circ = (mu / r_orbit).sqrt();
        let period = 2.0 * std::f64::consts::PI * (r_orbit.powi(3) / mu).sqrt();

        let r0 = Vector3::new(r_orbit, 0.0, 0.0);
        let v0 = Vector3::new(0.0, v_circ, 0.0);

        let force = TwoBodyTest { mu };
        let config = McpiConfig {
            poly_degree: 60,
            max_iterations: 30,
            tolerance: 1e-10,
        };

        let state = mcpi_propagate(&r0, &v0, 0.0, period / 2.0, &force, &config);

        assert!(state.converged);

        // After half period, should be at (-r_orbit, 0, 0)
        let rf = &state.positions[0];
        assert!(
            (rf.x + r_orbit).abs() < 1e-3,
            "x = {}, expected {}",
            rf.x,
            -r_orbit
        );
        assert!(rf.y.abs() < 1e-3, "y = {}, expected 0", rf.y);
        assert!(rf.z.abs() < 1e-3, "z = {}, expected 0", rf.z);
    }

    /// Verify Hamiltonian (energy) conservation over the arc.
    #[test]
    fn test_energy_conservation() {
        let mu: f64 = 398600.4418;
        let r_orbit: f64 = 7000.0;
        let v_circ = (mu / r_orbit).sqrt();
        let period = 2.0 * std::f64::consts::PI * (r_orbit.powi(3) / mu).sqrt();

        let r0 = Vector3::new(r_orbit, 0.0, 0.0);
        let v0 = Vector3::new(0.0, v_circ, 0.0);

        let force = TwoBodyTest { mu };
        let config = McpiConfig {
            poly_degree: 100,
            max_iterations: 40,
            tolerance: 1e-10,
        };

        let state = mcpi_propagate(&r0, &v0, 0.0, period, &force, &config);
        assert!(
            state.converged,
            "MCPI did not converge in {} iterations",
            state.iterations_used
        );

        let energy_0 = v0.norm_squared() / 2.0 - mu / r0.norm();

        // Check energy at every CGL node
        let mut max_energy_err: f64 = 0.0;
        for j in 0..state.positions.len() {
            let rj = &state.positions[j];
            let vj = &state.velocities[j];
            let e_j = vj.norm_squared() / 2.0 - mu / rj.norm();
            let err = (e_j - energy_0).abs();
            if err > max_energy_err {
                max_energy_err = err;
            }
        }

        assert!(
            max_energy_err < 1e-6,
            "max energy error = {max_energy_err:.6e} (should be < 1e-6)"
        );
    }

    /// Propagate an eccentric orbit (e=0.3) for one period.
    #[test]
    fn test_eccentric_orbit() {
        let mu: f64 = 398600.4418;
        let a: f64 = 10000.0; // km
        let e: f64 = 0.3;
        let rp = a * (1.0 - e); // periapsis
        let vp = (mu * (1.0 + e) / (a * (1.0 - e))).sqrt(); // velocity at periapsis
        let period = 2.0 * std::f64::consts::PI * (a.powi(3) / mu).sqrt();

        let r0 = Vector3::new(rp, 0.0, 0.0);
        let v0 = Vector3::new(0.0, vp, 0.0);

        let force = TwoBodyTest { mu };
        let config = McpiConfig {
            poly_degree: 100,
            max_iterations: 40,
            tolerance: 1e-10,
        };

        let state = mcpi_propagate(&r0, &v0, 0.0, period, &force, &config);

        assert!(
            state.converged,
            "MCPI did not converge for e=0.3 orbit in {} iterations",
            state.iterations_used
        );

        let rf = &state.positions[0];
        let err = (rf - r0).norm();
        assert!(
            err < 1e-2,
            "position error after full period = {err:.6e} km (e=0.3)"
        );
    }

    /// Test `evaluate_at` for an intermediate time.
    #[test]
    fn test_evaluate_at_intermediate() {
        let mu: f64 = 398600.4418;
        let r_orbit: f64 = 7000.0;
        let v_circ = (mu / r_orbit).sqrt();
        let period = 2.0 * std::f64::consts::PI * (r_orbit.powi(3) / mu).sqrt();

        let r0 = Vector3::new(r_orbit, 0.0, 0.0);
        let v0 = Vector3::new(0.0, v_circ, 0.0);

        let force = TwoBodyTest { mu };
        let config = McpiConfig {
            poly_degree: 60,
            max_iterations: 25,
            tolerance: 1e-10,
        };

        let t_end = period / 4.0; // quarter period
        let state = mcpi_propagate(&r0, &v0, 0.0, t_end, &force, &config);
        assert!(state.converged);

        // At t = T/4, a circular orbit should be at (0, r_orbit, 0)
        let (rf, vf) = evaluate_at(&state, t_end, 0.0, t_end);
        assert!(
            (rf.x).abs() < 1e-2,
            "x at T/4 = {}, expected ~0",
            rf.x
        );
        assert!(
            (rf.y - r_orbit).abs() < 1e-2,
            "y at T/4 = {}, expected {}",
            rf.y,
            r_orbit
        );

        // Velocity at T/4 should be (-v_circ, 0, 0) approximately
        assert!(
            (vf.x + v_circ).abs() < 1e-2,
            "vx at T/4 = {}, expected {}",
            vf.x,
            -v_circ
        );
        assert!(
            vf.y.abs() < 1e-2,
            "vy at T/4 = {}, expected 0",
            vf.y
        );
    }
}
