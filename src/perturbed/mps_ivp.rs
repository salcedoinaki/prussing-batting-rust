//! MCPI-MPS-IVP: Method of Particular Solutions with MCPI as the IVP
//! integrator for multi-revolution perturbed Lambert transfers
//! (transfer angle ≥ ~1.8π, or any N ≥ 1).
//!
//! The MPS approach avoids constructing a state-transition matrix entirely.
//! Instead, it generates three IVP perturbation trajectories, computes a
//! linear correction, and iterates until the terminal position converges.
//!
//! Reference: Woollands et al., "Multiple Revolution Solutions for the
//! Perturbed Lambert Problem using the Method of Particular Solutions
//! and Picard Iteration" (J. Astronaut. Sci., 2017).

use nalgebra::{Matrix3, Vector3};

use crate::force_models::ForceModel;
use crate::perturbed::mcpi::{mcpi_propagate, McpiConfig};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the MCPI-MPS-IVP solver.
#[derive(Debug, Clone)]
pub struct MpsIvpConfig {
    /// MCPI polynomial degree for reference and particular solutions.
    pub poly_degree: usize,
    /// Maximum MCPI Picard iterations per propagation.
    pub max_mcpi_iterations: usize,
    /// MCPI convergence tolerance (position, km).
    pub mcpi_tolerance: f64,
    /// Maximum outer MPS iterations.
    pub max_mps_iterations: usize,
    /// Convergence tolerance on terminal position error (km).
    pub mps_tolerance: f64,
    /// Relative perturbation magnitude for particular solutions:
    ///   δv_j = perturbation_scale * ||v_ref|| * e_j
    pub perturbation_scale: f64,
    /// If true, use `acceleration_low_fidelity` for particular solutions.
    pub variable_fidelity: bool,
}

impl Default for MpsIvpConfig {
    fn default() -> Self {
        Self {
            poly_degree: 100,
            max_mcpi_iterations: 50,
            mcpi_tolerance: 1e-12,
            max_mps_iterations: 15,
            mps_tolerance: 1e-6, // sub-km terminal accuracy
            perturbation_scale: 1e-7,
            variable_fidelity: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Result
// ---------------------------------------------------------------------------

/// Result of the MCPI-MPS-IVP solver.
#[derive(Debug, Clone)]
pub struct MpsIvpResult {
    /// Departure velocity at r1 (km/s).
    pub v1: Vector3<f64>,
    /// Arrival velocity at r2 (km/s).
    pub v2: Vector3<f64>,
    /// Whether the MPS outer iteration converged.
    pub converged: bool,
    /// Number of outer MPS iterations performed.
    pub mps_iterations_used: usize,
    /// Terminal position error at the final iteration (km).
    pub terminal_error: f64,
}

// ---------------------------------------------------------------------------
// Force model wrapper for variable-fidelity propagation
// ---------------------------------------------------------------------------

/// Wraps a full-fidelity force model and delegates to its low-fidelity
/// acceleration method. Used to propagate particular solutions cheaply.
struct LowFidelityWrapper<'a> {
    inner: &'a dyn ForceModel,
}

impl<'a> ForceModel for LowFidelityWrapper<'a> {
    fn acceleration(&self, t: f64, r: &Vector3<f64>, v: &Vector3<f64>) -> Vector3<f64> {
        self.inner.acceleration_low_fidelity(t, r, v)
    }
}

// ---------------------------------------------------------------------------
// Solver
// ---------------------------------------------------------------------------

/// Solve the multi-revolution perturbed Lambert problem using the Method
/// of Particular Solutions with MCPI-IVP propagation.
///
/// # Arguments
/// * `r1` – Departure position (km)
/// * `r2` – Target arrival position (km)
/// * `t0` – Initial time
/// * `tf` – Final time (tof = tf - t0)
/// * `v0_ref` – Keplerian warm-start departure velocity (km/s)
/// * `force_model` – Perturbed force model
/// * `config` – MPS-IVP configuration
///
/// The `v0_ref` should come from the Keplerian multi-revolution solver
/// (Prussing algorithm) for the desired N-rev branch.
pub fn solve_mps_ivp(
    r1: &Vector3<f64>,
    r2: &Vector3<f64>,
    t0: f64,
    tf: f64,
    v0_ref: &Vector3<f64>,
    force_model: &dyn ForceModel,
    config: &MpsIvpConfig,
) -> MpsIvpResult {
    let mcpi_config = McpiConfig {
        poly_degree: config.poly_degree,
        max_iterations: config.max_mcpi_iterations,
        tolerance: config.mcpi_tolerance,
    };

    let low_fidelity_wrapper = LowFidelityWrapper { inner: force_model };
    let particular_force: &dyn ForceModel = if config.variable_fidelity {
        &low_fidelity_wrapper
    } else {
        force_model
    };

    let mut v_ref = *v0_ref;
    let mut converged = false;
    let mut mps_iterations_used = 0;
    let mut terminal_error = f64::MAX;
    let mut v2_final = Vector3::zeros();

    for iter in 0..config.max_mps_iterations {
        mps_iterations_used = iter + 1;

        // -----------------------------------------------------------------
        // Step 1: Propagate the reference trajectory (full fidelity)
        // -----------------------------------------------------------------
        let ref_state = mcpi_propagate(r1, &v_ref, t0, tf, force_model, &mcpi_config);
        // CGL node j=0 corresponds to τ=+1 which is t=tf
        let rf_ref = ref_state.positions[0];
        let vf_ref = ref_state.velocities[0];

        // Terminal miss
        let miss = r2 - rf_ref;
        terminal_error = miss.norm();

        if terminal_error < config.mps_tolerance {
            converged = true;
            v2_final = vf_ref;
            break;
        }

        // -----------------------------------------------------------------
        // Step 2: Generate three small orthogonal velocity perturbations
        // -----------------------------------------------------------------
        let v_mag = v_ref.norm();
        let delta = config.perturbation_scale * v_mag;
        // Guard against zero velocity (shouldn't happen for physical orbits)
        let delta = if delta < 1e-15 { 1e-10 } else { delta };

        // Axis-aligned perturbation directions. This is simpler than
        // constructing an orthonormal basis from v_ref and works well for
        // typical orbital mechanics problems where v_ref is never closely
        // aligned with a coordinate axis. If the resulting 3×3 matrix
        // becomes singular (rank-deficient), the LU solve below will
        // detect and handle it.
        let dv = [
            Vector3::new(delta, 0.0, 0.0),
            Vector3::new(0.0, delta, 0.0),
            Vector3::new(0.0, 0.0, delta),
        ];

        // -----------------------------------------------------------------
        // Step 3: Propagate three particular solutions
        // -----------------------------------------------------------------
        let mut delta_rf = [Vector3::zeros(); 3];

        for j in 0..3 {
            let v_pert = v_ref + dv[j];
            let pert_state = mcpi_propagate(
                r1,
                &v_pert,
                t0,
                tf,
                particular_force,
                &mcpi_config,
            );
            let rf_pert = pert_state.positions[0];

            // Departure motion: Eq. 11 from the paper
            delta_rf[j] = rf_pert - rf_ref;
        }

        // -----------------------------------------------------------------
        // Step 4: Solve 3×3 linear system for combination coefficients
        //   [δr1 | δr2 | δr3] · α = miss
        // -----------------------------------------------------------------
        let m = Matrix3::from_columns(&[delta_rf[0], delta_rf[1], delta_rf[2]]);

        // Use LU decomposition for the 3×3 solve
        let decomp = m.lu();
        let alpha = match decomp.solve(&miss) {
            Some(a) => a,
            None => {
                // Singular matrix — perturbations are linearly dependent.
                // This shouldn't happen with orthogonal perturbations, but
                // handle gracefully by breaking.
                break;
            }
        };

        // -----------------------------------------------------------------
        // Step 5: Update the initial velocity (Eq. 16)
        // -----------------------------------------------------------------
        v_ref += alpha.x * dv[0] + alpha.y * dv[1] + alpha.z * dv[2];
    }

    // If we exited without converging, do one final propagation to get v2
    if !converged {
        let final_state = mcpi_propagate(r1, &v_ref, t0, tf, force_model, &mcpi_config);
        v2_final = final_state.velocities[0];
        let rf = final_state.positions[0];
        terminal_error = (r2 - rf).norm();
        if terminal_error < config.mps_tolerance {
            converged = true;
        }
    }

    MpsIvpResult {
        v1: v_ref,
        v2: v2_final,
        converged,
        mps_iterations_used,
        terminal_error,
    }
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::force_models::two_body::TwoBody;
    use approx::assert_relative_eq;

    /// Under two-body dynamics, the MPS-IVP solver should recover the
    /// Keplerian solution (v1 unchanged, terminal position converged).
    #[test]
    fn test_mps_ivp_two_body_identity() {
        let mu = 398600.4418;
        let r1 = Vector3::new(7000.0, 0.0, 0.0);
        let r2 = Vector3::new(0.0, 7000.0, 0.0);
        let tof = 2000.0;

        // Get Keplerian solution
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
        let config = MpsIvpConfig {
            poly_degree: 80,
            max_mcpi_iterations: 30,
            mcpi_tolerance: 1e-10,
            max_mps_iterations: 10,
            mps_tolerance: 1e-6,
            perturbation_scale: 1e-7,
            variable_fidelity: false,
        };

        let result = solve_mps_ivp(&r1, &r2, 0.0, tof, &v1_kep, &force, &config);
        assert!(
            result.converged,
            "MPS-IVP two-body should converge, terminal_error = {:.6e}",
            result.terminal_error
        );

        // v1 should be very close to the Keplerian solution
        let v_err = (result.v1 - v1_kep).norm();
        assert!(
            v_err < 1e-4,
            "v1 should match Keplerian under two-body: error = {v_err:.6e}"
        );
    }

    /// Under two-body dynamics with a slightly perturbed warm start,
    /// MPS-IVP should still converge to the correct solution.
    #[test]
    fn test_mps_ivp_two_body_perturbed_warmstart() {
        let mu = 398600.4418;
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

        // Perturb the warm start by 1%
        let v1_perturbed = v1_kep * 1.01;

        let force = TwoBody::new(mu);
        let config = MpsIvpConfig {
            poly_degree: 80,
            max_mcpi_iterations: 30,
            mcpi_tolerance: 1e-10,
            max_mps_iterations: 15,
            mps_tolerance: 1e-3,
            perturbation_scale: 1e-5,
            variable_fidelity: false,
        };

        let result = solve_mps_ivp(&r1, &r2, 0.0, tof, &v1_perturbed, &force, &config);
        assert!(
            result.converged,
            "MPS-IVP should converge from perturbed start, terminal_error = {:.6e}",
            result.terminal_error
        );

        // Should recover close to Keplerian solution
        let v_err = (result.v1 - v1_kep).norm();
        assert_relative_eq!(result.v1.x, v1_kep.x, epsilon = 0.01);
        assert_relative_eq!(result.v1.y, v1_kep.y, epsilon = 0.01);
        assert_relative_eq!(result.v1.z, v1_kep.z, epsilon = 0.01);
        assert!(
            v_err < 0.05,
            "v1 should recover Keplerian: error = {v_err:.6e}"
        );
    }
}
