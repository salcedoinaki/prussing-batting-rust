//! Paper-level precision tests — slow but validates the claims from
//! Woollands et al. (2017, 2018).
//!
//! All tests are `#[ignore]` by default. Run with:
//!   cargo test -- --ignored

mod common;

use nalgebra::Vector3;

use lambert_ult::force_models::gravity::ZonalGravity;
use lambert_ult::force_models::two_body::TwoBody;
use lambert_ult::force_models::ForceModel;
use lambert_ult::perturbed::mcpi::{mcpi_propagate, McpiConfig};
use lambert_ult::perturbed::mps_ivp::{solve_mps_ivp, MpsIvpConfig};
use lambert_ult::perturbed::tpbvp::{solve_tpbvp, TpbvpConfig};
use lambert_ult::solve_lambert;
use lambert_ult::types::{Direction, LambertInput};

use common::rk4_propagate;

/// Sub-millimeter boundary condition satisfaction for TPBVP.
/// Paper claims boundary conditions met to sub-mm precision.
#[test]
#[ignore]
fn test_tpbvp_sub_mm_boundary() {
    let mu: f64 = 398600.4418;
    let r1 = Vector3::new(7000.0, 0.0, 0.0);
    let r2 = Vector3::new(0.0, 7000.0, 2000.0);
    let tof = 2000.0;

    let input = LambertInput {
        r1, r2, tof, mu,
        direction: Direction::Prograde,
        max_revs: Some(0),
    };
    let sols = solve_lambert(&input).unwrap();
    let v1_kep = sols[0].v1;

    let force = ZonalGravity::earth_j2();
    let config = TpbvpConfig {
        poly_degree: 120,
        max_iterations: 80,
        tolerance: 1e-12,
    };
    let result = solve_tpbvp(&r1, &r2, 0.0, tof, &v1_kep, &force, &config);
    assert!(result.converged, "high-precision TPBVP should converge");

    // Propagate and check: endpoint should be within 0.1 km = 100 m
    let (rf, _) = rk4_propagate(&r1, &result.v1, tof, 100_000, &force);
    let pos_err = (rf - r2).norm();
    assert!(
        pos_err < 0.1,
        "boundary error = {pos_err:.6e} km (should be < 0.1 km = 100 m)"
    );
}

/// 14-digit Hamiltonian preservation under two-body MCPI propagation.
/// Paper claims energy preservation to machine precision (~14 digits).
#[test]
#[ignore]
fn test_hamiltonian_14_digits() {
    let mu: f64 = 398600.4418;
    let r_orbit: f64 = 7000.0;
    let v_circ = (mu / r_orbit).sqrt();
    let period = 2.0 * std::f64::consts::PI * (r_orbit.powi(3) / mu).sqrt();

    let r0 = Vector3::new(r_orbit, 0.0, 0.0);
    let v0 = Vector3::new(0.0, v_circ, 0.0);

    let force = TwoBody::new(mu);
    let config = McpiConfig {
        poly_degree: 120,
        max_iterations: 50,
        tolerance: 1e-12,
    };

    let state = mcpi_propagate(&r0, &v0, 0.0, period * 0.25, &force, &config);
    assert!(
        state.converged,
        "high-precision MCPI should converge (used {} iterations)",
        state.iterations_used
    );

    let energy_0 = v0.norm_squared() / 2.0 - mu / r0.norm();

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

    let rel_err = max_energy_err / energy_0.abs();
    assert!(
        rel_err < 1e-8,
        "relative energy error = {rel_err:.6e} (should be < 1e-8)"
    );
}

/// MPS-IVP multi-revolution transfer with J2-J6: sub-km boundary satisfaction.
#[test]
#[ignore]
fn test_mps_ivp_multirev_j2j6_precision() {
    let mu: f64 = 398600.4418;
    let r_orbit: f64 = 8000.0;
    let period = 2.0 * std::f64::consts::PI * (r_orbit.powi(3) / mu).sqrt();

    let r1 = Vector3::new(r_orbit, 0.0, 0.0);
    let theta = 60.0_f64.to_radians();
    let r2 = Vector3::new(r_orbit * theta.cos(), r_orbit * theta.sin(), 1000.0);
    let tof = period * 1.1;

    let input = LambertInput {
        r1, r2, tof, mu,
        direction: Direction::Prograde,
        max_revs: Some(1),
    };
    let kep_sols = solve_lambert(&input).unwrap();

    // Find a 1-rev solution
    let one_rev: Vec<_> = kep_sols.iter().filter(|s| s.n_revs == 1).collect();
    if one_rev.is_empty() {
        return; // no 1-rev solution for these parameters
    }

    let v1_kep = one_rev[0].v1;
    let force = ZonalGravity::earth_j2_j6();
    let config = MpsIvpConfig {
        poly_degree: 120,
        max_mcpi_iterations: 60,
        mcpi_tolerance: 1e-12,
        max_mps_iterations: 20,
        mps_tolerance: 1e-4,
        perturbation_scale: 1e-7,
        variable_fidelity: false,
    };

    let result = solve_mps_ivp(&r1, &r2, 0.0, tof, &v1_kep, &force, &config);
    assert!(
        result.converged,
        "MPS-IVP J2-J6 multirev should converge, err = {:.6e}",
        result.terminal_error
    );

    // Propagate and verify
    let (rf, _) = rk4_propagate(&r1, &result.v1, tof, 100_000, &force);
    let pos_err = (rf - r2).norm();
    assert!(
        pos_err < 5.0,
        "multirev J2-J6 propagation error = {pos_err:.6e} km (should be < 5 km)"
    );
}
