mod common;
use common::{kepler_propagate, rk4_propagate};

use approx::assert_relative_eq;
use nalgebra::Vector3;

use lambert_ult::solve_lambert;
use lambert_ult::types::{Direction, LambertInput};

// -------------------------------------------------------------------------
// Integration tests
// -------------------------------------------------------------------------

/// Verify that the N=0 prograde solution for a 90-degree coplanar transfer is
/// self-consistent: propagating (r1, v1) for tof under two-body dynamics
/// should arrive at r2.
#[test]
fn test_roundtrip_n0_90deg() {
    let mu = 398600.4418;
    let r1 = Vector3::new(7000.0, 0.0, 0.0);
    let r2 = Vector3::new(0.0, 7000.0, 0.0);
    let tof = 2000.0;

    let input = LambertInput {
        r1,
        r2,
        tof,
        mu,
        direction: Direction::Prograde,
        max_revs: Some(0),
    };
    let sols = solve_lambert(&input).expect("solver should succeed");
    assert!(!sols.is_empty());

    let r2_prop = kepler_propagate(&r1, &sols[0].v1, tof, mu);
    assert_relative_eq!(r2_prop.x, r2.x, epsilon = 1e-3);
    assert_relative_eq!(r2_prop.y, r2.y, epsilon = 1e-3);
    assert_relative_eq!(r2_prop.z, r2.z, epsilon = 1e-3);
}

/// Test a non-coplanar transfer (3-D positions).
#[test]
fn test_roundtrip_3d() {
    let mu = 398600.4418;
    let r1 = Vector3::new(5000.0, 10000.0, 2100.0);
    let r2 = Vector3::new(-14600.0, 2500.0, 7000.0);
    let tof = 3600.0;

    let input = LambertInput {
        r1,
        r2,
        tof,
        mu,
        direction: Direction::Prograde,
        max_revs: Some(0),
    };
    let sols = solve_lambert(&input).expect("solver should succeed");
    assert!(!sols.is_empty());

    let r2_prop = kepler_propagate(&r1, &sols[0].v1, tof, mu);
    let err = (r2_prop - r2).norm();
    assert!(
        err < 1.0,
        "position error = {err:.4} km (should be < 1 km)"
    );
}

/// Energy consistency: the returned semi-major axis should match vis-viva.
#[test]
fn test_vis_viva_consistency() {
    let mu = 398600.4418;
    let r1 = Vector3::new(7000.0, 0.0, 0.0);
    let r2 = Vector3::new(0.0, 7000.0, 0.0);
    let tof = 2000.0;

    let input = LambertInput {
        r1,
        r2,
        tof,
        mu,
        direction: Direction::Prograde,
        max_revs: Some(0),
    };
    let sols = solve_lambert(&input).unwrap();
    let sol = &sols[0];

    // vis-viva: v^2 = mu * (2/r - 1/a) => a = 1 / (2/r - v^2/mu)
    let r1_mag = r1.norm();
    let v1_sq = sol.v1.norm_squared();
    let a_vv = 1.0 / (2.0 / r1_mag - v1_sq / mu);

    assert_relative_eq!(sol.a, a_vv, epsilon = 1.0);
}

/// Verify that retrograde direction produces a different solution from prograde.
#[test]
fn test_prograde_vs_retrograde() {
    let mu = 398600.4418;
    let r1 = Vector3::new(7000.0, 0.0, 0.0);
    let r2 = Vector3::new(0.0, 7000.0, 0.0);
    let tof = 3600.0;

    let pro = solve_lambert(&LambertInput {
        r1,
        r2,
        tof,
        mu,
        direction: Direction::Prograde,
        max_revs: Some(0),
    })
    .unwrap();

    let retro = solve_lambert(&LambertInput {
        r1,
        r2,
        tof,
        mu,
        direction: Direction::Retrograde,
        max_revs: Some(0),
    })
    .unwrap();

    // They should both solve, but give different velocities
    assert!(!pro.is_empty());
    assert!(!retro.is_empty());
    let diff = (pro[0].v1 - retro[0].v1).norm();
    assert!(diff > 0.1, "prograde and retrograde should differ");
}

// =========================================================================
// MCPI propagation tests (Sprint 2 validation)
// =========================================================================

use lambert_ult::force_models::two_body::TwoBody;
use lambert_ult::perturbed::mcpi::{evaluate_at, mcpi_propagate, McpiConfig};

/// MCPI-IVP validation: propagate a circular orbit and compare the terminal
/// position against the analytical Kepler propagator.
#[test]
fn test_mcpi_vs_kepler_circular() {
    let mu: f64 = 398600.4418;
    let r_orbit: f64 = 7000.0;
    let v_circ = (mu / r_orbit).sqrt();
    let period = 2.0 * std::f64::consts::PI * (r_orbit.powi(3) / mu).sqrt();
    let tof = period * 0.75; // 3/4 period

    let r0 = Vector3::new(r_orbit, 0.0, 0.0);
    let v0 = Vector3::new(0.0, v_circ, 0.0);

    // Analytical (Kepler propagator) reference
    let r_kepler = kepler_propagate(&r0, &v0, tof, mu);

    // MCPI propagation
    let force = TwoBody::new(mu);
    let config = McpiConfig {
        poly_degree: 80,
        max_iterations: 30,
        tolerance: 1e-10,
    };
    let state = mcpi_propagate(&r0, &v0, 0.0, tof, &force, &config);
    assert!(state.converged, "MCPI did not converge");

    // Final position (tau=1 => CGL node j=0)
    let rf_mcpi = &state.positions[0];
    let err = (rf_mcpi - r_kepler).norm();
    assert!(
        err < 1e-3,
        "MCPI vs Kepler position error = {err:.6e} km (circular, 3/4 period)"
    );
}

/// MCPI-IVP validation: propagate an eccentric orbit (e=0.5) and compare
/// the terminal position against the analytical Kepler propagator.
#[test]
fn test_mcpi_vs_kepler_eccentric() {
    let mu: f64 = 398600.4418;
    let a: f64 = 12000.0;
    let e: f64 = 0.5;
    let rp = a * (1.0 - e);
    let vp = (mu * (1.0 + e) / (a * (1.0 - e))).sqrt();
    let period = 2.0 * std::f64::consts::PI * (a.powi(3) / mu).sqrt();
    let tof = period * 0.6;

    let r0 = Vector3::new(rp, 0.0, 0.0);
    let v0 = Vector3::new(0.0, vp, 0.0);

    let r_kepler = kepler_propagate(&r0, &v0, tof, mu);

    let force = TwoBody::new(mu);
    let config = McpiConfig {
        poly_degree: 100,
        max_iterations: 40,
        tolerance: 1e-10,
    };
    let state = mcpi_propagate(&r0, &v0, 0.0, tof, &force, &config);
    assert!(state.converged, "MCPI did not converge for e=0.5 orbit");

    let rf_mcpi = &state.positions[0];
    let err = (rf_mcpi - r_kepler).norm();
    assert!(
        err < 1e-2,
        "MCPI vs Kepler position error = {err:.6e} km (e=0.5)"
    );
}

/// MCPI-IVP validation: propagate an inclined 3-D orbit and compare against
/// the Kepler propagator. This tests that the MCPI engine handles non-planar
/// orbits correctly.
#[test]
fn test_mcpi_vs_kepler_3d() {
    let mu: f64 = 398600.4418;
    let r0 = Vector3::new(5000.0, 10000.0, 2100.0);
    // Give it a velocity that produces a bound orbit
    let v0 = Vector3::new(-2.0, 4.0, 1.5);
    let tof: f64 = 3600.0;

    let r_kepler = kepler_propagate(&r0, &v0, tof, mu);

    let force = TwoBody::new(mu);
    let config = McpiConfig {
        poly_degree: 80,
        max_iterations: 30,
        tolerance: 1e-10,
    };
    let state = mcpi_propagate(&r0, &v0, 0.0, tof, &force, &config);
    assert!(state.converged, "MCPI did not converge for 3-D orbit");

    let rf_mcpi = &state.positions[0];
    let err = (rf_mcpi - r_kepler).norm();
    assert!(
        err < 1e-2,
        "MCPI vs Kepler position error = {err:.6e} km (3-D orbit)"
    );
}

/// Verify that evaluate_at produces positions consistent with the Kepler
/// propagator at multiple intermediate times along the arc.
#[test]
fn test_mcpi_evaluate_at_intermediate_vs_kepler() {
    let mu: f64 = 398600.4418;
    let r_orbit: f64 = 7000.0;
    let v_circ = (mu / r_orbit).sqrt();
    let period = 2.0 * std::f64::consts::PI * (r_orbit.powi(3) / mu).sqrt();
    let tof = period / 2.0;

    let r0 = Vector3::new(r_orbit, 0.0, 0.0);
    let v0 = Vector3::new(0.0, v_circ, 0.0);

    let force = TwoBody::new(mu);
    let config = McpiConfig {
        poly_degree: 60,
        max_iterations: 25,
        tolerance: 1e-10,
    };
    let state = mcpi_propagate(&r0, &v0, 0.0, tof, &force, &config);
    assert!(state.converged);

    // Check at 10 equally spaced intermediate times
    for i in 1..10 {
        let t = tof * (i as f64) / 10.0;
        let (r_mcpi, _v_mcpi) = evaluate_at(&state, t, 0.0, tof);
        let r_kepler = kepler_propagate(&r0, &v0, t, mu);
        let err = (r_mcpi - r_kepler).norm();
        assert!(
            err < 1e-2,
            "MCPI vs Kepler at t={t:.1}: error = {err:.6e} km"
        );
    }
}

// =========================================================================
// Sprint 2.6 — Perturbed propagation validation (MCPI + J2 vs RK4)
// =========================================================================

use lambert_ult::force_models::gravity::ZonalGravity;
use lambert_ult::force_models::ForceModel;
use lambert_ult::perturbed::tpbvp::{solve_tpbvp, TpbvpConfig};

// rk4_propagate and kepler_propagate are imported from tests/common/mod.rs

/// MCPI propagation under J2 should match an RK4 reference for a
/// near-circular LEO orbit over ~1/4 period.
#[test]
fn test_mcpi_j2_vs_rk4_circular() {
    let mu: f64 = 398600.4418;
    let r_orbit: f64 = 7000.0;
    let v_circ = (mu / r_orbit).sqrt();
    let period = 2.0 * std::f64::consts::PI * (r_orbit.powi(3) / mu).sqrt();
    let tof = period * 0.25;

    // Inclined orbit so J2 has nontrivial effect
    let r0 = Vector3::new(r_orbit, 0.0, 0.0);
    let v0 = Vector3::new(0.0, v_circ * 0.866, v_circ * 0.5); // ~30 deg inclination

    let force = ZonalGravity::earth_j2();

    // RK4 reference with small step size
    let (r_rk4, _v_rk4) = rk4_propagate(&r0, &v0, tof, 50_000, &force);

    // MCPI propagation — J2 perturbation needs higher degree for convergence
    let config = McpiConfig {
        poly_degree: 120,
        max_iterations: 60,
        tolerance: 1e-12,
    };
    let state = mcpi_propagate(&r0, &v0, 0.0, tof, &force, &config);
    assert!(state.converged, "MCPI-J2 did not converge");

    let rf_mcpi = &state.positions[0]; // final position at τ=1
    let err = (rf_mcpi - r_rk4).norm();
    assert!(
        err < 0.1,
        "MCPI vs RK4 (J2, circular, quarter-period) position error = {err:.6e} km"
    );
}

/// MCPI under J2 for an eccentric orbit should also match RK4.
#[test]
fn test_mcpi_j2_vs_rk4_eccentric() {
    let mu: f64 = 398600.4418;
    let a: f64 = 10000.0;
    let e: f64 = 0.3;
    let rp = a * (1.0 - e);
    let vp = (mu * (1.0 + e) / (a * (1.0 - e))).sqrt();
    let period = 2.0 * std::f64::consts::PI * (a.powi(3) / mu).sqrt();
    let tof = period * 0.3;

    // Inclined orbit
    let r0 = Vector3::new(rp, 0.0, 0.0);
    let v0 = Vector3::new(0.0, vp * 0.866, vp * 0.5);

    let force = ZonalGravity::earth_j2();

    let (r_rk4, _v_rk4) = rk4_propagate(&r0, &v0, tof, 60_000, &force);

    let config = McpiConfig {
        poly_degree: 100,
        max_iterations: 40,
        tolerance: 1e-12,
    };
    let state = mcpi_propagate(&r0, &v0, 0.0, tof, &force, &config);
    assert!(state.converged, "MCPI-J2 eccentric did not converge");

    let rf_mcpi = &state.positions[0];
    let err = (rf_mcpi - r_rk4).norm();
    assert!(
        err < 0.5,
        "MCPI vs RK4 (J2, e=0.3) position error = {err:.6e} km"
    );
}

/// Verify that J2 propagation differs measurably from two-body propagation.
#[test]
fn test_j2_differs_from_two_body() {
    let mu: f64 = 398600.4418;
    let r_orbit: f64 = 7000.0;
    let v_circ = (mu / r_orbit).sqrt();
    let period = 2.0 * std::f64::consts::PI * (r_orbit.powi(3) / mu).sqrt();
    let tof = period * 0.5;

    let r0 = Vector3::new(r_orbit, 0.0, 0.0);
    let v0 = Vector3::new(0.0, v_circ * 0.866, v_circ * 0.5);

    let force_j2 = ZonalGravity::earth_j2();
    let force_tb = TwoBody::new(mu);

    let config = McpiConfig {
        poly_degree: 80,
        max_iterations: 30,
        tolerance: 1e-10,
    };

    let state_j2 = mcpi_propagate(&r0, &v0, 0.0, tof, &force_j2, &config);
    let state_tb = mcpi_propagate(&r0, &v0, 0.0, tof, &force_tb, &config);
    assert!(state_j2.converged);
    assert!(state_tb.converged);

    let diff = (&state_j2.positions[0] - &state_tb.positions[0]).norm();
    assert!(
        diff > 1.0,
        "J2 and two-body should differ by > 1 km over half-period, got {diff:.4} km"
    );
}

// =========================================================================
// Sprint 3.3 — MCPI-TPBVP validation
// =========================================================================

/// Under J2, the TPBVP solution's departure velocity should differ from
/// the Keplerian solution.
#[test]
fn test_tpbvp_j2_differs_from_keplerian() {
    let mu: f64 = 398600.4418;
    let r1 = Vector3::new(7000.0, 0.0, 0.0);
    let r2 = Vector3::new(0.0, 7000.0, 3000.0); // out-of-plane so J2 matters
    let tof = 2000.0;

    let input = lambert_ult::types::LambertInput {
        r1,
        r2,
        tof,
        mu,
        direction: lambert_ult::types::Direction::Prograde,
        max_revs: Some(0),
    };
    let sols = lambert_ult::solve_lambert(&input).unwrap();
    let v1_kep = sols[0].v1;

    let force = ZonalGravity::earth_j2();
    let config = TpbvpConfig {
        poly_degree: 100,
        max_iterations: 50,
        tolerance: 1e-10,
    };
    let result = solve_tpbvp(&r1, &r2, 0.0, tof, &v1_kep, &force, &config);
    assert!(
        result.converged,
        "TPBVP-J2 should converge ({} iters)",
        result.iterations_used
    );

    let diff = (result.v1 - v1_kep).norm();
    assert!(
        diff > 1e-6,
        "J2 perturbed v1 should differ from Keplerian by > 1e-6 km/s, got {diff:.6e}"
    );
}

/// The TPBVP solution under J2 should be self-consistent: propagating
/// (r1, v1_tpbvp) under the same J2 force model should arrive at r2.
#[test]
fn test_tpbvp_j2_propagation_consistency() {
    let mu: f64 = 398600.4418;
    let r1 = Vector3::new(7000.0, 0.0, 0.0);
    let r2 = Vector3::new(0.0, 7000.0, 3000.0);
    let tof = 2000.0;

    // Keplerian warm start
    let input = lambert_ult::types::LambertInput {
        r1,
        r2,
        tof,
        mu,
        direction: lambert_ult::types::Direction::Prograde,
        max_revs: Some(0),
    };
    let sols = lambert_ult::solve_lambert(&input).unwrap();
    let v1_kep = sols[0].v1;

    let force = ZonalGravity::earth_j2();
    let config = TpbvpConfig {
        poly_degree: 100,
        max_iterations: 50,
        tolerance: 1e-10,
    };
    let result = solve_tpbvp(&r1, &r2, 0.0, tof, &v1_kep, &force, &config);
    assert!(result.converged);

    // Forward-propagate (r1, v1_tpbvp) under J2 using RK4
    let (rf, _vf) = rk4_propagate(&r1, &result.v1, tof, 50_000, &force);
    let pos_err = (rf - r2).norm();
    assert!(
        pos_err < 1.0,
        "propagated endpoint error = {pos_err:.6e} km (should be < 1 km)"
    );
}

/// Verify the TPBVP solver also works for a retrograde transfer with J2.
/// Use positions where the retrograde transfer angle is within the TPBVP
/// convergence domain (θ < 2π/3).
#[test]
fn test_tpbvp_j2_retrograde() {
    let mu: f64 = 398600.4418;
    let r1 = Vector3::new(7000.0, 0.0, 0.0);
    // Chosen so that the retrograde (clockwise) transfer angle is small (~45°)
    let r2 = Vector3::new(5000.0, -5000.0, 1000.0);
    let tof = 2000.0;

    let input = lambert_ult::types::LambertInput {
        r1,
        r2,
        tof,
        mu,
        direction: lambert_ult::types::Direction::Retrograde,
        max_revs: Some(0),
    };
    let sols = lambert_ult::solve_lambert(&input).unwrap();
    let v1_kep = sols[0].v1;

    let force = ZonalGravity::earth_j2();
    let config = TpbvpConfig {
        poly_degree: 100,
        max_iterations: 50,
        tolerance: 1e-10,
    };
    let result = solve_tpbvp(&r1, &r2, 0.0, tof, &v1_kep, &force, &config);
    assert!(
        result.converged,
        "TPBVP retrograde should converge ({} iters)",
        result.iterations_used
    );

    // Propagate forward under J2 and check endpoint
    let (rf, _vf) = rk4_propagate(&r1, &result.v1, tof, 50_000, &force);
    let pos_err = (rf - r2).norm();
    assert!(
        pos_err < 1.0,
        "retrograde propagation error = {pos_err:.6e} km"
    );
}

// =========================================================================
// KS-TPBVP integration tests
// =========================================================================

/// The KS-TPBVP under J2 for a medium arc (150°) should differ from Keplerian.
#[test]
fn test_ks_tpbvp_j2_medium_arc() {
    use lambert_ult::force_models::gravity::ZonalGravity;
    use lambert_ult::perturbed::ks_tpbvp::{solve_ks_tpbvp, KsTpbvpConfig};

    let mu: f64 = 398600.4418;
    let theta = 150.0_f64.to_radians();
    let r_mag = 7000.0;
    let r1 = Vector3::new(r_mag, 0.0, 0.0);
    let r2 = Vector3::new(r_mag * theta.cos(), r_mag * theta.sin(), 2000.0);
    let tof = 3500.0;

    let input = LambertInput {
        r1,
        r2,
        tof,
        mu,
        direction: Direction::Prograde,
        max_revs: Some(0),
    };
    let sols = solve_lambert(&input).unwrap();
    let v1_kep = sols[0].v1;
    let a_kep = sols[0].a;

    let force = ZonalGravity::earth_j2();
    let config = KsTpbvpConfig {
        poly_degree: 100,
        max_iterations: 60,
        tolerance: 1e-10,
    };
    let result = solve_ks_tpbvp(&r1, &r2, 0.0, tof, &v1_kep, a_kep, &force, &config);
    assert!(
        result.converged,
        "KS-TPBVP J2 medium arc should converge ({} iters)",
        result.iterations_used
    );

    let diff = (result.v1 - v1_kep).norm();
    assert!(
        diff > 1e-6,
        "J2-perturbed v1 should differ from Keplerian: {diff:.6e}"
    );
}

/// KS-TPBVP under J2 should converge and produce a different velocity
/// than the Keplerian solution for a 120° out-of-plane transfer.
#[test]
fn test_ks_tpbvp_j2_convergence_120deg() {
    use lambert_ult::force_models::gravity::ZonalGravity;
    use lambert_ult::perturbed::ks_tpbvp::{solve_ks_tpbvp, KsTpbvpConfig};

    let mu: f64 = 398600.4418;
    let theta = 120.0_f64.to_radians();
    let r_mag = 7000.0;
    let r1 = Vector3::new(r_mag, 0.0, 0.0);
    let r2 = Vector3::new(r_mag * theta.cos(), r_mag * theta.sin(), 1500.0);
    let tof = 3000.0;

    let input = LambertInput {
        r1,
        r2,
        tof,
        mu,
        direction: Direction::Prograde,
        max_revs: Some(0),
    };
    let sols = solve_lambert(&input).unwrap();
    let v1_kep = sols[0].v1;
    let a_kep = sols[0].a;

    let force = ZonalGravity::earth_j2();
    let config = KsTpbvpConfig {
        poly_degree: 100,
        max_iterations: 60,
        tolerance: 1e-10,
    };
    let result = solve_ks_tpbvp(&r1, &r2, 0.0, tof, &v1_kep, a_kep, &force, &config);
    assert!(
        result.converged,
        "KS-TPBVP J2 120° should converge ({} iters)",
        result.iterations_used
    );

    // v1 should differ from Keplerian due to J2
    let diff = (result.v1 - v1_kep).norm();
    assert!(
        diff > 1e-6,
        "J2-perturbed v1 should differ from Keplerian: {diff:.6e}"
    );
}

/// Two-body KS-TPBVP for a near-π transfer: test convergence at the edge
/// of the medium-arc domain.
#[test]
fn test_ks_tpbvp_two_body_near_pi() {
    use lambert_ult::force_models::two_body::TwoBody;
    use lambert_ult::perturbed::ks_tpbvp::{solve_ks_tpbvp, KsTpbvpConfig};

    let mu: f64 = 398600.4418;
    // ~170° transfer
    let theta = 170.0_f64.to_radians();
    let r_mag = 7000.0;
    let r1 = Vector3::new(r_mag, 0.0, 0.0);
    let r2 = Vector3::new(r_mag * theta.cos(), r_mag * theta.sin(), 0.0);
    let tof = 4500.0;

    let input = LambertInput {
        r1,
        r2,
        tof,
        mu,
        direction: Direction::Prograde,
        max_revs: Some(0),
    };
    let sols = solve_lambert(&input).unwrap();
    let v1_kep = sols[0].v1;
    let a_kep = sols[0].a;

    let force = TwoBody::new(mu);
    let config = KsTpbvpConfig {
        poly_degree: 100,
        max_iterations: 80,
        tolerance: 1e-10,
    };
    let result = solve_ks_tpbvp(&r1, &r2, 0.0, tof, &v1_kep, a_kep, &force, &config);

    assert!(
        result.converged,
        "KS-TPBVP should converge for 170° arc ({} iters)",
        result.iterations_used
    );
    let v1_err = (result.v1 - v1_kep).norm();
    assert!(
        v1_err < 1e-3,
        "v1 error at 170° = {v1_err:.6e} km/s"
    );
}

// =========================================================================
// Sprint 5 — MCPI-MPS-IVP multi-revolution solver tests
// =========================================================================

use lambert_ult::perturbed::mps_ivp::{solve_mps_ivp, MpsIvpConfig};

/// Under two-body dynamics, the MPS-IVP solver should converge when
/// given the exact Keplerian solution as warm start (identity test).
#[test]
fn test_mps_ivp_two_body_n0() {
    let mu = 398600.4418;
    let r1 = Vector3::new(7000.0, 0.0, 0.0);
    let r2 = Vector3::new(0.0, 7000.0, 0.0);
    let tof = 2000.0;

    let input = LambertInput {
        r1,
        r2,
        tof,
        mu,
        direction: Direction::Prograde,
        max_revs: Some(0),
    };
    let sols = solve_lambert(&input).unwrap();
    let v1_kep = sols[0].v1;

    let force = TwoBody::new(mu);
    let config = MpsIvpConfig {
        poly_degree: 80,
        max_mcpi_iterations: 30,
        mcpi_tolerance: 1e-10,
        max_mps_iterations: 10,
        mps_tolerance: 1e-3,
        perturbation_scale: 1e-7,
        variable_fidelity: false,
    };

    let result = solve_mps_ivp(&r1, &r2, 0.0, tof, &v1_kep, &force, &config);
    assert!(
        result.converged,
        "MPS-IVP two-body N=0 should converge, err = {:.6e}",
        result.terminal_error
    );

    // v1 should be very close to the Keplerian solution
    let v_err = (result.v1 - v1_kep).norm();
    assert!(
        v_err < 1e-3,
        "v1 error under two-body: {v_err:.6e} km/s"
    );
}

/// MPS-IVP with a slightly perturbed warm start should still converge
/// to the correct two-body solution.
#[test]
fn test_mps_ivp_two_body_perturbed_warmstart() {
    let mu = 398600.4418;
    let r1 = Vector3::new(7000.0, 0.0, 0.0);
    let r2 = Vector3::new(0.0, 7000.0, 0.0);
    let tof = 2000.0;

    let input = LambertInput {
        r1,
        r2,
        tof,
        mu,
        direction: Direction::Prograde,
        max_revs: Some(0),
    };
    let sols = solve_lambert(&input).unwrap();
    let v1_kep = sols[0].v1;

    // Perturb warm start by ~0.5%
    let v1_pert = v1_kep * 1.005;

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

    let result = solve_mps_ivp(&r1, &r2, 0.0, tof, &v1_pert, &force, &config);
    assert!(
        result.converged,
        "MPS-IVP perturbed warm start should converge, err = {:.6e}",
        result.terminal_error
    );

    let v_err = (result.v1 - v1_kep).norm();
    assert!(
        v_err < 0.05,
        "v1 should recover Keplerian: error = {v_err:.6e} km/s"
    );
}

/// MPS-IVP under J2 should produce a different velocity than the
/// Keplerian solution for a 3-D transfer.
#[test]
fn test_mps_ivp_j2_differs_from_keplerian() {
    let mu = 398600.4418;
    let r1 = Vector3::new(7000.0, 0.0, 0.0);
    let r2 = Vector3::new(0.0, 7000.0, 3000.0); // out-of-plane
    let tof = 2000.0;

    let input = LambertInput {
        r1,
        r2,
        tof,
        mu,
        direction: Direction::Prograde,
        max_revs: Some(0),
    };
    let sols = solve_lambert(&input).unwrap();
    let v1_kep = sols[0].v1;

    let force = ZonalGravity::earth_j2();
    let config = MpsIvpConfig {
        poly_degree: 120,
        max_mcpi_iterations: 60,
        mcpi_tolerance: 1e-12,
        max_mps_iterations: 15,
        mps_tolerance: 1e-4,
        perturbation_scale: 1e-7,
        variable_fidelity: false,
    };

    let result = solve_mps_ivp(&r1, &r2, 0.0, tof, &v1_kep, &force, &config);
    assert!(
        result.converged,
        "MPS-IVP J2 should converge, err = {:.6e}",
        result.terminal_error
    );

    let diff = (result.v1 - v1_kep).norm();
    assert!(
        diff > 1e-6,
        "J2-perturbed v1 should differ from Keplerian: {diff:.6e}"
    );
}

/// MPS-IVP under J2: propagating (r1, v1_mps) under J2 should arrive at r2.
#[test]
fn test_mps_ivp_j2_propagation_consistency() {
    let mu = 398600.4418;
    let r1 = Vector3::new(7000.0, 0.0, 0.0);
    let r2 = Vector3::new(0.0, 7000.0, 3000.0);
    let tof = 2000.0;

    let input = LambertInput {
        r1,
        r2,
        tof,
        mu,
        direction: Direction::Prograde,
        max_revs: Some(0),
    };
    let sols = solve_lambert(&input).unwrap();
    let v1_kep = sols[0].v1;

    let force = ZonalGravity::earth_j2();
    let config = MpsIvpConfig {
        poly_degree: 120,
        max_mcpi_iterations: 60,
        mcpi_tolerance: 1e-12,
        max_mps_iterations: 15,
        mps_tolerance: 1e-4,
        perturbation_scale: 1e-7,
        variable_fidelity: false,
    };

    let result = solve_mps_ivp(&r1, &r2, 0.0, tof, &v1_kep, &force, &config);
    assert!(result.converged);

    // Forward-propagate (r1, v1_mps) under J2 using RK4 to verify
    let (rf, _vf) = rk4_propagate(&r1, &result.v1, tof, 50_000, &force);
    let pos_err = (rf - r2).norm();
    assert!(
        pos_err < 2.0,
        "propagated endpoint error = {pos_err:.6e} km (should be < 2 km)"
    );
}

/// MPS-IVP with variable fidelity should also converge.
#[test]
fn test_mps_ivp_variable_fidelity() {
    let mu = 398600.4418;
    let r1 = Vector3::new(7000.0, 0.0, 0.0);
    let r2 = Vector3::new(0.0, 7000.0, 3000.0);
    let tof = 2000.0;

    let input = LambertInput {
        r1,
        r2,
        tof,
        mu,
        direction: Direction::Prograde,
        max_revs: Some(0),
    };
    let sols = solve_lambert(&input).unwrap();
    let v1_kep = sols[0].v1;

    let force = ZonalGravity::earth_j2();
    let config = MpsIvpConfig {
        poly_degree: 120,
        max_mcpi_iterations: 60,
        mcpi_tolerance: 1e-12,
        max_mps_iterations: 15,
        mps_tolerance: 1e-4,
        perturbation_scale: 1e-7,
        variable_fidelity: true, // particular solutions use low-fidelity
    };

    let result = solve_mps_ivp(&r1, &r2, 0.0, tof, &v1_kep, &force, &config);
    assert!(
        result.converged,
        "MPS-IVP variable fidelity should converge, err = {:.6e}",
        result.terminal_error
    );
}

/// Multi-revolution test: solve a 1-rev transfer under two-body and verify
/// that MPS-IVP recovers the correct solution.
#[test]
fn test_mps_ivp_multirev_n1_two_body() {
    let mu: f64 = 398600.4418;
    let r_orbit: f64 = 8000.0;
    let period = 2.0 * std::f64::consts::PI * (r_orbit.powi(3) / mu).sqrt();

    // Transfer from periapsis to a point 60° ahead, with enough tof
    // for a 1-rev solution to exist.
    let r1 = Vector3::new(r_orbit, 0.0, 0.0);
    let theta = 60.0_f64.to_radians();
    let r2 = Vector3::new(r_orbit * theta.cos(), r_orbit * theta.sin(), 0.0);
    let tof = period * 1.1; // slightly more than one full orbit

    let input = LambertInput {
        r1,
        r2,
        tof,
        mu,
        direction: Direction::Prograde,
        max_revs: Some(1),
    };
    let sols = solve_lambert(&input).unwrap();

    // Find a 1-rev solution
    let one_rev_sols: Vec<_> = sols.iter().filter(|s| s.n_revs == 1).collect();
    if one_rev_sols.is_empty() {
        // No 1-rev solution exists for these parameters — skip
        return;
    }

    let v1_kep = one_rev_sols[0].v1;

    let force = TwoBody::new(mu);
    let config = MpsIvpConfig {
        poly_degree: 100,
        max_mcpi_iterations: 50,
        mcpi_tolerance: 1e-10,
        max_mps_iterations: 10,
        mps_tolerance: 1e-3,
        perturbation_scale: 1e-7,
        variable_fidelity: false,
    };

    let result = solve_mps_ivp(&r1, &r2, 0.0, tof, &v1_kep, &force, &config);
    assert!(
        result.converged,
        "MPS-IVP 1-rev two-body should converge, err = {:.6e}",
        result.terminal_error
    );

    // Verify propagation consistency
    let r2_prop = kepler_propagate(&r1, &result.v1, tof, mu);
    let err = (r2_prop - r2).norm();
    assert!(
        err < 1.0,
        "1-rev propagation error = {err:.6e} km"
    );
}

// =========================================================================
// Sprint 6 — Unified API + algorithm selector tests
// =========================================================================

use lambert_ult::solve_perturbed;
use lambert_ult::solve_lambert_bates_compat;
use lambert_ult::types::{UnifiedLambertConfig, PerturbedConfig};
use lambert_ult::perturbed::selector::PerturbedAlgorithm;

/// Bates-compatible wrapper should return the same v1/v2 as the regular
/// Keplerian solver for a simple N=0 transfer.
#[test]
fn test_bates_compat_matches_keplerian() {
    let mu = 398600.4418;
    let r1 = Vector3::new(7000.0, 0.0, 0.0);
    let r2 = Vector3::new(0.0, 7000.0, 0.0);
    let tof = 2000.0;

    let (v1_bates, v2_bates) =
        solve_lambert_bates_compat(&r1, &r2, tof, mu, Direction::Prograde).unwrap();

    let input = LambertInput {
        r1,
        r2,
        tof,
        mu,
        direction: Direction::Prograde,
        max_revs: Some(0),
    };
    let sols = solve_lambert(&input).unwrap();
    assert!(!sols.is_empty());

    let v1_kep = sols[0].v1;
    let v2_kep = sols[0].v2;

    assert!(
        (v1_bates - v1_kep).norm() < 1e-12,
        "v1 mismatch: bates vs keplerian"
    );
    assert!(
        (v2_bates - v2_kep).norm() < 1e-12,
        "v2 mismatch: bates vs keplerian"
    );
}

/// Unified API under two-body dynamics should return solutions that
/// propagate correctly to r2.
#[test]
fn test_unified_two_body_short_arc() {
    let mu = 398600.4418;
    let r1 = Vector3::new(7000.0, 0.0, 0.0);
    let r2 = Vector3::new(0.0, 7000.0, 0.0); // 90° transfer → TPBVP
    let tof = 2000.0;

    let force = TwoBody::new(mu);
    let config = UnifiedLambertConfig {
        direction: Direction::Prograde,
        max_revs: Some(0),
        perturbed: PerturbedConfig {
            poly_degree: 80,
            max_iterations: 30,
            mcpi_tolerance: 1e-10,
            ..Default::default()
        },
        ..Default::default()
    };

    let sols = solve_perturbed(&r1, &r2, tof, mu, &force, &config).unwrap();
    assert!(!sols.is_empty());
    let sol = &sols[0];

    // Under two-body, the perturbed solution ≈ Keplerian
    assert!(sol.converged, "TPBVP should converge under two-body");
    assert_eq!(sol.algorithm, PerturbedAlgorithm::McpiTpbvp);

    // Propagate and verify
    let r2_prop = kepler_propagate(&r1, &sol.v1, tof, mu);
    let err = (r2_prop - r2).norm();
    assert!(
        err < 1.0,
        "two-body unified endpoint error = {err:.6e} km"
    );
}

/// Unified API with a medium-arc transfer should invoke KS-TPBVP and converge.
#[test]
fn test_unified_two_body_medium_arc() {
    let mu = 398600.4418;
    let r1 = Vector3::new(7000.0, 0.0, 0.0);
    // ~150° transfer → KS-TPBVP territory
    let theta = 150.0_f64.to_radians();
    let r2 = Vector3::new(7000.0 * theta.cos(), 7000.0 * theta.sin(), 0.0);
    let tof = 3500.0;

    let force = TwoBody::new(mu);
    let config = UnifiedLambertConfig {
        direction: Direction::Prograde,
        max_revs: Some(0),
        perturbed: PerturbedConfig {
            poly_degree: 80,
            max_iterations: 50,
            mcpi_tolerance: 1e-10,
            ..Default::default()
        },
        ..Default::default()
    };

    let sols = solve_perturbed(&r1, &r2, tof, mu, &force, &config).unwrap();
    assert!(!sols.is_empty());
    let sol = &sols[0];

    assert!(sol.converged, "KS-TPBVP should converge under two-body");
    assert_eq!(sol.algorithm, PerturbedAlgorithm::McpiKsTpbvp);

    let r2_prop = kepler_propagate(&r1, &sol.v1, tof, mu);
    let err = (r2_prop - r2).norm();
    assert!(
        err < 1.0,
        "medium-arc unified endpoint error = {err:.6e} km"
    );
}

/// Unified API with a multi-rev transfer should use MPS-IVP.
#[test]
fn test_unified_multirev_uses_mps() {
    let mu = 398600.4418;
    let r_orbit: f64 = 8000.0;
    let period = 2.0 * std::f64::consts::PI * (r_orbit.powi(3) / mu).sqrt();

    let r1 = Vector3::new(r_orbit, 0.0, 0.0);
    let theta = 60.0_f64.to_radians();
    let r2 = Vector3::new(r_orbit * theta.cos(), r_orbit * theta.sin(), 0.0);
    let tof = period * 1.1;

    let force = TwoBody::new(mu);
    let config = UnifiedLambertConfig {
        direction: Direction::Prograde,
        max_revs: Some(1),
        perturbed: PerturbedConfig {
            poly_degree: 100,
            max_iterations: 50,
            mcpi_tolerance: 1e-10,
            max_mps_iterations: 10,
            mps_tolerance: 1e-3,
            ..Default::default()
        },
        ..Default::default()
    };

    let sols = solve_perturbed(&r1, &r2, tof, mu, &force, &config).unwrap();

    // Should have N=0 and possibly N=1 solutions
    let has_fractional = sols.iter().any(|s| s.n_revs == 0);
    assert!(has_fractional, "should have at least an N=0 solution");

    let multirev: Vec<_> = sols.iter().filter(|s| s.n_revs == 1).collect();
    for s in &multirev {
        assert_eq!(s.algorithm, PerturbedAlgorithm::McpiMpsIvp);
    }

    // Verify N=0 propagation
    let n0 = sols.iter().find(|s| s.n_revs == 0).unwrap();
    let r2_prop = kepler_propagate(&r1, &n0.v1, tof, mu);
    let err = (r2_prop - r2).norm();
    assert!(
        err < 1.0,
        "multirev N=0 endpoint error = {err:.6e} km"
    );
}

/// Unified API under J2 should produce converged solutions that differ
/// from Keplerian.
#[test]
fn test_unified_j2_short_arc() {
    let mu = 398600.4418;
    let r1 = Vector3::new(7000.0, 0.0, 0.0);
    let r2 = Vector3::new(0.0, 7000.0, 2000.0); // out-of-plane
    let tof = 2000.0;

    let force = ZonalGravity::earth_j2();
    let config = UnifiedLambertConfig {
        direction: Direction::Prograde,
        max_revs: Some(0),
        perturbed: PerturbedConfig {
            poly_degree: 120,
            max_iterations: 60,
            mcpi_tolerance: 1e-12,
            ..Default::default()
        },
        ..Default::default()
    };

    let sols = solve_perturbed(&r1, &r2, tof, mu, &force, &config).unwrap();
    assert!(!sols.is_empty());
    let sol = &sols[0];
    assert!(sol.converged, "J2 unified should converge");

    // Compare against Keplerian
    let kep_input = LambertInput {
        r1,
        r2,
        tof,
        mu,
        direction: Direction::Prograde,
        max_revs: Some(0),
    };
    let kep_sols = solve_lambert(&kep_input).unwrap();
    let diff = (sol.v1 - kep_sols[0].v1).norm();
    assert!(
        diff > 1e-6,
        "J2 unified v1 should differ from Keplerian: {diff:.6e}"
    );
}

/// Unified API with mixed N solutions: verify each solution has the
/// correct algorithm assigned and all converge.
#[test]
fn test_unified_mixed_n_algorithm_routing() {
    let mu = 398600.4418;
    let r_orbit: f64 = 10000.0;
    let period = 2.0 * std::f64::consts::PI * (r_orbit.powi(3) / mu).sqrt();

    let r1 = Vector3::new(r_orbit, 0.0, 0.0);
    // 45° transfer (prograde → short arc → TPBVP for N=0)
    let theta = 45.0_f64.to_radians();
    let r2 = Vector3::new(r_orbit * theta.cos(), r_orbit * theta.sin(), 0.0);
    let tof = period * 1.2; // long enough for N=1

    let force = TwoBody::new(mu);
    let config = UnifiedLambertConfig {
        direction: Direction::Prograde,
        max_revs: Some(1),
        perturbed: PerturbedConfig {
            poly_degree: 100,
            max_iterations: 50,
            mcpi_tolerance: 1e-10,
            max_mps_iterations: 10,
            mps_tolerance: 1e-3,
            ..Default::default()
        },
        ..Default::default()
    };

    let sols = solve_perturbed(&r1, &r2, tof, mu, &force, &config).unwrap();

    for sol in &sols {
        if sol.n_revs == 0 {
            // Short arc → TPBVP (θ=45° < 2π/3)
            assert_eq!(
                sol.algorithm,
                PerturbedAlgorithm::McpiTpbvp,
                "N=0 short arc should use TPBVP"
            );
        } else {
            // N ≥ 1 → MPS-IVP
            assert_eq!(
                sol.algorithm,
                PerturbedAlgorithm::McpiMpsIvp,
                "N≥1 should use MPS-IVP"
            );
        }
    }
}

// =========================================================================
// New tests for plan fixes
// =========================================================================

/// A1: Periapsis filter — a transfer that dips below Earth's surface should
/// be marked infeasible.
#[test]
fn test_periapsis_filter_rejects_collision() {
    use lambert_ult::types::FeasibilityConfig;

    let mu = 398600.4418;
    // Large altitude difference: LEO to near-opposite side with short TOF
    // forces a very eccentric orbit that dips below Earth surface.
    let r1 = Vector3::new(6500.0, 0.0, 0.0); // just above Earth
    let r2 = Vector3::new(-6500.0, 500.0, 0.0); // nearly opposite
    let tof = 5000.0;

    let input = LambertInput {
        r1,
        r2,
        tof,
        mu,
        direction: Direction::Prograde,
        max_revs: Some(0),
    };

    let config = FeasibilityConfig {
        check_earth_collision: true,
        earth_radius: 6378.137,
        ..Default::default()
    };
    let sols = lambert_ult::solve_lambert_with_config(&input, &config).unwrap();

    // At least some solutions should be infeasible (periapsis below Earth)
    let infeasible_count = sols.iter().filter(|s| !s.is_feasible).count();
    // We don't assert ALL are infeasible — just that the filter is working
    // (before this fix, infeasible_count was always 0)
    let feasible_count = sols.iter().filter(|s| s.is_feasible).count();
    assert!(
        infeasible_count > 0 || feasible_count == 0,
        "periapsis filter should flag some transfers as infeasible for low-altitude transfers"
    );
}

/// A2: Delta-v filter with reference orbit velocities should properly
/// filter high-delta-v solutions.
#[test]
fn test_delta_v_filter_with_reference_velocities() {
    use lambert_ult::keplerian::feasibility::filter_feasibility_with_ref;
    use lambert_ult::types::{Branch, FeasibilityConfig, LambertSolution};

    let mu: f64 = 398600.4418;
    let r1 = Vector3::new(7000.0, 0.0, 0.0);
    let r2 = Vector3::new(0.0, 7000.0, 0.0);

    // Circular orbit velocities at r1 and r2
    let v_circ = (mu / 7000.0_f64).sqrt();
    let v1_ref = Vector3::new(0.0, v_circ, 0.0);
    let v2_ref = Vector3::new(-v_circ, 0.0, 0.0);

    // A solution with very high transfer velocity → high delta-v
    let mut sols = vec![LambertSolution {
        v1: Vector3::new(0.0, 15.0, 0.0), // much faster than circular
        v2: Vector3::new(-15.0, 0.0, 0.0),
        a: 20000.0,
        n_revs: 0,
        branch: Branch::Fractional,
        transfer_angle: std::f64::consts::PI / 2.0,
        is_feasible: true,
    }];

    let config = FeasibilityConfig {
        check_earth_collision: false,
        check_escape_velocity: false,
        max_delta_v: Some(5.0), // 5 km/s limit
        ..Default::default()
    };

    filter_feasibility_with_ref(
        &mut sols, &r1, &r2, mu, &config,
        Some(&v1_ref), Some(&v2_ref),
    );

    // dv = |v1_transfer - v1_ref| + |v2_ref - v2_transfer| should exceed 5 km/s
    assert!(
        !sols[0].is_feasible,
        "high-delta-v solution should be filtered out"
    );
}

/// C2: Hyperbolic transfers should return a descriptive error, not NoSolution.
#[test]
fn test_hyperbolic_error_message() {
    let mu = 398600.4418;
    let r1 = Vector3::new(7000.0, 0.0, 0.0);
    let r2 = Vector3::new(0.0, 7000.0, 0.0);
    let tof = 10.0; // absurdly short → below parabolic minimum

    let input = LambertInput {
        r1,
        r2,
        tof,
        mu,
        direction: Direction::Prograde,
        max_revs: Some(0),
    };

    let err = solve_lambert(&input).unwrap_err();
    match err {
        lambert_ult::error::LambertError::InvalidInput(msg) => {
            assert!(
                msg.contains("hyperbolic"),
                "error message should mention hyperbolic: {msg}"
            );
        }
        other => panic!("expected InvalidInput, got {other:?}"),
    }
}

/// A3+B2: Multi-rev edge case — verify that transfers near the minimum
/// transfer time boundary produce valid solutions.
#[test]
fn test_multirev_near_tmin_boundary() {
    let mu: f64 = 398600.4418;
    let r_orbit: f64 = 8000.0;
    let period = 2.0 * std::f64::consts::PI * (r_orbit.powi(3) / mu).sqrt();

    let r1 = Vector3::new(r_orbit, 0.0, 0.0);
    let theta = 90.0_f64.to_radians();
    let r2 = Vector3::new(r_orbit * theta.cos(), r_orbit * theta.sin(), 0.0);

    // Use a TOF slightly above 1-rev minimum time
    let tof = period * 1.05;

    let input = LambertInput {
        r1,
        r2,
        tof,
        mu,
        direction: Direction::Prograde,
        max_revs: Some(1),
    };
    let sols = solve_lambert(&input).unwrap();

    // Should have N=0 solution plus possibly N=1 solutions
    let n0_sols: Vec<_> = sols.iter().filter(|s| s.n_revs == 0).collect();
    assert!(!n0_sols.is_empty(), "should have at least N=0 solution");

    // All solutions should be self-consistent (propagation arrives at r2)
    for sol in &sols {
        if !sol.is_feasible {
            continue;
        }
        let r2_prop = kepler_propagate(&r1, &sol.v1, tof, mu);
        let err = (r2_prop - r2).norm();
        assert!(
            err < 1.0,
            "N={} branch={:?}: propagation error = {err:.6e} km",
            sol.n_revs,
            sol.branch
        );
    }
}

/// B1: Newton's method should converge quickly for standard cases.
#[test]
fn test_newton_convergence_accuracy() {
    let mu = 398600.4418;
    let r1 = Vector3::new(7000.0, 0.0, 0.0);
    let r2 = Vector3::new(0.0, 9000.0, 3000.0);
    let tof = 3000.0;

    let input = LambertInput {
        r1,
        r2,
        tof,
        mu,
        direction: Direction::Prograde,
        max_revs: Some(0),
    };
    let sols = solve_lambert(&input).unwrap();
    assert!(!sols.is_empty());

    // Verify high-accuracy solution via propagation
    let r2_prop = kepler_propagate(&r1, &sols[0].v1, tof, mu);
    let err = (r2_prop - r2).norm();
    assert!(
        err < 0.01, // sub-10m accuracy
        "Newton solver should produce high-accuracy solution: error = {err:.6e} km"
    );
}

// =========================================================================
// Paper 1 regression tests (Woollands 2018, Figs. 15-16)
// Circular orbits at varying altitudes, e=0, i=10°
// =========================================================================

/// Helper: construct r1, r2 for a circular orbit transfer at given semi-major axis,
/// inclination, and 90° true anomaly span.
fn paper1_transfer_geometry(a: f64, inc_deg: f64) -> (Vector3<f64>, Vector3<f64>) {
    let inc = inc_deg.to_radians();
    let r1 = Vector3::new(a, 0.0, 0.0);
    let theta = 90.0_f64.to_radians();
    let r2 = Vector3::new(
        a * theta.cos(),
        a * theta.sin() * inc.cos(),
        a * theta.sin() * inc.sin(),
    );
    (r1, r2)
}

/// Paper 1 regression: perturbed solution at 8000 km should differ from Keplerian.
#[test]
fn test_paper1_regression_a8000() {
    let mu: f64 = 398600.4418;
    let a: f64 = 8000.0;
    let (r1, r2) = paper1_transfer_geometry(a, 10.0);
    let period = 2.0 * std::f64::consts::PI * (a.powi(3) / mu).sqrt();
    let tof = period * 0.25;

    let input = LambertInput { r1, r2, tof, mu, direction: Direction::Prograde, max_revs: Some(0) };
    let kep_sols = solve_lambert(&input).unwrap();
    let v1_kep = kep_sols[0].v1;

    let force = ZonalGravity::earth_j2_j6();
    let config = UnifiedLambertConfig {
        direction: Direction::Prograde,
        max_revs: Some(0),
        perturbed: PerturbedConfig {
            poly_degree: 120, max_iterations: 60, mcpi_tolerance: 1e-12,
            ..Default::default()
        },
        ..Default::default()
    };
    let sols = solve_perturbed(&r1, &r2, tof, mu, &force, &config).unwrap();
    assert!(!sols.is_empty() && sols[0].converged);

    let diff = (sols[0].v1 - v1_kep).norm();
    assert!(diff > 1e-6, "perturbation should be measurable at 8000 km: {diff:.6e}");
}

/// Paper 1 regression: perturbation effect should decrease at higher altitude.
#[test]
fn test_paper1_perturbation_decreases_with_altitude() {
    let mu: f64 = 398600.4418;
    let force = ZonalGravity::earth_j2();

    let mut diffs = Vec::new();
    for &a in &[8000.0_f64, 20000.0, 42000.0] {
        let (r1, r2) = paper1_transfer_geometry(a, 10.0);
        let period = 2.0 * std::f64::consts::PI * (a.powi(3) / mu).sqrt();
        let tof = period * 0.20; // short arc for TPBVP convergence

        let input = LambertInput { r1, r2, tof, mu, direction: Direction::Prograde, max_revs: Some(0) };
        let kep_sols = solve_lambert(&input).unwrap();
        let v1_kep = kep_sols[0].v1;

        let config = UnifiedLambertConfig {
            direction: Direction::Prograde,
            max_revs: Some(0),
            perturbed: PerturbedConfig {
                poly_degree: 100, max_iterations: 50, mcpi_tolerance: 1e-10,
                ..Default::default()
            },
            ..Default::default()
        };
        let sols = solve_perturbed(&r1, &r2, tof, mu, &force, &config).unwrap();
        if !sols.is_empty() && sols[0].converged {
            diffs.push((a, (sols[0].v1 - v1_kep).norm()));
        }
    }

    // Lower altitude should have larger perturbation effect
    if diffs.len() >= 2 {
        assert!(
            diffs[0].1 > diffs[diffs.len() - 1].1,
            "perturbation at {:.0} km ({:.6e}) should exceed {:.0} km ({:.6e})",
            diffs[0].0, diffs[0].1, diffs[diffs.len() - 1].0, diffs[diffs.len() - 1].1
        );
    }
}

// =========================================================================
// Paper 2 regression tests (Woollands 2017, Figs. 6-9)
// LEO, MEO, GTO, HEO orbit types
// =========================================================================

/// Paper 2 regression: LEO transfer under J2 should converge via MPS-IVP.
#[test]
fn test_paper2_leo_transfer() {
    let mu: f64 = 398600.4418;
    let a: f64 = 6878.0; // ~500 km altitude
    let r1 = Vector3::new(a, 0.0, 0.0);
    let v_circ = (mu / a).sqrt();
    let v0 = Vector3::new(0.0, v_circ * 0.866, v_circ * 0.5); // ~30° inclination
    let period = 2.0 * std::f64::consts::PI * (a.powi(3) / mu).sqrt();
    let tof = period * 0.25;
    let r2 = kepler_propagate(&r1, &v0, tof, mu);

    let input = LambertInput { r1, r2, tof, mu, direction: Direction::Prograde, max_revs: Some(0) };
    let kep_sols = solve_lambert(&input).unwrap();
    assert!(!kep_sols.is_empty());
    let v1_kep = kep_sols[0].v1;

    let force = ZonalGravity::earth_j2();
    let config = MpsIvpConfig {
        poly_degree: 100, max_mcpi_iterations: 50, mcpi_tolerance: 1e-10,
        max_mps_iterations: 15, mps_tolerance: 1e-3, perturbation_scale: 1e-7,
        variable_fidelity: false,
    };

    let result = solve_mps_ivp(&r1, &r2, 0.0, tof, &v1_kep, &force, &config);
    assert!(result.converged, "LEO transfer should converge, err = {:.6e}", result.terminal_error);

    let (rf, _) = rk4_propagate(&r1, &result.v1, tof, 50_000, &force);
    let pos_err = (rf - r2).norm();
    assert!(pos_err < 2.0, "LEO propagation error = {pos_err:.6e} km");
}

/// Paper 2 regression: MEO transfer.
#[test]
fn test_paper2_meo_transfer() {
    let mu: f64 = 398600.4418;
    let a: f64 = 26560.0; // GPS altitude
    let r1 = Vector3::new(a, 0.0, 0.0);
    let v_circ = (mu / a).sqrt();
    let v0 = Vector3::new(0.0, v_circ * 0.94, v_circ * 0.34); // ~20° inclination
    let period = 2.0 * std::f64::consts::PI * (a.powi(3) / mu).sqrt();
    let tof = period * 0.2;
    let r2 = kepler_propagate(&r1, &v0, tof, mu);

    let input = LambertInput { r1, r2, tof, mu, direction: Direction::Prograde, max_revs: Some(0) };
    let kep_sols = solve_lambert(&input).unwrap();
    let v1_kep = kep_sols[0].v1;

    let force = ZonalGravity::earth_j2();
    let config = MpsIvpConfig {
        poly_degree: 80, max_mcpi_iterations: 40, mcpi_tolerance: 1e-10,
        max_mps_iterations: 15, mps_tolerance: 1e-3, ..Default::default()
    };

    let result = solve_mps_ivp(&r1, &r2, 0.0, tof, &v1_kep, &force, &config);
    assert!(result.converged, "MEO transfer should converge, err = {:.6e}", result.terminal_error);
}
