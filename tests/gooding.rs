//! Integration tests for the Gooding Lambert solver (new parallel path to
//! Prussing). The core contract is that `solve_gooding` returns the same
//! `Vec<LambertSolution>` as `solve_prussing` for every well-posed input,
//! and that `solve_perturbed_gooding` plugs into the unified refinement
//! pipeline identically to `solve_perturbed`.

mod common;
use common::kepler_propagate;

use approx::assert_relative_eq;
use nalgebra::Vector3;

use lambert_ult::force_models::two_body::TwoBody;
use lambert_ult::force_models::gravity::ZonalGravity;
use lambert_ult::perturbed::selector::PerturbedAlgorithm;
use lambert_ult::types::{
    Branch, Direction, LambertInput, PerturbedConfig, UnifiedLambertConfig,
};
use lambert_ult::{
    solve_lambert, solve_lambert_gooding, solve_perturbed, solve_perturbed_gooding,
};

fn sort_by_branch_and_a(mut v: Vec<lambert_ult::types::LambertSolution>) -> Vec<lambert_ult::types::LambertSolution> {
    v.sort_by(|a, b| {
        let key_a = (a.n_revs, branch_rank(a.branch), a.a);
        let key_b = (b.n_revs, branch_rank(b.branch), b.a);
        key_a.partial_cmp(&key_b).unwrap()
    });
    v
}

fn branch_rank(b: Branch) -> u8 {
    match b {
        Branch::Fractional => 0,
        Branch::Lower => 1,
        Branch::Upper => 2,
    }
}

// -------------------------------------------------------------------------
// Gooding vs Prussing cross-check
// -------------------------------------------------------------------------

#[test]
fn test_gooding_matches_prussing_n0_90deg() {
    let input = LambertInput {
        r1: Vector3::new(7000.0, 0.0, 0.0),
        r2: Vector3::new(0.0, 7000.0, 0.0),
        tof: 2000.0,
        mu: 398600.4418,
        direction: Direction::Prograde,
        max_revs: Some(0),
    };
    let pr = sort_by_branch_and_a(solve_lambert(&input).unwrap());
    let gd = sort_by_branch_and_a(solve_lambert_gooding(&input).unwrap());
    assert_eq!(pr.len(), gd.len());
    for (p, g) in pr.iter().zip(gd.iter()) {
        assert_eq!(p.n_revs, g.n_revs);
        assert_eq!(p.branch, g.branch);
        assert_relative_eq!(p.a, g.a, epsilon = 1.0e-6);
        assert!((p.v1 - g.v1).norm() < 1.0e-8);
        assert!((p.v2 - g.v2).norm() < 1.0e-8);
    }
}

#[test]
fn test_gooding_matches_prussing_3d() {
    let input = LambertInput {
        r1: Vector3::new(5000.0, 10000.0, 2100.0),
        r2: Vector3::new(-14600.0, 2500.0, 7000.0),
        tof: 3600.0,
        mu: 398600.4418,
        direction: Direction::Prograde,
        max_revs: Some(0),
    };
    let pr = sort_by_branch_and_a(solve_lambert(&input).unwrap());
    let gd = sort_by_branch_and_a(solve_lambert_gooding(&input).unwrap());
    assert_eq!(pr.len(), gd.len());
    for (p, g) in pr.iter().zip(gd.iter()) {
        assert_relative_eq!(p.a, g.a, epsilon = 1.0e-4);
        assert!((p.v1 - g.v1).norm() < 1.0e-7);
        assert!((p.v2 - g.v2).norm() < 1.0e-7);
    }
}

#[test]
fn test_gooding_matches_prussing_retrograde() {
    let input = LambertInput {
        r1: Vector3::new(7000.0, 0.0, 0.0),
        r2: Vector3::new(0.0, 7000.0, 0.0),
        tof: 3600.0,
        mu: 398600.4418,
        direction: Direction::Retrograde,
        max_revs: Some(0),
    };
    let pr = sort_by_branch_and_a(solve_lambert(&input).unwrap());
    let gd = sort_by_branch_and_a(solve_lambert_gooding(&input).unwrap());
    assert_eq!(pr.len(), gd.len());
    for (p, g) in pr.iter().zip(gd.iter()) {
        assert_relative_eq!(p.a, g.a, epsilon = 1.0e-4);
        assert!((p.v1 - g.v1).norm() < 1.0e-7);
    }
}

#[test]
fn test_gooding_matches_prussing_multirev_standard() {
    // Circular geometry, transfer 60° ahead, tof slightly over one period.
    let mu = 398600.4418;
    let r_orbit: f64 = 8000.0;
    let period = 2.0 * std::f64::consts::PI * (r_orbit.powi(3) / mu).sqrt();
    let theta_deg: f64 = 60.0;
    let theta = theta_deg.to_radians();
    let input = LambertInput {
        r1: Vector3::new(r_orbit, 0.0, 0.0),
        r2: Vector3::new(r_orbit * theta.cos(), r_orbit * theta.sin(), 0.0),
        tof: period * 1.1,
        mu,
        direction: Direction::Prograde,
        max_revs: Some(2),
    };
    let pr = sort_by_branch_and_a(solve_lambert(&input).unwrap());
    let gd = sort_by_branch_and_a(solve_lambert_gooding(&input).unwrap());
    assert!(!pr.is_empty());
    assert_eq!(pr.len(), gd.len(), "multirev solution count mismatch");
    for (p, g) in pr.iter().zip(gd.iter()) {
        assert_eq!(p.n_revs, g.n_revs, "n_revs mismatch");
        assert_eq!(p.branch, g.branch, "branch mismatch");
        assert_relative_eq!(p.a, g.a, epsilon = 1.0e-4);
        assert!(
            (p.v1 - g.v1).norm() < 1.0e-6,
            "multirev v1 mismatch for N={} {:?}: {:.3e}",
            p.n_revs,
            p.branch,
            (p.v1 - g.v1).norm()
        );
    }
}

#[test]
fn test_gooding_roundtrip_propagation() {
    // Every Gooding solution must propagate under two-body from r1 for tof
    // and arrive back at r2 within sub-km accuracy.
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
    let sols = solve_lambert_gooding(&input).expect("solver should succeed");
    assert!(!sols.is_empty());
    for sol in &sols {
        let r2_prop = kepler_propagate(&r1, &sol.v1, tof, mu);
        let err = (r2_prop - r2).norm();
        assert!(
            err < 1.0e-3,
            "Gooding N={} branch {:?}: position error = {err:.4} km",
            sol.n_revs,
            sol.branch
        );
    }
}

// -------------------------------------------------------------------------
// Unified perturbed pipeline via Gooding
// -------------------------------------------------------------------------

#[test]
fn test_solve_perturbed_gooding_matches_solve_perturbed_two_body() {
    // Under TwoBody dynamics, Gooding-warmstarted and Prussing-warmstarted
    // perturbed solvers should converge to the same (v1, v2).
    let mu = 398600.4418;
    let r1 = Vector3::new(7000.0, 0.0, 0.0);
    let r2 = Vector3::new(0.0, 7000.0, 0.0);
    let tof = 2000.0;

    let force = TwoBody::new(mu);
    let config = UnifiedLambertConfig {
        direction: Direction::Prograde,
        max_revs: Some(0),
        perturbed: PerturbedConfig {
            poly_degree: 80,
            max_iterations: 30,
            mcpi_tolerance: 1.0e-10,
            ..Default::default()
        },
        ..Default::default()
    };

    let pr = solve_perturbed(&r1, &r2, tof, mu, &force, &config).unwrap();
    let gd = solve_perturbed_gooding(&r1, &r2, tof, mu, &force, &config).unwrap();
    assert_eq!(pr.len(), gd.len());
    for (p, g) in pr.iter().zip(gd.iter()) {
        assert_eq!(p.n_revs, g.n_revs);
        assert_eq!(p.branch, g.branch);
        assert_eq!(p.algorithm, g.algorithm);
        assert!(
            (p.v1 - g.v1).norm() < 1.0e-6,
            "perturbed v1 mismatch: {:.6e}",
            (p.v1 - g.v1).norm()
        );
    }
}

#[test]
fn test_solve_perturbed_gooding_j2_short_arc_converges() {
    // Mirror of `test_unified_j2_short_arc` but via the Gooding path.
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
            max_iterations: 100,
            mcpi_tolerance: 1.0e-10,
            ..Default::default()
        },
        ..Default::default()
    };

    let sols = solve_perturbed_gooding(&r1, &r2, tof, mu, &force, &config).unwrap();
    assert!(!sols.is_empty());
    let sol = &sols[0];
    assert!(sol.converged, "Gooding+J2 should converge");
    assert_eq!(sol.algorithm, PerturbedAlgorithm::McpiTpbvp);

    // Differs meaningfully from the Keplerian warm start.
    let kep_input = LambertInput {
        r1,
        r2,
        tof,
        mu,
        direction: Direction::Prograde,
        max_revs: Some(0),
    };
    let kep_sols = solve_lambert_gooding(&kep_input).unwrap();
    let diff = (sol.v1 - kep_sols[0].v1).norm();
    assert!(
        diff > 1.0e-6,
        "J2 perturbed v1 should differ from Keplerian: {diff:.6e}"
    );
}
