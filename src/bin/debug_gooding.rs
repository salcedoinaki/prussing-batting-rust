//! Scratch debugging binary comparing Prussing vs Gooding on tricky
//! geometries. Run with `cargo run --bin debug_gooding`.

use lambert_ult::force_models::gravity::ZonalGravity;
use lambert_ult::keplerian::geometry::{auxiliary_angles, compute_transfer_geometry, tof_from_a};
use lambert_ult::types::{
    Direction, LambertInput, PerturbedConfig, UnifiedLambertConfig,
};
use lambert_ult::{
    solve_lambert, solve_lambert_gooding, solve_perturbed, solve_perturbed_gooding,
};
use nalgebra::Vector3;

fn main() {
    // J2 short-arc geometry from tests/integration.rs::test_unified_j2_short_arc.
    let mu = 398600.4418;
    let r1 = Vector3::new(7000.0, 0.0, 0.0);
    let r2 = Vector3::new(0.0, 7000.0, 2000.0);
    let tof = 2000.0;

    // Step 1: Keplerian comparison.
    let input = LambertInput {
        r1,
        r2,
        tof,
        mu,
        direction: Direction::Prograde,
        max_revs: Some(0),
    };
    let pr = solve_lambert(&input).unwrap();
    let gd = solve_lambert_gooding(&input).unwrap();
    let geom = compute_transfer_geometry(&r1, &r2, Direction::Prograde);
    println!("== Keplerian ==");
    for (p, g) in pr.iter().zip(gd.iter()) {
        // residuals for each
        let upper_p = matches!(p.branch, lambert_ult::types::Branch::Upper);
        let upper_g = matches!(g.branch, lambert_ult::types::Branch::Upper);
        let (ap, bp) = auxiliary_angles(p.a, geom.s, geom.c, geom.theta, upper_p);
        let (ag, bg) = auxiliary_angles(g.a, geom.s, geom.c, geom.theta, upper_g);
        let res_p = tof_from_a(p.a, ap, bp, p.n_revs, mu) - tof;
        let res_g = tof_from_a(g.a, ag, bg, g.n_revs, mu) - tof;
        println!(
            "  Prussing: a={:.12} res_tof={:.3e} s",
            p.a, res_p
        );
        println!(
            "  Gooding:  a={:.12} res_tof={:.3e} s",
            g.a, res_g
        );
        println!(
            "  diff a={:.3e} v1={:.3e}",
            (p.a - g.a).abs(),
            (p.v1 - g.v1).norm()
        );
    }

    // Step 2: Perturbed comparison under J2.
    let force = ZonalGravity::earth_j2();
    let config = UnifiedLambertConfig {
        direction: Direction::Prograde,
        max_revs: Some(0),
        perturbed: PerturbedConfig {
            poly_degree: 120,
            max_iterations: 60,
            mcpi_tolerance: 1.0e-12,
            ..Default::default()
        },
        ..Default::default()
    };

    let pr_pert = solve_perturbed(&r1, &r2, tof, mu, &force, &config).unwrap();
    let gd_pert = solve_perturbed_gooding(&r1, &r2, tof, mu, &force, &config).unwrap();
    println!("== Perturbed (J2), tight tol (mcpi=1e-12, maxiter=60) ==");
    for (p, g) in pr_pert.iter().zip(gd_pert.iter()) {
        println!("  Prussing: converged={}, v1={:?}", p.converged, p.v1);
        println!("  Gooding:  converged={}, v1={:?}", g.converged, g.v1);
        println!("  diff v1={:.3e}", (p.v1 - g.v1).norm());
    }

    // Looser: 1e-10 tolerance.
    let config2 = UnifiedLambertConfig {
        direction: Direction::Prograde,
        max_revs: Some(0),
        perturbed: PerturbedConfig {
            poly_degree: 120,
            max_iterations: 60,
            mcpi_tolerance: 1.0e-10,
            ..Default::default()
        },
        ..Default::default()
    };
    let pr_pert2 = solve_perturbed(&r1, &r2, tof, mu, &force, &config2).unwrap();
    let gd_pert2 = solve_perturbed_gooding(&r1, &r2, tof, mu, &force, &config2).unwrap();
    println!("== Perturbed (J2), looser tol (mcpi=1e-10, maxiter=60) ==");
    for (p, g) in pr_pert2.iter().zip(gd_pert2.iter()) {
        println!("  Prussing: converged={}, v1={:?}", p.converged, p.v1);
        println!("  Gooding:  converged={}, v1={:?}", g.converged, g.v1);
        println!("  diff v1={:.3e}", (p.v1 - g.v1).norm());
    }

    // More iterations.
    let config3 = UnifiedLambertConfig {
        direction: Direction::Prograde,
        max_revs: Some(0),
        perturbed: PerturbedConfig {
            poly_degree: 120,
            max_iterations: 100,
            mcpi_tolerance: 1.0e-12,
            ..Default::default()
        },
        ..Default::default()
    };
    let pr_pert3 = solve_perturbed(&r1, &r2, tof, mu, &force, &config3).unwrap();
    let gd_pert3 = solve_perturbed_gooding(&r1, &r2, tof, mu, &force, &config3).unwrap();
    println!("== Perturbed (J2), tight tol + more iters (mcpi=1e-12, maxiter=100) ==");
    for (p, g) in pr_pert3.iter().zip(gd_pert3.iter()) {
        println!("  Prussing: converged={}, v1={:?}", p.converged, p.v1);
        println!("  Gooding:  converged={}, v1={:?}", g.converged, g.v1);
        println!("  diff v1={:.3e}", (p.v1 - g.v1).norm());
    }
}
