use approx::assert_relative_eq;
use nalgebra::Vector3;

use lambert_ult::solve_lambert;
use lambert_ult::types::{Direction, LambertInput};

/// Helper: propagate a two-body orbit from (r, v) for time `dt` using a
/// simple universal-variable Kepler propagator, and return the final position.
/// This is used to verify that the Lambert solution is self-consistent:
/// propagating from (r1, v1) for `tof` should arrive at r2.
fn kepler_propagate(r0: &Vector3<f64>, v0: &Vector3<f64>, dt: f64, mu: f64) -> Vector3<f64> {
    // --- Stumpff functions ---
    fn stumpff_c(z: f64) -> f64 {
        if z.abs() < 1e-10 {
            1.0 / 2.0 - z / 24.0
        } else if z > 0.0 {
            (1.0 - z.sqrt().cos()) / z
        } else {
            ((-z).sqrt().cosh() - 1.0) / (-z)
        }
    }
    fn stumpff_s(z: f64) -> f64 {
        if z.abs() < 1e-10 {
            1.0 / 6.0 - z / 120.0
        } else if z > 0.0 {
            (z.sqrt() - z.sqrt().sin()) / z.powf(1.5)
        } else {
            ((-z).sqrt().sinh() - (-z).sqrt()) / (-z).powf(1.5)
        }
    }

    let r0_mag = r0.norm();
    let v0_mag_sq = v0.dot(v0);
    let alpha = 2.0 / r0_mag - v0_mag_sq / mu; // 1/a

    // Initial guess for the universal anomaly chi
    let mut chi = mu.sqrt() * dt * alpha.abs();
    if chi == 0.0 {
        chi = 0.1;
    }

    // Newton iteration
    for _ in 0..100 {
        let psi = chi * chi * alpha;
        let c2 = stumpff_c(psi);
        let c3 = stumpff_s(psi);
        let r0_dot_v0 = r0.dot(v0);

        let r = chi * chi * c2
            + r0_dot_v0 / mu.sqrt() * chi * (1.0 - psi * c3)
            + r0_mag * (1.0 - psi * c2);

        let f_val = r0_mag * chi * (1.0 - psi * c3)
            + r0_dot_v0 / mu.sqrt() * chi * chi * c2
            + chi * chi * chi * c3
            - mu.sqrt() * dt;

        let f_prime = r;
        let delta = f_val / f_prime;
        chi -= delta;
        if delta.abs() < 1e-12 {
            break;
        }
    }

    let psi = chi * chi * alpha;
    let c2 = stumpff_c(psi);
    let c3 = stumpff_s(psi);

    let f = 1.0 - chi * chi / r0_mag * c2;
    let g = dt - chi * chi * chi / mu.sqrt() * c3;

    f * r0 + g * v0
}

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
