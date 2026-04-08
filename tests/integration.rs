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
