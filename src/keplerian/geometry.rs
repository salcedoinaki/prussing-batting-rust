use nalgebra::Vector3;
use std::f64::consts::PI;

use crate::types::{Direction, TransferGeometry};

/// Compute all transfer-geometry quantities from the two position vectors and
/// the desired transfer direction.
pub fn compute_transfer_geometry(
    r1: &Vector3<f64>,
    r2: &Vector3<f64>,
    direction: Direction,
) -> TransferGeometry {
    let r1_mag = r1.norm();
    let r2_mag = r2.norm();

    let chord_vec = r2 - r1;
    let c = chord_vec.norm();

    let s = (r1_mag + r2_mag + c) / 2.0;

    let cos_theta = r1.dot(r2) / (r1_mag * r2_mag);
    let cross = r1.cross(r2);
    let sin_theta_unsigned = cross.norm() / (r1_mag * r2_mag);

    // Sign convention: prograde means the transfer angle is < pi when the
    // orbit normal (cross product) has a positive z-component, and > pi
    // otherwise.  Retrograde flips.
    let z_sign = if cross.z >= 0.0 { 1.0 } else { -1.0 };
    let dir_sign = match direction {
        Direction::Prograde => 1.0,
        Direction::Retrograde => -1.0,
    };
    let sin_theta = z_sign * dir_sign * sin_theta_unsigned;
    let mut theta = f64::atan2(sin_theta, cos_theta);
    if theta < 0.0 {
        theta += 2.0 * PI;
    }

    let a_min = s / 2.0;

    TransferGeometry {
        r1_mag,
        r2_mag,
        c,
        s,
        theta,
        a_min,
    }
}

/// Compute the principal auxiliary angles alpha_0 and beta_0 for a given
/// semi-major axis `a`.
///
/// Returns `(alpha_0, beta_0)` both in `[0, pi]`.
pub fn auxiliary_angles_principal(a: f64, s: f64, c: f64) -> (f64, f64) {
    // sin(alpha/2) = sqrt(s / (2a))
    let sin_alpha_half = (s / (2.0 * a)).sqrt().clamp(-1.0, 1.0);
    let alpha_0 = 2.0 * sin_alpha_half.asin();

    // sin(beta/2) = sqrt((s - c) / (2a))
    let sin_beta_half = ((s - c) / (2.0 * a)).sqrt().clamp(-1.0, 1.0);
    let beta_0 = 2.0 * sin_beta_half.asin();

    (alpha_0, beta_0)
}

/// Compute the correct (alpha, beta) pair for a given semi-major axis,
/// revolution count, transfer angle, and whether we are on the upper or lower
/// time branch.
///
/// - `theta`: transfer angle (0, 2*pi)
/// - `upper`: true if the desired time exceeds the minimum-energy time for
///   this N (upper branch / long-period solution).
pub fn auxiliary_angles(
    a: f64,
    s: f64,
    c: f64,
    theta: f64,
    upper: bool,
) -> (f64, f64) {
    let (alpha_0, beta_0) = auxiliary_angles_principal(a, s, c);

    // beta sign: short-way (theta <= pi) -> +beta_0, long-way -> -beta_0
    let beta = if theta <= PI { beta_0 } else { -beta_0 };

    // alpha branch: lower time branch -> alpha_0, upper -> 2*pi - alpha_0
    let alpha = if upper { 2.0 * PI - alpha_0 } else { alpha_0 };

    (alpha, beta)
}

/// Minimum-energy transfer time for `n_revs` complete revolutions.
///
/// At the minimum-energy semi-major axis `a_m = s/2` the auxiliary angle
/// `alpha_m = pi`.
pub fn time_min_energy(s: f64, c: f64, theta: f64, n_revs: u32, mu: f64) -> f64 {
    let sin_beta_m_half = ((s - c) / s).sqrt().clamp(-1.0, 1.0);
    let beta_m0 = 2.0 * sin_beta_m_half.asin();
    let beta_m = if theta <= PI { beta_m0 } else { -beta_m0 };

    let alpha_m = PI; // at minimum energy

    let k = (2 * n_revs) as f64 * PI + alpha_m - beta_m + beta_m.sin();
    (s / 2.0).powf(1.5) * k / mu.sqrt()
}

/// Parabolic transfer time — lower bound for elliptic solutions (N = 0).
pub fn time_parabolic(s: f64, c: f64, theta: f64, mu: f64) -> f64 {
    let sgn = if theta <= PI { 1.0 } else { -1.0 };
    (2.0_f64).sqrt() / 3.0 * (s.powf(1.5) - sgn * (s - c).powf(1.5)) / mu.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_geometry_coplanar() {
        let r1 = Vector3::new(7000.0, 0.0, 0.0);
        let r2 = Vector3::new(0.0, 7000.0, 0.0);
        let geom = compute_transfer_geometry(&r1, &r2, Direction::Prograde);

        assert_relative_eq!(geom.r1_mag, 7000.0, epsilon = 1e-10);
        assert_relative_eq!(geom.r2_mag, 7000.0, epsilon = 1e-10);
        assert_relative_eq!(geom.theta, PI / 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_parabolic_time_positive() {
        let s = 14000.0;
        let c = 9899.0; // rough chord for 90-deg transfer at r=7000
        let tp = time_parabolic(s, c, PI / 2.0, 398600.4418);
        assert!(tp > 0.0);
    }
}
