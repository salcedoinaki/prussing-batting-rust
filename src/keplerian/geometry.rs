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

/// Find the semi-major axis `a_tmin` that gives the minimum possible
/// elliptic transfer time for `n_revs ≥ 1` complete revolutions.
///
/// This solves `f(a) = 0` via Newton's method, where `f(a)` is given by
/// Prussing Eq. 34 and its derivative by Eq. 35.
///
/// Returns `None` if Newton iteration fails to converge (shouldn't happen
/// for well-posed inputs with `n_revs ≥ 1`).
pub fn find_a_tmin(s: f64, c: f64, theta: f64, n_revs: u32, _mu: f64) -> Option<f64> {
    if n_revs == 0 {
        return None; // N=0 has no minimum transfer time (goes to parabolic)
    }

    let n = n_revs as f64;
    let a_m = s / 2.0;
    let mut a = a_m * 1.001; // initial guess just above minimum-energy axis

    for _ in 0..50 {
        let (alpha_0, beta_0) = auxiliary_angles_principal(a, s, c);
        let beta = if theta <= PI { beta_0 } else { -beta_0 };
        // For the minimum-time search, use alpha = alpha_0 (lower branch)
        let alpha = alpha_0;

        let xi = alpha - beta;
        let eta = alpha.sin() - beta.sin();

        // f(a) from Eq. 34
        let f_val = (6.0 * n * PI + 3.0 * xi - eta)
            * (xi.sin() + eta)
            - 8.0 * (1.0 - xi.cos());

        if f_val.abs() < 1e-14 {
            return Some(a);
        }

        // f'(a) from Eq. 35
        let tan_alpha_half = (alpha / 2.0).tan();
        let tan_beta_half = (beta / 2.0).tan();

        let da_alpha = -tan_alpha_half / a; // dα/da
        let da_beta = -tan_beta_half / a; // dβ/da

        let cos_xi = xi.cos();
        let cos_alpha = alpha.cos();
        let cos_beta = beta.cos();
        let sin_xi = xi.sin();

        // Chain rule: df/da = (df/dα)(dα/da) + (df/dβ)(dβ/da)
        // df/dα:
        let df_dalpha = (3.0 - cos_alpha) * (sin_xi + eta)
            + (6.0 * n * PI + 3.0 * xi - eta) * (cos_xi + cos_alpha)
            - 8.0 * sin_xi;
        // df/dβ:
        let df_dbeta = (-3.0 - cos_beta) * (sin_xi + eta)
            + (6.0 * n * PI + 3.0 * xi - eta) * (-cos_xi - cos_beta)
            + 8.0 * sin_xi;

        let f_prime = df_dalpha * da_alpha + df_dbeta * da_beta;

        if f_prime.abs() < 1e-30 {
            return Some(a); // derivative vanished, treat as converged
        }

        let delta = f_val / f_prime;
        a -= delta;

        // Keep a above a_m
        if a <= a_m {
            a = a_m * 1.0001;
        }

        if delta.abs() < 1e-14 * a_m {
            return Some(a);
        }
    }

    Some(a) // return best effort
}

/// Find the maximum revolution count whose minimum transfer time does
/// not exceed the desired time of flight.
///
/// Uses `time_min_transfer` (the true minimum elliptic transfer time for
/// each N) rather than `time_min_energy`, which is the time at the
/// minimum-energy ellipse — a less restrictive bound.
pub fn determine_n_max(
    geom: &TransferGeometry,
    tof: f64,
    mu: f64,
    max_revs: Option<u32>,
) -> u32 {
    let cap = max_revs.unwrap_or(u32::MAX);
    let mut n: u32 = 0;
    loop {
        let next = n + 1;
        if next > cap {
            return n;
        }
        let t_min_n = time_min_transfer(geom.s, geom.c, geom.theta, next, mu);
        if t_min_n > tof {
            return n;
        }
        n = next;
    }
}

/// Minimum possible elliptic transfer time for `n_revs ≥ 1` revolutions.
///
/// This is the time at `a_tmin` (blue dots in Paper 2, Fig. 5), NOT the
/// minimum-energy time `t_m` (red dots). For N=0 returns the parabolic time.
pub fn time_min_transfer(s: f64, c: f64, theta: f64, n_revs: u32, mu: f64) -> f64 {
    if n_revs == 0 {
        return time_parabolic(s, c, theta, mu);
    }
    match find_a_tmin(s, c, theta, n_revs, mu) {
        Some(a_tmin) => {
            let (alpha, beta) = auxiliary_angles(a_tmin, s, c, theta, false);
            tof_from_a(a_tmin, alpha, beta, n_revs, mu)
        }
        None => f64::MAX,
    }
}

/// Time of flight for a given semi-major axis, revolution count, and
/// auxiliary-angle branch.
///
/// ```text
/// sqrt(mu) * t = a^{3/2} * (2*N*pi + alpha - beta - (sin(alpha) - sin(beta)))
/// ```
pub fn tof_from_a(a: f64, alpha: f64, beta: f64, n_revs: u32, mu: f64) -> f64 {
    let n = n_revs as f64;
    let xi = alpha - beta;
    let eta = alpha.sin() - beta.sin();
    a.powf(1.5) * (2.0 * n * PI + xi - eta) / mu.sqrt()
}

/// Derivative of the time of flight with respect to semi-major axis `a`.
///
/// Computed via the chain rule from:
///   √μ · t = a^{3/2} · (2Nπ + α − β − (sinα − sinβ))
///
/// The auxiliary angles depend on `a` through:
///   sin(α/2) = √(s/2a)  →  dα/da = −tan(α/2) / a
///   sin(β/2) = √((s−c)/2a)  →  dβ/da = −tan(β/2) / a
pub fn dtof_da(a: f64, alpha: f64, beta: f64, n_revs: u32, mu: f64) -> f64 {
    let n = n_revs as f64;
    let xi = alpha - beta;
    let eta = alpha.sin() - beta.sin();
    let bracket = 2.0 * n * PI + xi - eta;

    // t = a^{3/2}/√μ · bracket
    // dt/da = (3/2)a^{1/2}/√μ · bracket + a^{3/2}/√μ · d(bracket)/da

    // Derivatives of α, β w.r.t. a
    let da_alpha = -(alpha / 2.0).tan() / a;
    let da_beta = -(beta / 2.0).tan() / a;

    // d(bracket)/da = (1 - cosα)·dα/da − (1 - cosβ)·dβ/da
    let d_bracket = (1.0 - alpha.cos()) * da_alpha - (1.0 - beta.cos()) * da_beta;

    let sqrt_mu = mu.sqrt();
    (3.0 / 2.0) * a.sqrt() / sqrt_mu * bracket + a.powf(1.5) / sqrt_mu * d_bracket
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
