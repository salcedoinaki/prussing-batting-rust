use nalgebra::Vector3;

use lambert_ult::force_models::ForceModel;

/// Propagate a two-body orbit from (r, v) for time `dt` using a universal-variable
/// Kepler propagator. Returns the final position.
pub fn kepler_propagate(r0: &Vector3<f64>, v0: &Vector3<f64>, dt: f64, mu: f64) -> Vector3<f64> {
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
    let alpha = 2.0 / r0_mag - v0_mag_sq / mu;

    let mut chi = mu.sqrt() * dt * alpha.abs();
    if chi == 0.0 {
        chi = 0.1;
    }

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

/// Simple RK4 propagator for test reference.
pub fn rk4_propagate(
    r0: &Vector3<f64>,
    v0: &Vector3<f64>,
    dt: f64,
    n_steps: usize,
    force: &dyn ForceModel,
) -> (Vector3<f64>, Vector3<f64>) {
    let h = dt / n_steps as f64;
    let mut r = *r0;
    let mut v = *v0;
    let mut t = 0.0;

    for _ in 0..n_steps {
        let a1 = force.acceleration(t, &r, &v);
        let r2 = r + 0.5 * h * v;
        let v2 = v + 0.5 * h * a1;
        let a2 = force.acceleration(t + 0.5 * h, &r2, &v2);
        let r3 = r + 0.5 * h * v2;
        let v3 = v + 0.5 * h * a2;
        let a3 = force.acceleration(t + 0.5 * h, &r3, &v3);
        let r4 = r + h * v3;
        let v4 = v + h * a3;
        let a4 = force.acceleration(t + h, &r4, &v4);

        r += h / 6.0 * (v + 2.0 * v2 + 2.0 * v3 + v4);
        v += h / 6.0 * (a1 + 2.0 * a2 + 2.0 * a3 + a4);
        t += h;
    }

    (r, v)
}
