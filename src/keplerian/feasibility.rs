use nalgebra::Vector3;

use crate::types::{FeasibilityConfig, LambertSolution};

/// Apply feasibility filters to a set of Lambert solutions **in place**,
/// setting `is_feasible = false` on solutions that fail any check.
///
/// The `v1_ref` and `v2_ref` parameters are the velocities on the departure
/// and arrival orbits, respectively. They are required for the `max_delta_v`
/// check, which computes `Δv = ‖v1_transfer − v1_ref‖ + ‖v2_ref − v2_transfer‖`.
/// Pass `None` to skip the delta-v check even if `max_delta_v` is set.
pub fn filter_feasibility(
    solutions: &mut [LambertSolution],
    r1: &Vector3<f64>,
    r2: &Vector3<f64>,
    mu: f64,
    config: &FeasibilityConfig,
) {
    filter_feasibility_with_ref(solutions, r1, r2, mu, config, None, None);
}

/// Like [`filter_feasibility`], but accepts optional reference orbit
/// velocities for accurate delta-v computation.
pub fn filter_feasibility_with_ref(
    solutions: &mut [LambertSolution],
    r1: &Vector3<f64>,
    r2: &Vector3<f64>,
    mu: f64,
    config: &FeasibilityConfig,
    v1_ref: Option<&Vector3<f64>>,
    v2_ref: Option<&Vector3<f64>>,
) {
    for sol in solutions.iter_mut() {
        if config.check_earth_collision {
            if !check_periapsis(r1, &sol.v1, sol.a, mu, config.earth_radius) {
                sol.is_feasible = false;
                continue;
            }
        }
        if config.check_escape_velocity {
            let r1_mag = r1.norm();
            let r2_mag = r2.norm();
            let v_esc_1 = (2.0 * mu / r1_mag).sqrt();
            let v_esc_2 = (2.0 * mu / r2_mag).sqrt();
            if sol.v1.norm() > v_esc_1 || sol.v2.norm() > v_esc_2 {
                sol.is_feasible = false;
                continue;
            }
        }
        if let Some(max_dv) = config.max_delta_v {
            // Delta-v requires reference orbit velocities at departure and
            // arrival. Without them we cannot compute a meaningful delta-v.
            if let (Some(v1r), Some(v2r)) = (v1_ref, v2_ref) {
                let dv = (sol.v1 - v1r).norm() + (v2r - sol.v2).norm();
                if dv > max_dv {
                    sol.is_feasible = false;
                }
            }
        }
    }
}

/// Check that the periapsis of the transfer orbit does not dip below
/// `min_radius` (e.g., Earth's surface).
///
/// Uses the angular-momentum / semi-latus-rectum approach:
///   h = |r1 × v1|,  p = h²/μ,  e = sqrt(max(0, 1 - p/a)),  r_p = a(1 - e)
fn check_periapsis(
    r1: &Vector3<f64>,
    v1: &Vector3<f64>,
    a: f64,
    mu: f64,
    min_radius: f64,
) -> bool {
    if a <= 0.0 {
        // Hyperbolic — periapsis check still possible but we only handle
        // elliptic here; allow it through.
        return true;
    }

    let h = r1.cross(v1);
    let h_sq = h.norm_squared();
    let p = h_sq / mu; // semi-latus rectum
    let e_sq = (1.0 - p / a).max(0.0);
    let e = e_sq.sqrt();
    let r_periapsis = a * (1.0 - e);

    r_periapsis >= min_radius
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Branch;
    use nalgebra::Vector3;

    #[test]
    fn test_escape_velocity_filter() {
        let mu = 398600.4418;
        let r1 = Vector3::new(7000.0, 0.0, 0.0);
        let r2 = Vector3::new(0.0, 7000.0, 0.0);
        let mut sols = vec![LambertSolution {
            v1: Vector3::new(100.0, 0.0, 0.0), // absurdly fast
            v2: Vector3::new(0.0, 100.0, 0.0),
            a: 10000.0,
            n_revs: 0,
            branch: Branch::Fractional,
            transfer_angle: std::f64::consts::PI / 2.0,
            is_feasible: true,
        }];
        let cfg = FeasibilityConfig {
            check_escape_velocity: true,
            ..Default::default()
        };
        filter_feasibility(&mut sols, &r1, &r2, mu, &cfg);
        assert!(!sols[0].is_feasible);
    }
}
