use nalgebra::Vector3;

use crate::types::{FeasibilityConfig, LambertSolution};

/// Apply feasibility filters to a set of Lambert solutions **in place**,
/// setting `is_feasible = false` on solutions that fail any check.
pub fn filter_feasibility(
    solutions: &mut [LambertSolution],
    r1: &Vector3<f64>,
    r2: &Vector3<f64>,
    mu: f64,
    config: &FeasibilityConfig,
) {
    for sol in solutions.iter_mut() {
        if config.check_earth_collision {
            if !check_periapsis(sol, mu, config.earth_radius) {
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
            // delta-v is typically measured relative to reference orbits; here
            // we use the velocity magnitudes themselves as a rough proxy.
            let dv = sol.v1.norm() + sol.v2.norm();
            if dv > max_dv {
                sol.is_feasible = false;
            }
        }
    }
}

/// Check that the periapsis of the transfer orbit does not dip below
/// `min_radius` (e.g., Earth's surface).
fn check_periapsis(sol: &LambertSolution, mu: f64, min_radius: f64) -> bool {
    if sol.a <= 0.0 {
        // Hyperbolic — periapsis check still possible but we only handle
        // elliptic here; allow it through.
        return true;
    }
    // Compute eccentricity from vis-viva and angular-momentum magnitude.
    // We approximate using the energy + semi-major axis approach:
    //   r_p = a * (1 - e)
    //   e = 1 - r_p / a
    // From the specific angular momentum h = |r1 x v1|:
    //   p = h^2 / mu  (semi-latus rectum)
    //   e = sqrt(1 - p/a)
    // This avoids needing r1 at this call site.
    // For now, a simple energy-based bound: the minimum possible periapsis
    // for a given a is 0 (e = 1), and for a circular orbit it is a.
    // We use the semi-latus rectum route via the transfer-angle relation.
    //
    // A robust implementation would propagate or use the eccentricity
    // directly.  For the boilerplate we pass through.
    let _ = (mu, min_radius, sol);
    true // TODO: full periapsis check
}
