use crate::error::LambertError;
use crate::keplerian::geometry::{
    auxiliary_angles, compute_transfer_geometry, determine_n_max, dtof_da, find_a_tmin,
    time_min_energy, time_parabolic, tof_from_a,
};
use crate::keplerian::velocity::terminal_velocities;
use crate::types::{Branch, LambertInput, LambertSolution, TransferGeometry};

/// Solve the Keplerian (two-body) Lambert problem using the Prussing/Ochoa
/// multi-revolution algorithm.
///
/// Returns up to `2*N_max + 1` solutions: one fractional-orbit (N=0) and two
/// per additional revolution (upper and lower branch).
pub fn solve_prussing(input: &LambertInput) -> Result<Vec<LambertSolution>, LambertError> {
    // --- validate inputs ---
    if input.tof <= 0.0 {
        return Err(LambertError::InvalidInput("tof must be positive".into()));
    }
    if input.mu <= 0.0 {
        return Err(LambertError::InvalidInput("mu must be positive".into()));
    }
    let r1_mag = input.r1.norm();
    let r2_mag = input.r2.norm();
    if r1_mag < 1e-12 || r2_mag < 1e-12 {
        return Err(LambertError::InvalidInput(
            "position vectors must be non-zero".into(),
        ));
    }

    let geom = compute_transfer_geometry(&input.r1, &input.r2, input.direction);
    let mu = input.mu;
    let tof = input.tof;

    // Parabolic lower bound for N=0
    let t_p = time_parabolic(geom.s, geom.c, geom.theta, mu);
    if tof < t_p {
        return Err(LambertError::InvalidInput(
            "time of flight is below parabolic minimum — hyperbolic transfers not supported".into(),
        ));
    }

    // Minimum-energy time for N=0
    let t_m0 = time_min_energy(geom.s, geom.c, geom.theta, 0, mu);

    // --- determine N_max ---
    let n_max = determine_n_max(&geom, tof, mu, input.max_revs);

    let mut solutions: Vec<LambertSolution> = Vec::with_capacity(2 * n_max as usize + 1);

    // --- N = 0 (fractional orbit) ---
    // For N=0 there is exactly one solution. The branch depends on tof vs t_m.
    let upper_n0 = tof > t_m0;
    if let Some(sol) = solve_branch(input, &geom, 0, upper_n0)? {
        solutions.push(sol);
    }

    // --- N >= 1: upper and lower branches ---
    for n in 1..=n_max {
        let t_m_n = time_min_energy(geom.s, geom.c, geom.theta, n, mu);

        if tof > t_m_n {
            // Standard case: one root on lower branch, one on upper
            if let Some(sol) = solve_branch(input, &geom, n, false)? {
                solutions.push(sol);
            }
            if let Some(sol) = solve_branch(input, &geom, n, true)? {
                solutions.push(sol);
            }
        } else {
            // Edge case: t_min_N ≤ tof ≤ t_m_N — both roots on lower branch,
            // one on each side of a_tmin (Paper 2, Fig. 5).
            if let Some(a_tmin) = find_a_tmin(geom.s, geom.c, geom.theta, n, mu) {
                // Root 1: a ∈ (a_m, a_tmin) — the "fast" lower-branch root
                if let Some(sol) = solve_branch_bounded(input, &geom, n, false, geom.a_min * 1.00001, a_tmin)? {
                    solutions.push(sol);
                }
                // Root 2: a ∈ (a_tmin, ∞) — the "slow" lower-branch root
                if let Some(sol) = solve_branch_bounded(input, &geom, n, false, a_tmin, a_tmin * 100.0)? {
                    solutions.push(sol);
                }
            }
        }
    }

    if solutions.is_empty() {
        return Err(LambertError::NoSolution);
    }

    Ok(solutions)
}

/// Solve for a single branch using bisection with explicit bounds on `a`.
///
/// Used for the edge case where both N≥1 roots are on the lower branch.
fn solve_branch_bounded(
    input: &LambertInput,
    geom: &TransferGeometry,
    n_revs: u32,
    upper: bool,
    lo_init: f64,
    hi_init: f64,
) -> Result<Option<LambertSolution>, LambertError> {
    let mu = input.mu;
    let tof = input.tof;

    let f = |a: f64| -> f64 {
        let (alpha, beta) = auxiliary_angles(a, geom.s, geom.c, geom.theta, upper);
        tof_from_a(a, alpha, beta, n_revs, mu) - tof
    };

    let mut lo = lo_init;
    let mut hi = hi_init;

    // Verify bracket
    if f(lo) * f(hi) > 0.0 {
        return Ok(None);
    }

    // Bisection
    for _ in 0..100 {
        let mid = (lo + hi) / 2.0;
        let f_mid = f(mid);

        if f_mid.abs() < 1e-10 || (hi - lo) < 1e-14 * geom.a_min {
            let (alpha, beta) =
                auxiliary_angles(mid, geom.s, geom.c, geom.theta, upper);
            let (v1, v2) =
                terminal_velocities(&input.r1, &input.r2, geom, mid, alpha, beta, mu);
            let branch = match n_revs {
                0 => Branch::Fractional,
                _ if upper => Branch::Upper,
                _ => Branch::Lower,
            };
            return Ok(Some(LambertSolution {
                v1,
                v2,
                a: mid,
                n_revs,
                branch,
                transfer_angle: geom.theta,
                is_feasible: true,
            }));
        }

        if f(lo) * f_mid <= 0.0 {
            hi = mid;
        } else {
            lo = mid;
        }
    }

    Ok(None)
}

/// Solve for a single branch using Newton's method on the semi-major axis `a`,
/// with bisection as fallback if Newton diverges.
///
/// Returns `Ok(None)` when the branch is geometrically infeasible.
fn solve_branch(
    input: &LambertInput,
    geom: &TransferGeometry,
    n_revs: u32,
    upper: bool,
) -> Result<Option<LambertSolution>, LambertError> {
    let mu = input.mu;
    let tof = input.tof;
    let a_m = geom.s / 2.0;

    // Residual function: positive when computed tof > desired tof.
    let f = |a: f64| -> f64 {
        let (alpha, beta) = auxiliary_angles(a, geom.s, geom.c, geom.theta, upper);
        tof_from_a(a, alpha, beta, n_revs, mu) - tof
    };

    // --- Bracket the root in [lo, hi] ---
    let mut lo = a_m * 1.00001;
    let mut hi = a_m * 2.0;

    // Physical cap: no legitimate transfer semi-major axis should ever
    // exceed ~10⁸·a_min (≈ 4600 AU for a LEO geometry). If we can't
    // bracket a root inside that, there is no physical solution on
    // this branch.
    let hi_cap = geom.a_min * 1e8;
    let f_lo_init = f(lo);
    for _ in 0..60 {
        let f_hi = f(hi);
        if f_lo_init * f_hi <= 0.0 {
            break;
        }
        hi *= 2.0;
        if hi > hi_cap {
            return Ok(None);
        }
    }

    if f(lo) * f(hi) > 0.0 {
        return Ok(None);
    }

    // --- Newton's method with bisection safeguard ---
    let mut a = (lo + hi) / 2.0;
    let newton_tol = 1e-12;

    for _ in 0..crate::constants::NEWTON_MAX_ITER {
        let (alpha, beta) = auxiliary_angles(a, geom.s, geom.c, geom.theta, upper);
        let residual = tof_from_a(a, alpha, beta, n_revs, mu) - tof;

        if residual.abs() < newton_tol {
            let (v1, v2) =
                terminal_velocities(&input.r1, &input.r2, geom, a, alpha, beta, mu);
            let branch = match n_revs {
                0 => Branch::Fractional,
                _ if upper => Branch::Upper,
                _ => Branch::Lower,
            };
            return Ok(Some(LambertSolution {
                v1,
                v2,
                a,
                n_revs,
                branch,
                transfer_angle: geom.theta,
                is_feasible: true,
            }));
        }

        let deriv = dtof_da(a, alpha, beta, n_revs, mu);

        // Newton step
        let a_new = if deriv.abs() > 1e-30 {
            a - residual / deriv
        } else {
            // Degenerate derivative — fall back to bisection step
            (lo + hi) / 2.0
        };

        // If Newton step is outside bracket, use bisection instead
        let a_new = if a_new <= lo || a_new >= hi {
            (lo + hi) / 2.0
        } else {
            a_new
        };

        // Update bracket
        if f(lo) * residual <= 0.0 {
            hi = a;
        } else {
            lo = a;
        }

        a = a_new;

        if (hi - lo) < 1e-14 * a_m {
            // Bracket converged
            let (alpha, beta) = auxiliary_angles(a, geom.s, geom.c, geom.theta, upper);
            let (v1, v2) =
                terminal_velocities(&input.r1, &input.r2, geom, a, alpha, beta, mu);
            let branch = match n_revs {
                0 => Branch::Fractional,
                _ if upper => Branch::Upper,
                _ => Branch::Lower,
            };
            return Ok(Some(LambertSolution {
                v1,
                v2,
                a,
                n_revs,
                branch,
                transfer_angle: geom.theta,
                is_feasible: true,
            }));
        }
    }

    Ok(None) // did not converge
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Direction;
    use approx::assert_relative_eq;
    use nalgebra::Vector3;

    /// Basic smoke test: a 90-degree coplanar LEO transfer should produce
    /// exactly one N=0 solution with reasonable velocity magnitudes.
    #[test]
    fn test_solve_n0_coplanar() {
        let input = LambertInput {
            r1: Vector3::new(7000.0, 0.0, 0.0),
            r2: Vector3::new(0.0, 7000.0, 0.0),
            tof: 2000.0,
            mu: 398600.4418,
            direction: Direction::Prograde,
            max_revs: Some(0),
        };
        let sols = solve_prussing(&input).expect("should find a solution");
        assert!(!sols.is_empty(), "should have at least one solution");
        let sol = &sols[0];
        assert_eq!(sol.n_revs, 0);

        // The velocity magnitudes should be on the order of ~7 km/s for LEO
        let v1_mag = sol.v1.norm();
        let v2_mag = sol.v2.norm();
        assert!(v1_mag > 1.0 && v1_mag < 20.0, "v1 = {v1_mag} out of range");
        assert!(v2_mag > 1.0 && v2_mag < 20.0, "v2 = {v2_mag} out of range");
    }

    /// The N=0 transfer angle for a prograde 90-degree geometry should be pi/2.
    #[test]
    fn test_transfer_angle() {
        let input = LambertInput {
            r1: Vector3::new(7000.0, 0.0, 0.0),
            r2: Vector3::new(0.0, 7000.0, 0.0),
            tof: 2000.0,
            mu: 398600.4418,
            direction: Direction::Prograde,
            max_revs: Some(0),
        };
        let sols = solve_prussing(&input).unwrap();
        assert_relative_eq!(
            sols[0].transfer_angle,
            std::f64::consts::PI / 2.0,
            epsilon = 1e-8
        );
    }
}
