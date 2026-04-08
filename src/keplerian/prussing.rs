use crate::error::LambertError;
use crate::keplerian::geometry::{
    auxiliary_angles, compute_transfer_geometry, time_min_energy, time_parabolic, tof_from_a,
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

    let _geom = compute_transfer_geometry(&input.r1, &input.r2, input.direction);

    // TODO: implement solver
    Err(LambertError::NoSolution)
}

/// Find the maximum revolution count whose minimum-energy transfer time does
/// not exceed the desired time of flight.
fn determine_n_max(geom: &TransferGeometry, tof: f64, mu: f64, max_revs: Option<u32>) -> u32 {
    let cap = max_revs.unwrap_or(u32::MAX);
    let mut n: u32 = 0;
    loop {
        let next = n + 1;
        if next > cap {
            return n;
        }
        let t_min_n = time_min_energy(geom.s, geom.c, geom.theta, next, mu);
        if t_min_n > tof {
            return n;
        }
        n = next;
    }
}

/// Solve for a single branch using bisection on the semi-major axis `a`.
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
    let mut lo = a_m * 1.00001; // just above minimum energy
    let mut hi = a_m * 2.0;

    let f_lo_init = f(lo);
    // Grow hi until we bracket (sign change) or give up
    for _ in 0..60 {
        let f_hi = f(hi);
        if f_lo_init * f_hi <= 0.0 {
            break;
        }
        hi *= 2.0;
        if hi > 1e15 {
            return Ok(None); // infeasible
        }
    }

    // Verify bracket
    if f(lo) * f(hi) > 0.0 {
        return Ok(None);
    }

    // --- Bisection ---
    for _ in 0..100 {
        let mid = (lo + hi) / 2.0;
        let f_mid = f(mid);

        if f_mid.abs() < 1e-10 || (hi - lo) < 1e-14 * a_m {
            // Converged
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

    Ok(None) // did not converge
}
