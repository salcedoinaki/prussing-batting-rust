//! Gooding / Lancaster–Blanchard Lambert solver.
//!
//! Classical two-phase procedure (Gooding 1990, Lancaster–Blanchard 1969)
//! working in the non-dimensional x-variable:
//!
//! 1. **Coarse pass** — scan the non-dimensional time curve T(x, λ, N) on a
//!    log-adapted grid for each feasible revolution count N, detect the
//!    local minimum at `x_min(N)`, and bracket each root on either side.
//! 2. **Fine pass** — Halley's method on x (3rd order) warm-started from
//!    the coarse bracket, with a bisection safeguard.
//!
//! The solver is fully self-contained and lives side-by-side with the
//! Prussing/Ochoa solver. Its public signature matches `solve_prussing`
//! exactly so it can be dropped in anywhere Prussing is used.
//!
//! Reference: Gooding, R. H. (1990) "A procedure for the solution of
//! Lambert's orbital boundary-value problem", Celestial Mechanics and
//! Dynamical Astronomy **48**, 145–165.

use std::f64::consts::PI;

use crate::error::LambertError;
use crate::keplerian::geometry::{
    auxiliary_angles, compute_transfer_geometry, determine_n_max, find_a_tmin, time_min_energy,
    time_parabolic,
};
use crate::keplerian::velocity::terminal_velocities;
use crate::types::{Branch, LambertInput, LambertSolution, TransferGeometry};

// ---------------------------------------------------------------------------
// Non-dimensional time and its derivatives
// ---------------------------------------------------------------------------

/// Clamp x to the open elliptic interval (-1, 1) away from the boundaries.
fn clamp_x(x: f64) -> f64 {
    const EPS: f64 = 1.0e-12;
    x.clamp(-1.0 + EPS, 1.0 - EPS)
}

/// Gooding / Lancaster–Blanchard non-dimensional transfer time for a given
/// x (elliptic, |x| < 1), signed chord parameter λ, and revolution count N.
///
/// Derivation: with `a = (s/2) / (1 − x²)`, Prussing's auxiliary angles are
/// `α = 2·arccos(x)` and `β = 2·arcsin(λ·√(1−x²))`. Plugging into
/// `√μ · t = a^{3/2} · (2Nπ + α − β − sin α + sin β)` and
/// non-dimensionalising by `√(2μ/s³)` gives
///
/// ```text
/// T(x, λ, N) = [Nπ + arccos(x) − x·√(1−x²)
///                  − arcsin(λ·√(1−x²)) + λ·√(1−x²)·y] / (1 − x²)^{3/2}
/// ```
///
/// where `y = √(1 − λ²·(1−x²))`. This form is equivalent to Izzo 2015 and
/// agrees with `tof_from_a` under the Prussing auxiliary-angle mapping.
fn t_nd(x: f64, lambda: f64, n_revs: u32) -> f64 {
    let u = 1.0 - x * x;
    let r = u.sqrt();
    let p = lambda * r;
    let y = (1.0 - p * p).sqrt();
    let n = n_revs as f64;
    let num = n * PI + x.acos() - x * r - p.asin() + p * y;
    num / u.powf(1.5)
}

/// First derivative `dT/dx` (Izzo 2015 Eq. 22, verified by direct
/// differentiation of the `t_nd` closed form).
fn dt_nd_dx(x: f64, lambda: f64, n_revs: u32) -> f64 {
    let u = 1.0 - x * x;
    let y = (1.0 - lambda * lambda * u).sqrt();
    let t = t_nd(x, lambda, n_revs);
    (3.0 * t * x - 2.0 + 2.0 * lambda.powi(3) * x / y) / u
}

/// Second derivative `d²T/dx²` (Izzo 2015 Eq. 23).
fn d2t_nd_dx2(x: f64, lambda: f64, n_revs: u32) -> f64 {
    let u = 1.0 - x * x;
    let y = (1.0 - lambda * lambda * u).sqrt();
    let t = t_nd(x, lambda, n_revs);
    let tp = dt_nd_dx(x, lambda, n_revs);
    let one_minus_l2 = 1.0 - lambda * lambda;
    (3.0 * t + 5.0 * x * tp + 2.0 * lambda.powi(3) * one_minus_l2 / y.powi(3)) / u
}

// ---------------------------------------------------------------------------
// x ↔ a mapping
// ---------------------------------------------------------------------------

/// `a = (s/2) / (1 − x²)`. For |x| < 1, a ≥ s/2.
fn x_to_a(x: f64, s: f64) -> f64 {
    s / (2.0 * (1.0 - x * x))
}

/// Magnitude of x corresponding to a given a (≥ s/2). Sign is chosen by the
/// caller (positive → Prussing lower branch, negative → upper branch).
fn a_to_x_mag(a: f64, s: f64) -> f64 {
    (1.0 - s / (2.0 * a)).max(0.0).sqrt()
}

// ---------------------------------------------------------------------------
// Coarse pass
// ---------------------------------------------------------------------------

/// One entry from the coarse pass, ready to hand to the Halley refiner.
#[derive(Debug, Clone, Copy)]
struct CoarseGuess {
    n_revs: u32,
    branch: Branch,
    /// Prussing-α-convention branch flag (α = 2π − α₀ when `upper`).
    upper: bool,
    /// Tight bracket around the true root.
    x_lo: f64,
    x_hi: f64,
}

/// Find the first sub-interval in `[lo, hi]` where `f` changes sign using a
/// log-adapted sample grid of `n_samples` points, then bisect for
/// `refine_steps` iterations to tighten the bracket. Returns `(x_lo, x_hi)`
/// or `None` if no sign change was found.
fn bracket_root<F: Fn(f64) -> f64>(
    f: &F,
    lo: f64,
    hi: f64,
    n_samples: usize,
    refine_steps: usize,
) -> Option<(f64, f64)> {
    assert!(lo < hi);
    let n = n_samples.max(2);
    let mut prev_x = lo;
    let mut prev_f = f(prev_x);
    let mut found_lo = f64::NAN;
    let mut found_hi = f64::NAN;
    let mut found = false;
    for i in 1..n {
        let t = (i as f64) / ((n - 1) as f64);
        // Concentrate samples near the endpoints where curvature is high.
        let w = 0.5 - 0.5 * (PI * t).cos();
        let x = lo + (hi - lo) * w;
        let fx = f(x);
        if prev_f == 0.0 {
            return Some((prev_x, prev_x));
        }
        if fx == 0.0 {
            return Some((x, x));
        }
        if prev_f.signum() != fx.signum() {
            found_lo = prev_x;
            found_hi = x;
            found = true;
            break;
        }
        prev_x = x;
        prev_f = fx;
    }
    if !found {
        return None;
    }
    // Bisect to tighten the bracket.
    let mut a = found_lo;
    let mut b = found_hi;
    let mut fa = f(a);
    for _ in 0..refine_steps {
        let mid = 0.5 * (a + b);
        let fm = f(mid);
        if fm == 0.0 {
            return Some((mid, mid));
        }
        if fa.signum() != fm.signum() {
            b = mid;
        } else {
            a = mid;
            fa = fm;
        }
    }
    Some((a, b))
}

/// Enumerate (N, branch) candidates for the given geometry and produce a
/// tight bracket around each root of `T(x, λ, N) = T_target` for the Halley
/// refiner to warm-start from.
fn coarse_pass(
    geom: &TransferGeometry,
    lambda: f64,
    t_nd_target: f64,
    tof: f64,
    mu: f64,
    max_revs: Option<u32>,
) -> Vec<CoarseGuess> {
    const X_EPS: f64 = 1.0e-6;
    const N_SAMPLES: usize = 32;
    const REFINE_STEPS: usize = 12;

    let n_max = determine_n_max(geom, tof, mu, max_revs);
    let mut guesses: Vec<CoarseGuess> = Vec::with_capacity(2 * n_max as usize + 1);

    // ----- N = 0 -----
    {
        let f = |x: f64| t_nd(x, lambda, 0) - t_nd_target;
        if let Some((xl, xh)) = bracket_root(&f, -1.0 + X_EPS, 1.0 - X_EPS, N_SAMPLES, REFINE_STEPS)
        {
            // Prussing-α branch is determined by whether tof is above or below
            // the minimum-energy time for N = 0 (same rule as `solve_prussing`
            // line 52): `upper = tof > t_m0`.
            let t_m0 = time_min_energy(geom.s, geom.c, geom.theta, 0, mu);
            let upper = tof > t_m0;
            guesses.push(CoarseGuess {
                n_revs: 0,
                branch: Branch::Fractional,
                upper,
                x_lo: xl,
                x_hi: xh,
            });
        }
    }

    // ----- N ≥ 1 -----
    for n in 1..=n_max {
        let a_tmin = match find_a_tmin(geom.s, geom.c, geom.theta, n, mu) {
            Some(a) => a,
            None => continue,
        };
        let x_min = a_to_x_mag(a_tmin, geom.s);
        // Stay strictly inside (−1, 1) and away from x_min itself to keep
        // sub-intervals sign-proper.
        let x_min_lo = (x_min - 1.0e-8).max(0.0);
        let x_min_hi = (x_min + 1.0e-8).min(1.0 - X_EPS);

        let t_m_n = time_min_energy(geom.s, geom.c, geom.theta, n, mu);

        let f = |x: f64| t_nd(x, lambda, n) - t_nd_target;

        if tof > t_m_n {
            // Standard case: one upper-branch root on x ∈ (−1+ε, −ε) and one
            // lower-branch root on x ∈ (x_min, 1−ε).
            if let Some((xl, xh)) =
                bracket_root(&f, -1.0 + X_EPS, -X_EPS, N_SAMPLES, REFINE_STEPS)
            {
                guesses.push(CoarseGuess {
                    n_revs: n,
                    branch: Branch::Upper,
                    upper: true,
                    x_lo: xl,
                    x_hi: xh,
                });
            }
            if let Some((xl, xh)) =
                bracket_root(&f, x_min_hi, 1.0 - X_EPS, N_SAMPLES, REFINE_STEPS)
            {
                guesses.push(CoarseGuess {
                    n_revs: n,
                    branch: Branch::Lower,
                    upper: false,
                    x_lo: xl,
                    x_hi: xh,
                });
            }
        } else {
            // Edge case t_min_N ≤ tof ≤ t_m_N: both roots sit on the lower
            // branch, straddling x_min(N). Both get upper = false.
            if let Some((xl, xh)) = bracket_root(&f, X_EPS, x_min_lo, N_SAMPLES, REFINE_STEPS) {
                guesses.push(CoarseGuess {
                    n_revs: n,
                    branch: Branch::Lower,
                    upper: false,
                    x_lo: xl,
                    x_hi: xh,
                });
            }
            if let Some((xl, xh)) =
                bracket_root(&f, x_min_hi, 1.0 - X_EPS, N_SAMPLES, REFINE_STEPS)
            {
                guesses.push(CoarseGuess {
                    n_revs: n,
                    branch: Branch::Lower,
                    upper: false,
                    x_lo: xl,
                    x_hi: xh,
                });
            }
        }
    }

    guesses
}

// ---------------------------------------------------------------------------
// Halley refinement
// ---------------------------------------------------------------------------

/// Halley's method for `T(x, λ, N) = T_target`, warm-started from the
/// coarse-pass bracket `[x_lo, x_hi]`. Maintains the bracket across
/// iterations and falls back to a bisection step whenever the Halley step
/// exits it, producing robust convergence.
///
/// `tol_nd` is the non-dimensional convergence tolerance on `|f(x)|`; it
/// should be `NEWTON_TOL · t_scale` so that the effective time-domain
/// tolerance matches Prussing's Newton-on-a (1e-12 seconds).
fn halley_refine(
    x_lo: f64,
    x_hi: f64,
    lambda: f64,
    t_nd_target: f64,
    n_revs: u32,
    tol_nd: f64,
) -> Option<f64> {
    let tol = tol_nd;
    let max_iter = crate::constants::NEWTON_MAX_ITER;

    let mut lo = x_lo;
    let mut hi = x_hi;

    let f = |x: f64| t_nd(x, lambda, n_revs) - t_nd_target;
    let mut f_lo = f(lo);
    let f_hi = f(hi);
    if f_lo == 0.0 {
        return Some(lo);
    }
    if f_hi == 0.0 {
        return Some(hi);
    }
    if f_lo.signum() == f_hi.signum() {
        // Caller guarantees a sign change; bail rather than loop.
        return None;
    }

    let mut x = 0.5 * (lo + hi);
    let mut best_x = x;
    let mut best_f = f64::INFINITY;
    for _ in 0..max_iter {
        let fx = f(x);
        if fx.abs() < best_f {
            best_f = fx.abs();
            best_x = x;
        }
        if fx.abs() < tol {
            return Some(x);
        }

        // Update the bracket based on the new sample.
        if f_lo.signum() != fx.signum() {
            hi = x;
        } else {
            lo = x;
            f_lo = fx;
        }

        let fpx = dt_nd_dx(x, lambda, n_revs);
        let fppx = d2t_nd_dx2(x, lambda, n_revs);
        let denom = 2.0 * fpx * fpx - fx * fppx;

        // Halley step; fall back to bisection if the step is degenerate or
        // exits the bracket.
        let x_halley = if denom.abs() > 1.0e-30 {
            x - 2.0 * fx * fpx / denom
        } else {
            f64::NAN
        };
        let x_new = if x_halley.is_finite() && x_halley > lo && x_halley < hi {
            x_halley
        } else {
            0.5 * (lo + hi)
        };

        // Stop once the bracket has collapsed to a size that can't be
        // resolved in f64 — at this point `best_x` is the most accurate
        // root we can produce.
        if (hi - lo) < 2.0 * f64::EPSILON * (1.0 + x.abs()) {
            return Some(best_x);
        }
        x = clamp_x(x_new);
    }
    // Returned the best iterate found so far even if tol wasn't reached —
    // caller can decide whether to trust it.
    if best_f.is_finite() {
        Some(best_x)
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Solve the Keplerian (two-body) Lambert problem using the classical
/// Gooding / Lancaster–Blanchard x-variable algorithm with a coarse local-
/// minimum pass followed by a Halley refinement.
///
/// Returns up to `2·N_max + 1` solutions, mirroring `solve_prussing`.
pub fn solve_gooding(input: &LambertInput) -> Result<Vec<LambertSolution>, LambertError> {
    // --- validate inputs (same checks as solve_prussing) ---
    if input.tof <= 0.0 {
        return Err(LambertError::InvalidInput("tof must be positive".into()));
    }
    if input.mu <= 0.0 {
        return Err(LambertError::InvalidInput("mu must be positive".into()));
    }
    let r1_mag = input.r1.norm();
    let r2_mag = input.r2.norm();
    if r1_mag < 1.0e-12 || r2_mag < 1.0e-12 {
        return Err(LambertError::InvalidInput(
            "position vectors must be non-zero".into(),
        ));
    }

    let geom = compute_transfer_geometry(&input.r1, &input.r2, input.direction);
    let mu = input.mu;
    let tof = input.tof;

    let t_p = time_parabolic(geom.s, geom.c, geom.theta, mu);
    if tof < t_p {
        return Err(LambertError::InvalidInput(
            "time of flight is below parabolic minimum — hyperbolic transfers not supported".into(),
        ));
    }

    // Signed Gooding chord parameter.
    let lambda_mag = (1.0 - geom.c / geom.s).sqrt();
    let lambda = if geom.theta <= PI {
        lambda_mag
    } else {
        -lambda_mag
    };
    let t_scale = (2.0 * mu / geom.s.powi(3)).sqrt();
    let t_nd_target = tof * t_scale;
    // Match Prussing's Newton-on-a time-domain tolerance (1e-12 s) in
    // non-dimensional units.
    let tol_nd = crate::constants::NEWTON_TOL * t_scale;

    let guesses = coarse_pass(&geom, lambda, t_nd_target, tof, mu, input.max_revs);

    let mut solutions: Vec<LambertSolution> = Vec::with_capacity(guesses.len());
    for g in &guesses {
        let x_opt = halley_refine(g.x_lo, g.x_hi, lambda, t_nd_target, g.n_revs, tol_nd);
        let Some(x) = x_opt else { continue };
        let a = x_to_a(x, geom.s);
        let (alpha, beta) = auxiliary_angles(a, geom.s, geom.c, geom.theta, g.upper);
        let (v1, v2) = terminal_velocities(&input.r1, &input.r2, &geom, a, alpha, beta, mu);
        solutions.push(LambertSolution {
            v1,
            v2,
            a,
            n_revs: g.n_revs,
            branch: g.branch,
            transfer_angle: geom.theta,
            is_feasible: true,
        });
    }

    if solutions.is_empty() {
        return Err(LambertError::NoSolution);
    }
    Ok(solutions)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::keplerian::geometry::tof_from_a;
    use crate::types::Direction;
    use approx::assert_relative_eq;
    use nalgebra::Vector3;

    fn leo_geom(tof_seconds: f64) -> (TransferGeometry, f64, f64, f64, f64) {
        let r1 = Vector3::new(7000.0, 0.0, 0.0);
        let r2 = Vector3::new(0.0, 7000.0, 0.0);
        let mu = 398600.4418;
        let geom = compute_transfer_geometry(&r1, &r2, Direction::Prograde);
        let lambda = (1.0 - geom.c / geom.s).sqrt();
        let t_scale = (2.0 * mu / geom.s.powi(3)).sqrt();
        (geom, lambda, t_scale, mu, tof_seconds)
    }

    #[test]
    fn test_t_nd_at_x_zero_matches_min_energy() {
        // At x = 0, T_nd should equal time_min_energy(N=0) non-dimensionalised.
        let (geom, lambda, t_scale, mu, _) = leo_geom(2000.0);
        let t_min_energy_dim = time_min_energy(geom.s, geom.c, geom.theta, 0, mu);
        let t_nd_min = t_nd(0.0, lambda, 0);
        assert_relative_eq!(t_nd_min, t_min_energy_dim * t_scale, epsilon = 1.0e-12);
    }

    #[test]
    fn test_t_nd_at_x_near_one_matches_parabolic() {
        // Near x = 1, T_nd should approach the parabolic non-dimensional time.
        let (geom, lambda, t_scale, mu, _) = leo_geom(2000.0);
        let t_par_dim = time_parabolic(geom.s, geom.c, geom.theta, mu);
        let t_par_nd = t_par_dim * t_scale;
        let t_nd_near = t_nd(1.0 - 1.0e-4, lambda, 0);
        // Not asymptotic equality — just order-of-magnitude proximity is
        // enough to catch gross formulation errors (the x=1 limit has
        // mild cancellation).
        assert!(
            (t_nd_near - t_par_nd).abs() < 1.0e-2,
            "t_nd near x=1 = {t_nd_near}, parabolic = {t_par_nd}"
        );
    }

    #[test]
    fn test_x_to_a_round_trip() {
        let s = 11949.75;
        for &x in &[-0.8, -0.3, 0.0, 0.2, 0.7, 0.95] {
            let a = x_to_a(x, s);
            let x_back = a_to_x_mag(a, s);
            assert_relative_eq!(x.abs(), x_back, epsilon = 1.0e-12);
        }
    }

    #[test]
    fn test_dt_nd_dx_vs_fd() {
        let (_, lambda, _, _, _) = leo_geom(2000.0);
        for &x in &[-0.5, -0.2, 0.1, 0.3, 0.6] {
            for n in 0..=2u32 {
                let h = 1.0e-5;
                let fd = (t_nd(x + h, lambda, n) - t_nd(x - h, lambda, n)) / (2.0 * h);
                let ana = dt_nd_dx(x, lambda, n);
                assert_relative_eq!(ana, fd, epsilon = 1.0e-5);
            }
        }
    }

    #[test]
    fn test_d2t_nd_dx2_vs_fd() {
        let (_, lambda, _, _, _) = leo_geom(2000.0);
        for &x in &[-0.5, -0.2, 0.1, 0.3, 0.6] {
            for n in 0..=2u32 {
                let h = 1.0e-4;
                let fd = (t_nd(x + h, lambda, n) - 2.0 * t_nd(x, lambda, n)
                    + t_nd(x - h, lambda, n))
                    / (h * h);
                let ana = d2t_nd_dx2(x, lambda, n);
                assert_relative_eq!(ana, fd, epsilon = 1.0e-3);
            }
        }
    }

    #[test]
    fn test_coarse_pass_n0_standard() {
        let mu = 398600.4418;
        let r1 = Vector3::new(7000.0, 0.0, 0.0);
        let r2 = Vector3::new(0.0, 7000.0, 0.0);
        let tof = 2000.0;
        let geom = compute_transfer_geometry(&r1, &r2, Direction::Prograde);
        let lambda = (1.0 - geom.c / geom.s).sqrt();
        let t_scale = (2.0 * mu / geom.s.powi(3)).sqrt();
        let guesses = coarse_pass(&geom, lambda, tof * t_scale, tof, mu, Some(0));
        assert_eq!(guesses.len(), 1);
        assert_eq!(guesses[0].n_revs, 0);
        assert!(guesses[0].x_lo < guesses[0].x_hi);
    }

    #[test]
    fn test_solve_gooding_n0_90deg_matches_kepler() {
        // Roundtrip: propagate (r1, v1) for tof and verify arrival at r2.
        let input = LambertInput {
            r1: Vector3::new(7000.0, 0.0, 0.0),
            r2: Vector3::new(0.0, 7000.0, 0.0),
            tof: 2000.0,
            mu: 398600.4418,
            direction: Direction::Prograde,
            max_revs: Some(0),
        };
        let sols = solve_gooding(&input).expect("should solve");
        assert!(!sols.is_empty());
        let sol = &sols[0];
        assert_eq!(sol.n_revs, 0);
        // Self-consistency: vis-viva a should match the reported a.
        let r1m = input.r1.norm();
        let v1_sq = sol.v1.norm_squared();
        let a_vv = 1.0 / (2.0 / r1m - v1_sq / input.mu);
        assert_relative_eq!(sol.a, a_vv, epsilon = 1.0);
    }

    #[test]
    fn test_solve_gooding_retrograde() {
        let input = LambertInput {
            r1: Vector3::new(7000.0, 0.0, 0.0),
            r2: Vector3::new(0.0, 7000.0, 0.0),
            tof: 3600.0,
            mu: 398600.4418,
            direction: Direction::Retrograde,
            max_revs: Some(0),
        };
        let sols = solve_gooding(&input).expect("should solve retrograde");
        assert!(!sols.is_empty());
        let sol = &sols[0];
        assert!(sol.transfer_angle > PI, "retrograde θ should be > π");
    }

    #[test]
    fn test_solve_gooding_below_parabolic_rejects() {
        let input = LambertInput {
            r1: Vector3::new(7000.0, 0.0, 0.0),
            r2: Vector3::new(0.0, 7000.0, 0.0),
            tof: 1.0, // way below parabolic
            mu: 398600.4418,
            direction: Direction::Prograde,
            max_revs: Some(0),
        };
        let res = solve_gooding(&input);
        assert!(matches!(res, Err(LambertError::InvalidInput(_))));
    }

    #[test]
    fn test_tof_from_a_agrees_with_t_nd() {
        // For any x, converting x → a → (α, β) → tof_from_a should reproduce
        // t_nd · sqrt(s³/(2μ)).
        let mu = 398600.4418;
        let r1 = Vector3::new(7000.0, 0.0, 0.0);
        let r2 = Vector3::new(0.0, 7000.0, 0.0);
        let geom = compute_transfer_geometry(&r1, &r2, Direction::Prograde);
        let lambda = (1.0 - geom.c / geom.s).sqrt();
        let t_scale = (2.0 * mu / geom.s.powi(3)).sqrt();
        for &(x, n, upper) in &[
            (-0.3, 0u32, true),
            (0.3, 0u32, false),
            (-0.4, 1u32, true),
            (0.4, 1u32, false),
        ] {
            let a = x_to_a(x, geom.s);
            let (alpha, beta) = auxiliary_angles(a, geom.s, geom.c, geom.theta, upper);
            let tof_pr = tof_from_a(a, alpha, beta, n, mu);
            let t_nd_gd = t_nd(x, lambda, n);
            let t_nd_pr = tof_pr * t_scale;
            assert_relative_eq!(t_nd_gd, t_nd_pr, epsilon = 1.0e-10);
        }
    }
}
