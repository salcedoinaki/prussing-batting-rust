//! # Unified Lambert Tool (ULT)
//!
//! Multi-revolution Keplerian Lambert solver based on the Prussing/Ochoa
//! algorithm, with MCPI-based perturbed refinement.
//!
//! ## Quick start — Keplerian only
//!
//! ```rust
//! use nalgebra::Vector3;
//! use lambert_ult::{solve_lambert, types::{LambertInput, Direction}};
//!
//! let input = LambertInput {
//!     r1: Vector3::new(7000.0, 0.0, 0.0),
//!     r2: Vector3::new(0.0, 7000.0, 0.0),
//!     tof: 2000.0,
//!     mu: 398600.4418,
//!     direction: Direction::Prograde,
//!     max_revs: None,
//! };
//! let solutions = solve_lambert(&input).unwrap();
//! for sol in &solutions {
//!     println!("v1 = {}, v2 = {}, a = {:.1} km", sol.v1, sol.v2, sol.a);
//! }
//! ```

pub mod constants;
pub mod error;
pub mod force_models;
pub mod keplerian;
pub mod perturbed;
pub mod types;

use error::LambertError;
use force_models::ForceModel;
use nalgebra::Vector3;
use perturbed::selector::{select_algorithm, PerturbedAlgorithm};
use types::{
    FeasibilityConfig, LambertInput, LambertSolution, PerturbedConfig, PerturbedSolution,
    UnifiedLambertConfig,
};

// =========================================================================
// Keplerian-only API (existing)
// =========================================================================

/// Solve the Lambert problem, returning all feasible Keplerian solutions
/// (up to 2·N_max + 1).
pub fn solve_lambert(input: &LambertInput) -> Result<Vec<LambertSolution>, LambertError> {
    let mut solutions = keplerian::prussing::solve_prussing(input)?;

    // Apply default feasibility filtering
    let config = FeasibilityConfig::default();
    keplerian::feasibility::filter_feasibility(
        &mut solutions,
        &input.r1,
        &input.r2,
        input.mu,
        &config,
    );

    Ok(solutions)
}

/// Solve with custom feasibility configuration.
pub fn solve_lambert_with_config(
    input: &LambertInput,
    config: &FeasibilityConfig,
) -> Result<Vec<LambertSolution>, LambertError> {
    let mut solutions = keplerian::prussing::solve_prussing(input)?;
    keplerian::feasibility::filter_feasibility(
        &mut solutions,
        &input.r1,
        &input.r2,
        input.mu,
        config,
    );
    Ok(solutions)
}

// =========================================================================
// Unified API: Keplerian + optional perturbed refinement
// =========================================================================

/// Solve the Lambert problem with a full force model.
///
/// 1. Computes all feasible Keplerian multi-rev solutions.
/// 2. For each solution, selects the appropriate perturbed solver
///    (TPBVP, KS-TPBVP, or MPS-IVP) based on transfer angle and N.
/// 3. Refines under the given force model.
///
/// Returns perturbed solutions sorted by ascending delta-v
/// (‖v1 − v1_dep‖ + ‖v2_arr − v2‖ is not available without departure/
/// arrival orbits, so results are sorted by ‖v1‖ as a proxy).
pub fn solve_perturbed(
    r1: &Vector3<f64>,
    r2: &Vector3<f64>,
    tof: f64,
    mu: f64,
    force_model: &dyn ForceModel,
    config: &UnifiedLambertConfig,
) -> Result<Vec<PerturbedSolution>, LambertError> {
    // Step 1: Keplerian solutions
    let kep_input = LambertInput {
        r1: *r1,
        r2: *r2,
        tof,
        mu,
        direction: config.direction,
        max_revs: config.max_revs,
    };
    let mut kep_solutions = keplerian::prussing::solve_prussing(&kep_input)?;
    keplerian::feasibility::filter_feasibility(
        &mut kep_solutions,
        r1,
        r2,
        mu,
        &config.feasibility,
    );

    if kep_solutions.is_empty() {
        return Err(LambertError::NoSolution);
    }

    // Step 2: Refine each Keplerian solution with the appropriate solver
    let pc = &config.perturbed;
    let mut results = Vec::with_capacity(kep_solutions.len());

    for ksol in &kep_solutions {
        let algo = select_algorithm(ksol.transfer_angle, ksol.n_revs);
        let (v1, v2, converged) = refine_solution(
            r1, r2, tof, &ksol.v1, ksol.a, ksol.transfer_angle, ksol.n_revs,
            algo, force_model, pc,
        );

        results.push(PerturbedSolution {
            v1,
            v2,
            a_keplerian: ksol.a,
            n_revs: ksol.n_revs,
            branch: ksol.branch,
            transfer_angle: ksol.transfer_angle,
            algorithm: algo,
            converged,
        });
    }

    // Sort by v1 magnitude as a delta-v proxy
    results.sort_by(|a, b| a.v1.norm().partial_cmp(&b.v1.norm()).unwrap());

    Ok(results)
}

/// Drop-in replacement for the old Bates single-revolution solver.
///
/// Returns the minimum-energy N = 0 Keplerian solution only — no perturbed
/// refinement, no multi-rev.
pub fn solve_lambert_bates_compat(
    r1: &Vector3<f64>,
    r2: &Vector3<f64>,
    tof: f64,
    mu: f64,
    direction: types::Direction,
) -> Result<(Vector3<f64>, Vector3<f64>), LambertError> {
    let input = LambertInput {
        r1: *r1,
        r2: *r2,
        tof,
        mu,
        direction,
        max_revs: Some(0),
    };
    let solutions = solve_lambert(&input)?;
    let best = solutions
        .into_iter()
        .min_by(|a, b| a.v1.norm().partial_cmp(&b.v1.norm()).unwrap())
        .ok_or(LambertError::NoSolution)?;
    Ok((best.v1, best.v2))
}

// =========================================================================
// Internal: dispatch to the correct perturbed solver
// =========================================================================

fn refine_solution(
    r1: &Vector3<f64>,
    r2: &Vector3<f64>,
    tof: f64,
    v0_kep: &Vector3<f64>,
    a_kep: f64,
    _theta: f64,
    _n_revs: u32,
    algo: PerturbedAlgorithm,
    force_model: &dyn ForceModel,
    pc: &PerturbedConfig,
) -> (Vector3<f64>, Vector3<f64>, bool) {
    match algo {
        PerturbedAlgorithm::McpiTpbvp => {
            let cfg = perturbed::tpbvp::TpbvpConfig {
                poly_degree: pc.poly_degree,
                max_iterations: pc.max_iterations,
                tolerance: pc.mcpi_tolerance,
            };
            let res = perturbed::tpbvp::solve_tpbvp(r1, r2, 0.0, tof, v0_kep, force_model, &cfg);
            (res.v1, res.v2, res.converged)
        }
        PerturbedAlgorithm::McpiKsTpbvp => {
            let cfg = perturbed::ks_tpbvp::KsTpbvpConfig {
                poly_degree: pc.poly_degree,
                max_iterations: pc.max_iterations,
                tolerance: pc.mcpi_tolerance,
            };
            let res = perturbed::ks_tpbvp::solve_ks_tpbvp(
                r1, r2, 0.0, tof, v0_kep, a_kep, force_model, &cfg,
            );
            (res.v1, res.v2, res.converged)
        }
        PerturbedAlgorithm::McpiMpsIvp => {
            let cfg = perturbed::mps_ivp::MpsIvpConfig {
                poly_degree: pc.poly_degree,
                max_mcpi_iterations: pc.max_iterations,
                mcpi_tolerance: pc.mcpi_tolerance,
                max_mps_iterations: pc.max_mps_iterations,
                mps_tolerance: pc.mps_tolerance,
                perturbation_scale: pc.perturbation_scale,
                variable_fidelity: pc.variable_fidelity,
            };
            let res = perturbed::mps_ivp::solve_mps_ivp(
                r1, r2, 0.0, tof, v0_kep, force_model, &cfg,
            );
            (res.v1, res.v2, res.converged)
        }
    }
}
