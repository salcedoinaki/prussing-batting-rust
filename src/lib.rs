//! # Unified Lambert Tool (ULT)
//!
//! Multi-revolution Keplerian Lambert solver based on the Prussing/Ochoa
//! algorithm, with hooks for future MCPI-based perturbed solvers.
//!
//! ## Quick start
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
use types::{FeasibilityConfig, LambertInput, LambertSolution};

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
