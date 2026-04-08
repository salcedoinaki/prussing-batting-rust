pub mod constants;
pub mod error;
pub mod keplerian;
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
