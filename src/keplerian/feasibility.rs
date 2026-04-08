use nalgebra::Vector3;

use crate::types::{FeasibilityConfig, LambertSolution};

/// Apply feasibility filters to a set of Lambert solutions **in place**,
/// setting `is_feasible = false` on solutions that fail any check.
pub fn filter_feasibility(
    solutions: &mut [LambertSolution],
    _r1: &Vector3<f64>,
    _r2: &Vector3<f64>,
    _mu: f64,
    _config: &FeasibilityConfig,
) {
    // TODO: implement feasibility checks
    for sol in solutions.iter_mut() {
        sol.is_feasible = true;
    }
}
