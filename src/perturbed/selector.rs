//! Automatic algorithm selector for the perturbed Lambert problem.
//!
//! Routes each Keplerian solution to the most efficient perturbed solver
//! based on the transfer true-anomaly angle.

use std::f64::consts::PI;

/// Which perturbed solver to use for a given transfer arc.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PerturbedAlgorithm {
    /// MCPI-TPBVP — direct boundary value; most efficient, short arcs only.
    McpiTpbvp,
    /// MCPI-KS-TPBVP — regularized TPBVP; medium arcs, eccentricity robust.
    McpiKsTpbvp,
    /// MCPI-MPS-IVP — shooting via particular solutions; multi-revolution.
    McpiMpsIvp,
}

/// Select the best perturbed solver for a transfer with the given
/// true-anomaly angle `theta` (radians) and revolution count `n_revs`.
///
/// Thresholds follow Woollands et al. (2018):
///   - θ < 2π/3          → MCPI-TPBVP   (short arcs)
///   - 2π/3 ≤ θ < 1.8π   → MCPI-KS-TPBVP (medium arcs)
///   - θ ≥ 1.8π or N ≥ 1 → MCPI-MPS-IVP  (multi-revolution)
pub fn select_algorithm(theta: f64, n_revs: u32) -> PerturbedAlgorithm {
    if n_revs >= 1 {
        return PerturbedAlgorithm::McpiMpsIvp;
    }
    if theta < 2.0 * PI / 3.0 {
        PerturbedAlgorithm::McpiTpbvp
    } else if theta < 1.8 * PI {
        PerturbedAlgorithm::McpiKsTpbvp
    } else {
        PerturbedAlgorithm::McpiMpsIvp
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_short_arc() {
        assert_eq!(
            select_algorithm(1.0, 0),
            PerturbedAlgorithm::McpiTpbvp
        );
    }

    #[test]
    fn test_medium_arc() {
        assert_eq!(
            select_algorithm(3.0, 0),
            PerturbedAlgorithm::McpiKsTpbvp
        );
    }

    #[test]
    fn test_long_arc() {
        assert_eq!(
            select_algorithm(5.8, 0),
            PerturbedAlgorithm::McpiMpsIvp
        );
    }

    #[test]
    fn test_multi_rev_always_mps() {
        // Even short θ, if N ≥ 1 we use MPS-IVP
        assert_eq!(
            select_algorithm(0.5, 1),
            PerturbedAlgorithm::McpiMpsIvp
        );
        assert_eq!(
            select_algorithm(3.0, 2),
            PerturbedAlgorithm::McpiMpsIvp
        );
    }

    #[test]
    fn test_boundary_short_medium() {
        let boundary = 2.0 * PI / 3.0;
        // Just below → TPBVP
        assert_eq!(
            select_algorithm(boundary - 0.01, 0),
            PerturbedAlgorithm::McpiTpbvp
        );
        // At boundary → KS-TPBVP
        assert_eq!(
            select_algorithm(boundary, 0),
            PerturbedAlgorithm::McpiKsTpbvp
        );
    }

    #[test]
    fn test_boundary_medium_long() {
        let boundary = 1.8 * PI;
        // Just below → KS-TPBVP
        assert_eq!(
            select_algorithm(boundary - 0.01, 0),
            PerturbedAlgorithm::McpiKsTpbvp
        );
        // At boundary → MPS-IVP
        assert_eq!(
            select_algorithm(boundary, 0),
            PerturbedAlgorithm::McpiMpsIvp
        );
    }
}
