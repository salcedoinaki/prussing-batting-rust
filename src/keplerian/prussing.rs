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
