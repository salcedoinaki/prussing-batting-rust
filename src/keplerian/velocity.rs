use nalgebra::Vector3;

use crate::types::TransferGeometry;

/// Compute departure and arrival velocity vectors from converged semi-major
/// axis and auxiliary angles, using the Prussing skewed-unit-vector
/// formulation (Eqs. 38–44 from the plan).
///
/// # Arguments
/// * `r1`, `r2` — position vectors
/// * `geom` — pre-computed transfer geometry
/// * `a` — converged semi-major axis
/// * `alpha`, `beta` — auxiliary angles for this branch
/// * `mu` — gravitational parameter
pub fn terminal_velocities(
    r1: &Vector3<f64>,
    r2: &Vector3<f64>,
    geom: &TransferGeometry,
    a: f64,
    alpha: f64,
    beta: f64,
    mu: f64,
) -> (Vector3<f64>, Vector3<f64>) {
    let u1 = r1 / geom.r1_mag; // unit vector along r1
    let u2 = r2 / geom.r2_mag; // unit vector along r2
    let uc = (r2 - r1) / geom.c; // unit vector along chord

    // cot(x/2) = cos(x/2) / sin(x/2)
    let cot_alpha_half = (alpha / 2.0).cos() / (alpha / 2.0).sin();
    let cot_beta_half = (beta / 2.0).cos() / (beta / 2.0).sin();

    let coeff = (mu / (4.0 * a)).sqrt();
    let a_coeff = coeff * cot_alpha_half; // A in the plan
    let b_coeff = coeff * cot_beta_half; // B in the plan

    let v1 = (b_coeff + a_coeff) * &uc + (b_coeff - a_coeff) * &u1;
    let v2 = (b_coeff + a_coeff) * &uc - (b_coeff - a_coeff) * &u2;

    (v1, v2)
}
