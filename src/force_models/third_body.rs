//! Third-body gravitational perturbations from the Sun and Moon.
//!
//! Uses low-precision analytical ephemeris for body positions.
//! The acceleration on a satellite at position r due to a body at r_body is:
//!   a = μ_body · (r_rel/|r_rel|³ − r_body/|r_body|³)
//! where r_rel = r_body − r_sat.

use nalgebra::Vector3;

use super::ForceModel;
use crate::constants::{MU_MOON, MU_SUN};
use crate::force_models::ephemeris;

/// Third-body gravitational perturbation model.
#[derive(Debug, Clone)]
pub struct ThirdBodyGravity {
    pub include_sun: bool,
    pub include_moon: bool,
}

impl ThirdBodyGravity {
    pub fn sun_and_moon() -> Self {
        Self { include_sun: true, include_moon: true }
    }

    pub fn sun_only() -> Self {
        Self { include_sun: true, include_moon: false }
    }

    pub fn moon_only() -> Self {
        Self { include_sun: false, include_moon: true }
    }
}

/// Compute the third-body acceleration on a satellite at `r_sat` due to
/// a body with gravitational parameter `mu_body` at position `r_body`.
fn third_body_accel(mu_body: f64, r_sat: &Vector3<f64>, r_body: &Vector3<f64>) -> Vector3<f64> {
    let r_rel = r_body - r_sat;
    let r_rel_mag = r_rel.norm();
    let r_body_mag = r_body.norm();

    if r_rel_mag < 1.0 || r_body_mag < 1.0 {
        return Vector3::zeros();
    }

    mu_body * (r_rel / (r_rel_mag * r_rel_mag * r_rel_mag)
        - r_body / (r_body_mag * r_body_mag * r_body_mag))
}

impl ForceModel for ThirdBodyGravity {
    fn acceleration(&self, t: f64, r: &Vector3<f64>, _v: &Vector3<f64>) -> Vector3<f64> {
        let mut a = Vector3::zeros();

        if self.include_sun {
            let r_sun = ephemeris::sun_position_eci(t);
            a += third_body_accel(MU_SUN, r, &r_sun);
        }

        if self.include_moon {
            let r_moon = ephemeris::moon_position_eci(t);
            a += third_body_accel(MU_MOON, r, &r_moon);
        }

        a
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sun_perturbation_magnitude() {
        let model = ThirdBodyGravity::sun_only();
        let r = Vector3::new(7000.0, 0.0, 0.0);
        let v = Vector3::zeros();
        let a = model.acceleration(0.0, &r, &v);
        // Sun third-body tidal acceleration at LEO: ~1e-10 to 1e-6 km/s²
        // (depends on satellite-Sun geometry)
        let mag = a.norm();
        assert!(
            mag > 1e-12 && mag < 1e-4,
            "Sun perturbation magnitude {mag:.6e} km/s²"
        );
    }

    #[test]
    fn test_moon_perturbation_magnitude() {
        let model = ThirdBodyGravity::moon_only();
        let r = Vector3::new(7000.0, 0.0, 0.0);
        let v = Vector3::zeros();
        let a = model.acceleration(0.0, &r, &v);
        // Moon third-body at LEO: ~1e-8 to 1e-6 km/s²
        let mag = a.norm();
        assert!(
            mag > 1e-10 && mag < 1e-4,
            "Moon perturbation magnitude {mag:.6e} km/s²"
        );
    }

    #[test]
    fn test_sun_and_moon_equals_sum() {
        let r = Vector3::new(7000.0, 0.0, 0.0);
        let v = Vector3::zeros();
        let t = 86400.0;

        let a_both = ThirdBodyGravity::sun_and_moon().acceleration(t, &r, &v);
        let a_sun = ThirdBodyGravity::sun_only().acceleration(t, &r, &v);
        let a_moon = ThirdBodyGravity::moon_only().acceleration(t, &r, &v);

        let diff = (a_both - (a_sun + a_moon)).norm();
        assert!(diff < 1e-20, "sun+moon should equal combined: diff = {diff:.6e}");
    }

    #[test]
    fn test_perturbation_changes_with_time() {
        let model = ThirdBodyGravity::sun_and_moon();
        let r = Vector3::new(7000.0, 0.0, 0.0);
        let v = Vector3::zeros();

        let a0 = model.acceleration(0.0, &r, &v);
        let a1 = model.acceleration(86400.0 * 30.0, &r, &v);
        let diff = (a1 - a0).norm();
        assert!(diff > 1e-15, "perturbation should change over 30 days");
    }
}
