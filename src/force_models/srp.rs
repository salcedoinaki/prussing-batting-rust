//! Solar radiation pressure (SRP) force model.
//!
//! The SRP acceleration on a spacecraft is:
//!   a_srp = −P_sr · Cr · (A/m) · (AU/|r_sun|)² · r̂_sun→sat
//!
//! With optional cylindrical Earth shadow model.

use nalgebra::Vector3;

use super::ForceModel;
use crate::constants::{AU_KM, P_SR, R_EARTH};
use crate::force_models::ephemeris;

/// Solar radiation pressure model.
#[derive(Debug, Clone)]
pub struct SolarRadiationPressure {
    /// Reflectivity coefficient (1.0 = absorb, 2.0 = perfect reflect).
    pub cr: f64,
    /// Area-to-mass ratio (m²/kg).
    pub area_to_mass: f64,
    /// Enable cylindrical Earth shadow model.
    pub shadow_model: bool,
}

impl SolarRadiationPressure {
    pub fn new(cr: f64, area_to_mass: f64, shadow_model: bool) -> Self {
        Self { cr, area_to_mass, shadow_model }
    }

    /// Default spacecraft: Cr=1.2, A/m=0.02 m²/kg, shadow enabled.
    pub fn default_spacecraft() -> Self {
        Self::new(1.2, 0.02, true)
    }
}

/// Check if the satellite is in Earth's cylindrical shadow.
///
/// The satellite is in shadow when it is on the anti-Sun side of Earth
/// AND its perpendicular distance from the Sun-Earth line is less than R_EARTH.
fn in_earth_shadow(r_sat: &Vector3<f64>, r_sun: &Vector3<f64>) -> bool {
    let r_sun_mag = r_sun.norm();
    if r_sun_mag < 1.0 {
        return false;
    }
    let sun_hat = r_sun / r_sun_mag;

    // Project satellite position onto the Sun direction
    let proj = r_sat.dot(&sun_hat);

    // Satellite must be on the anti-Sun side (behind Earth)
    if proj >= 0.0 {
        return false;
    }

    // Perpendicular distance from the Earth-Sun line
    let perp = (r_sat - proj * sun_hat).norm();

    perp < R_EARTH
}

impl ForceModel for SolarRadiationPressure {
    fn acceleration(&self, t: f64, r: &Vector3<f64>, _v: &Vector3<f64>) -> Vector3<f64> {
        let r_sun = ephemeris::sun_position_eci(t);

        // Check shadow
        if self.shadow_model && in_earth_shadow(r, &r_sun) {
            return Vector3::zeros();
        }

        // Vector from Sun to satellite
        let r_sun_to_sat = r - &r_sun;
        let dist = r_sun_to_sat.norm();
        if dist < 1.0 {
            return Vector3::zeros();
        }

        // SRP acceleration: away from Sun
        // P_SR [N/m²] * Cr * (A/m) [m²/kg] * (AU/dist)² → [N/kg] = [m/s²]
        // Convert to km/s²: multiply by 1e-3
        let au_ratio_sq = (AU_KM / dist) * (AU_KM / dist);
        let mag = P_SR * self.cr * self.area_to_mass * au_ratio_sq * 1e-3;

        mag * r_sun_to_sat / dist
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_srp_magnitude() {
        let srp = SolarRadiationPressure::new(1.2, 0.02, false);
        let r = Vector3::new(7000.0, 0.0, 0.0);
        let v = Vector3::zeros();
        let a = srp.acceleration(0.0, &r, &v);
        // SRP at 1 AU: ~1e-8 to 1e-7 km/s²
        let mag = a.norm();
        assert!(
            mag > 1e-11 && mag < 1e-5,
            "SRP magnitude = {mag:.6e} km/s²"
        );
    }

    #[test]
    fn test_srp_direction_away_from_sun() {
        let srp = SolarRadiationPressure::new(1.2, 0.02, false);
        let r = Vector3::new(7000.0, 0.0, 0.0);
        let v = Vector3::zeros();

        let r_sun = ephemeris::sun_position_eci(0.0);
        let a = srp.acceleration(0.0, &r, &v);

        // SRP pushes away from Sun: dot(a, r_sat - r_sun) > 0
        let away = r - r_sun;
        assert!(a.dot(&away) > 0.0, "SRP should push away from Sun");
    }

    #[test]
    fn test_srp_zero_in_shadow() {
        let srp = SolarRadiationPressure::new(1.2, 0.02, true);

        // Place satellite directly behind Earth (anti-Sun direction)
        let r_sun = ephemeris::sun_position_eci(0.0);
        let sun_hat = r_sun / r_sun.norm();
        let r_sat = -7000.0 * sun_hat; // behind Earth, close to Earth-Sun line

        let v = Vector3::zeros();
        let a = srp.acceleration(0.0, &r_sat, &v);
        assert!(
            a.norm() < 1e-30,
            "SRP should be zero in Earth shadow"
        );
    }

    #[test]
    fn test_srp_nonzero_shadow_disabled() {
        let srp = SolarRadiationPressure::new(1.2, 0.02, false);

        // Same position as shadow test but shadow disabled
        let r_sun = ephemeris::sun_position_eci(0.0);
        let sun_hat = r_sun / r_sun.norm();
        let r_sat = -7000.0 * sun_hat;

        let v = Vector3::zeros();
        let a = srp.acceleration(0.0, &r_sat, &v);
        assert!(
            a.norm() > 1e-15,
            "SRP should be nonzero with shadow disabled"
        );
    }
}
