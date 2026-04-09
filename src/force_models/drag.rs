//! Atmospheric drag model using an exponential density profile.
//!
//! The drag acceleration is:
//!   a_drag = -½ · Cd · (A/m) · ρ · |v_rel|² · v̂_rel
//!
//! where v_rel = v - ω_earth × r accounts for atmospheric co-rotation.

use nalgebra::Vector3;

use super::ForceModel;
use crate::constants::{OMEGA_EARTH, R_EARTH};

/// Exponential atmosphere density table: (base altitude km, density kg/m³, scale height km).
/// Based on the U.S. Standard Atmosphere 1976 (simplified).
const ATMOSPHERE_TABLE: [(f64, f64, f64); 28] = [
    (0.0, 1.225, 7.249),
    (25.0, 3.899e-2, 6.349),
    (30.0, 1.774e-2, 6.682),
    (40.0, 3.972e-3, 7.554),
    (50.0, 1.057e-3, 8.382),
    (60.0, 3.206e-4, 7.714),
    (70.0, 8.770e-5, 6.549),
    (80.0, 1.905e-5, 5.799),
    (90.0, 3.396e-6, 5.382),
    (100.0, 5.297e-7, 5.877),
    (110.0, 9.661e-8, 7.263),
    (120.0, 2.438e-8, 9.473),
    (130.0, 8.484e-9, 12.636),
    (140.0, 3.845e-9, 16.149),
    (150.0, 2.070e-9, 22.523),
    (180.0, 5.464e-10, 29.740),
    (200.0, 2.789e-10, 37.105),
    (250.0, 7.248e-11, 45.546),
    (300.0, 2.418e-11, 53.628),
    (350.0, 9.518e-12, 53.298),
    (400.0, 3.725e-12, 58.515),
    (450.0, 1.585e-12, 60.828),
    (500.0, 6.967e-13, 63.822),
    (600.0, 1.454e-13, 71.835),
    (700.0, 3.614e-14, 88.667),
    (800.0, 1.170e-14, 124.64),
    (900.0, 5.245e-15, 181.05),
    (1000.0, 3.019e-15, 268.00),
];

/// Maximum altitude (km) above which drag is assumed negligible.
const MAX_DRAG_ALTITUDE: f64 = 1000.0;

/// Compute atmospheric density (kg/m³) at the given altitude (km)
/// using piecewise exponential interpolation.
fn exponential_density(altitude_km: f64) -> f64 {
    if altitude_km < 0.0 || altitude_km > MAX_DRAG_ALTITUDE {
        return 0.0;
    }

    // Find the appropriate layer
    let mut base_idx = 0;
    for (i, &(base_alt, _, _)) in ATMOSPHERE_TABLE.iter().enumerate() {
        if altitude_km >= base_alt {
            base_idx = i;
        } else {
            break;
        }
    }

    let (base_alt, base_rho, scale_height) = ATMOSPHERE_TABLE[base_idx];
    base_rho * (-(altitude_km - base_alt) / scale_height).exp()
}

/// Atmospheric drag force model.
#[derive(Debug, Clone)]
pub struct AtmosphericDrag {
    /// Drag coefficient (dimensionless, typically 2.0-2.5).
    pub cd: f64,
    /// Ballistic coefficient: area-to-mass ratio (m²/kg).
    pub area_to_mass: f64,
    /// Equatorial radius for altitude computation (km).
    pub r_eq: f64,
}

impl AtmosphericDrag {
    pub fn new(cd: f64, area_to_mass: f64) -> Self {
        Self {
            cd,
            area_to_mass,
            r_eq: R_EARTH,
        }
    }

    /// Default LEO satellite: Cd=2.2, A/m=0.01 m²/kg.
    pub fn default_leo() -> Self {
        Self::new(2.2, 0.01)
    }
}

impl ForceModel for AtmosphericDrag {
    fn acceleration(&self, _t: f64, r: &Vector3<f64>, v: &Vector3<f64>) -> Vector3<f64> {
        let r_mag = r.norm();
        let altitude = r_mag - self.r_eq;

        if altitude > MAX_DRAG_ALTITUDE || altitude < 0.0 {
            return Vector3::zeros();
        }

        let rho = exponential_density(altitude); // kg/m³
        if rho < 1e-30 {
            return Vector3::zeros();
        }

        // Atmospheric co-rotation velocity (km/s)
        let omega = Vector3::new(0.0, 0.0, OMEGA_EARTH);
        let v_atm = omega.cross(r);
        let v_rel = v - v_atm;
        let v_rel_mag = v_rel.norm();

        if v_rel_mag < 1e-15 {
            return Vector3::zeros();
        }

        // a_drag = -0.5 * Cd * (A/m) * rho * |v_rel|^2 * v_hat_rel
        // Units: rho [kg/m³] * v² [km²/s²] * (A/m) [m²/kg]
        //      = [km²/(m·s²)] = [1e6 m²/(m·s²)] = [1e6 m/s²]
        //      → multiply by 1e-3 to convert m/s² to km/s² → net factor 1e3
        let drag_mag = 0.5 * self.cd * self.area_to_mass * rho * v_rel_mag * v_rel_mag * 1e3;

        -drag_mag * v_rel / v_rel_mag
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zero_above_max_altitude() {
        let drag = AtmosphericDrag::default_leo();
        let r = Vector3::new(R_EARTH + 1500.0, 0.0, 0.0);
        let v = Vector3::new(0.0, 7.5, 0.0);
        let a = drag.acceleration(0.0, &r, &v);
        assert!(a.norm() < 1e-30, "drag should be zero above 1000 km");
    }

    #[test]
    fn test_increases_at_lower_altitude() {
        let drag = AtmosphericDrag::default_leo();
        let v = Vector3::new(0.0, 7.5, 0.0);

        let r_high = Vector3::new(R_EARTH + 500.0, 0.0, 0.0);
        let r_low = Vector3::new(R_EARTH + 300.0, 0.0, 0.0);

        let a_high = drag.acceleration(0.0, &r_high, &v).norm();
        let a_low = drag.acceleration(0.0, &r_low, &v).norm();

        assert!(
            a_low > a_high,
            "drag at 300km ({a_low:.6e}) should exceed drag at 500km ({a_high:.6e})"
        );
    }

    #[test]
    fn test_opposes_velocity() {
        let drag = AtmosphericDrag::default_leo();
        let r = Vector3::new(R_EARTH + 400.0, 0.0, 0.0);
        let v = Vector3::new(0.0, 7.5, 0.0);

        let a = drag.acceleration(0.0, &r, &v);
        // Drag should oppose velocity direction (negative dot product)
        assert!(a.dot(&v) < 0.0, "drag should oppose velocity");
    }

    #[test]
    fn test_correct_magnitude_at_400km() {
        let drag = AtmosphericDrag::default_leo();
        let r = Vector3::new(R_EARTH + 400.0, 0.0, 0.0);
        let v = Vector3::new(0.0, 7.5, 0.0);

        let a_mag = drag.acceleration(0.0, &r, &v).norm();
        // At 400 km, drag on typical LEO sat is ~1e-9 to 1e-7 km/s²
        assert!(
            a_mag > 1e-12 && a_mag < 1e-5,
            "drag magnitude {a_mag:.6e} km/s² out of expected range"
        );
    }

    #[test]
    fn test_zero_when_velocity_zero() {
        let drag = AtmosphericDrag::default_leo();
        let r = Vector3::new(R_EARTH + 400.0, 0.0, 0.0);
        let v = Vector3::zeros();

        let a = drag.acceleration(0.0, &r, &v);
        // With v=0, v_rel ≈ -v_atm (co-rotation), but magnitude is tiny
        // The drag should be very small
        assert!(a.norm() < 1e-10, "drag should be tiny with zero sat velocity");
    }
}
