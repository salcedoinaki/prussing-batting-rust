//! Two-body (Keplerian) gravitational acceleration: a = -mu/r³ * r

use nalgebra::Vector3;

use super::ForceModel;

/// Point-mass two-body gravitational force model.
#[derive(Debug, Clone)]
pub struct TwoBody {
    /// Gravitational parameter mu (km³/s²).
    pub mu: f64,
}

impl TwoBody {
    pub fn new(mu: f64) -> Self {
        Self { mu }
    }

    /// Earth default (WGS-84 mu).
    pub fn earth() -> Self {
        Self::new(crate::constants::MU_EARTH)
    }
}

impl ForceModel for TwoBody {
    fn acceleration(&self, _t: f64, r: &Vector3<f64>, _v: &Vector3<f64>) -> Vector3<f64> {
        let r_mag = r.norm();
        -self.mu / (r_mag * r_mag * r_mag) * r
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_two_body_magnitude() {
        let tb = TwoBody::earth();
        let r = Vector3::new(7000.0, 0.0, 0.0);
        let v = Vector3::zeros();
        let a = tb.acceleration(0.0, &r, &v);

        // Expected: mu / r^2 = 398600.4418 / 7000^2 ≈ 0.00813 km/s²
        let expected_mag = 398600.4418 / (7000.0 * 7000.0);
        let a_mag = a.norm();
        assert!(
            (a_mag - expected_mag).abs() < 1e-10,
            "a = {a_mag}, expected {expected_mag}"
        );

        // Direction should be -x
        assert!(a.x < 0.0);
        assert!(a.y.abs() < 1e-15);
        assert!(a.z.abs() < 1e-15);
    }

    #[test]
    fn test_two_body_inverse_square() {
        let tb = TwoBody::new(1.0);
        let r1 = Vector3::new(1.0, 0.0, 0.0);
        let r2 = Vector3::new(2.0, 0.0, 0.0);
        let v = Vector3::zeros();

        let a1 = tb.acceleration(0.0, &r1, &v).norm();
        let a2 = tb.acceleration(0.0, &r2, &v).norm();

        // a scales as 1/r^2 => a1/a2 = 4
        assert!(
            (a1 / a2 - 4.0).abs() < 1e-14,
            "ratio = {}, expected 4",
            a1 / a2
        );
    }
}
