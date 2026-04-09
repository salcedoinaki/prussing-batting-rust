//! Zonal harmonics gravity model: point-mass two-body plus J_n perturbations.
//!
//! The gravitational acceleration includes:
//!   a = -mu/r³ r + Σ_{n=2}^{N} ΔJ_n(r)
//!
//! where the J_n perturbation terms are derived from the zonal geopotential.

use nalgebra::Vector3;

use super::ForceModel;

// -------------------------------------------------------------------------
// Standard Earth zonal coefficients (EGM96 / WGS-84)
// -------------------------------------------------------------------------
pub const J2: f64 = 1.082_626_68e-3;
pub const J3: f64 = -2.532_656_48e-6;
pub const J4: f64 = -1.619_621_59e-6;
pub const J5: f64 = -2.272_960_82e-7;
pub const J6: f64 = 5.406_812_40e-7;

/// Zonal harmonics gravity model.
///
/// Computes the full gravitational acceleration as point-mass (−μ/r³ r)
/// plus zonal harmonic perturbations J2 through J_N.
#[derive(Debug, Clone)]
pub struct ZonalGravity {
    pub mu: f64,
    pub r_eq: f64,
    /// J coefficients in order: `j_coeffs[0]` = J2, `[1]` = J3, etc.
    j_coeffs: Vec<f64>,
}

impl ZonalGravity {
    pub fn new(mu: f64, r_eq: f64, j_coeffs: Vec<f64>) -> Self {
        Self { mu, r_eq, j_coeffs }
    }

    /// Earth with J2 only.
    pub fn earth_j2() -> Self {
        Self::new(
            crate::constants::MU_EARTH,
            crate::constants::R_EARTH,
            vec![J2],
        )
    }

    /// Earth with J2 through J6.
    pub fn earth_j2_j6() -> Self {
        Self::new(
            crate::constants::MU_EARTH,
            crate::constants::R_EARTH,
            vec![J2, J3, J4, J5, J6],
        )
    }
}

impl ForceModel for ZonalGravity {
    fn acceleration(&self, _t: f64, r: &Vector3<f64>, _v: &Vector3<f64>) -> Vector3<f64> {
        let r_mag = r.norm();
        let r_sq = r_mag * r_mag;

        // Point-mass two-body acceleration
        let mut a = -self.mu / (r_sq * r_mag) * r;

        if self.j_coeffs.is_empty() {
            return a;
        }

        let u = r.z / r_mag; // sin(geocentric latitude)

        // Legendre polynomials and their derivatives via recurrence:
        //   P_n  = ((2n-1) u P_{n-1} - (n-1) P_{n-2}) / n
        //   P_n' = ((2n-1) (P_{n-1} + u P_{n-1}') - (n-1) P_{n-2}') / n
        let mut p_prev2: f64 = 1.0; // P_0
        let mut p_prev1: f64 = u; // P_1
        let mut dp_prev2: f64 = 0.0; // P_0'
        let mut dp_prev1: f64 = 1.0; // P_1'

        // (R_e / r)^n — starts at n=2
        let re_over_r = self.r_eq / r_mag;
        let mut re_over_r_n = re_over_r * re_over_r; // (R_e/r)^2

        for (i, &j_n) in self.j_coeffs.iter().enumerate() {
            let n = (i + 2) as f64;

            // Legendre recurrence for P_n, P_n'
            let p_n = ((2.0 * n - 1.0) * u * p_prev1 - (n - 1.0) * p_prev2) / n;
            let dp_n =
                ((2.0 * n - 1.0) * (p_prev1 + u * dp_prev1) - (n - 1.0) * dp_prev2) / n;

            // Common factor: mu * J_n * (R_e/r)^n / r²
            let lambda = self.mu * j_n * re_over_r_n / r_sq;

            // x, y components: lambda * (coord/r) * [(n+1)*P_n + u*P_n']
            let q_xy = (n + 1.0) * p_n + u * dp_n;
            a.x += lambda * (r.x / r_mag) * q_xy;
            a.y += lambda * (r.y / r_mag) * q_xy;

            // z component: lambda * [(n+1)*u*P_n - (1-u²)*P_n']
            let q_z = (n + 1.0) * u * p_n - (1.0 - u * u) * dp_n;
            a.z += lambda * q_z;

            // Advance recurrence
            p_prev2 = p_prev1;
            p_prev1 = p_n;
            dp_prev2 = dp_prev1;
            dp_prev1 = dp_n;
            re_over_r_n *= re_over_r;
        }

        a
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::force_models::two_body::TwoBody;

    /// With no J coefficients, ZonalGravity should match TwoBody exactly.
    #[test]
    fn test_zonal_no_jn_matches_two_body() {
        let mu = crate::constants::MU_EARTH;
        let zonal = ZonalGravity::new(mu, crate::constants::R_EARTH, vec![]);
        let tb = TwoBody::new(mu);

        let r = Vector3::new(5000.0, 10000.0, 2100.0);
        let v = Vector3::zeros();

        let a_zonal = zonal.acceleration(0.0, &r, &v);
        let a_tb = tb.acceleration(0.0, &r, &v);

        assert!((a_zonal - a_tb).norm() < 1e-15);
    }

    /// Verify J2 perturbation against known analytical formula for equatorial point.
    #[test]
    fn test_j2_equatorial() {
        let mu = crate::constants::MU_EARTH;
        let re = crate::constants::R_EARTH;
        let zonal = ZonalGravity::earth_j2();

        // Equatorial satellite: r = [R, 0, 0] where R = 7000 km
        let r_val = 7000.0;
        let r = Vector3::new(r_val, 0.0, 0.0);
        let v = Vector3::zeros();

        let a = zonal.acceleration(0.0, &r, &v);

        // At equator, z=0, so u=0, the J2 perturbation in x is:
        //   Δa_x = (3/2) * mu * J2 * R_e² / r^5 * x * (5*0 - 1)
        //        = -(3/2) * mu * J2 * R_e² / r^4
        let two_body_x = -mu / (r_val * r_val);
        let j2_x = -(3.0 / 2.0) * mu * J2 * re * re / r_val.powi(4);
        let expected_x = two_body_x + j2_x;

        assert!(
            (a.x - expected_x).abs() < 1e-15,
            "a.x = {:.15e}, expected {:.15e}",
            a.x,
            expected_x
        );
        assert!(a.y.abs() < 1e-15, "a.y should be 0 at equator");
        assert!(a.z.abs() < 1e-15, "a.z should be 0 at equator");
    }

    /// Verify J2 perturbation against known analytical formula for polar point.
    #[test]
    fn test_j2_polar() {
        let mu = crate::constants::MU_EARTH;
        let re = crate::constants::R_EARTH;
        let zonal = ZonalGravity::earth_j2();

        // Polar satellite: r = [0, 0, R]
        let r_val = 7000.0;
        let r = Vector3::new(0.0, 0.0, r_val);
        let v = Vector3::zeros();

        let a = zonal.acceleration(0.0, &r, &v);

        // At pole, u=1. J2 z-perturbation:
        //   Δa_z = -(3/2) * mu * J2 * R_e² * z / r^5 * (3 - 5)
        //        =  (3) * mu * J2 * R_e² / r^4
        let two_body_z = -mu / (r_val * r_val);
        let j2_z = 3.0 * mu * J2 * re * re / r_val.powi(4);
        let expected_z = two_body_z + j2_z;

        assert!(
            (a.z - expected_z).abs() < 1e-14,
            "a.z = {:.15e}, expected {:.15e}",
            a.z,
            expected_z
        );
        assert!(a.x.abs() < 1e-15, "a.x should be 0 at pole");
        assert!(a.y.abs() < 1e-15, "a.y should be 0 at pole");
    }

    /// J2 at equator should increase the inward acceleration magnitude
    /// (Earth is oblate → equator is farther from center → extra mass pull).
    #[test]
    fn test_j2_increases_equatorial_gravity() {
        let mu = crate::constants::MU_EARTH;
        let zonal = ZonalGravity::earth_j2();
        let tb = TwoBody::new(mu);

        let r = Vector3::new(7000.0, 0.0, 0.0);
        let v = Vector3::zeros();

        let a_j2 = zonal.acceleration(0.0, &r, &v).norm();
        let a_tb = tb.acceleration(0.0, &r, &v).norm();

        assert!(
            a_j2 > a_tb,
            "J2 should strengthen equatorial gravity: {a_j2} > {a_tb}"
        );
    }
}
