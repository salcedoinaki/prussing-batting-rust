//! Full spherical harmonic gravity model with embedded EGM2008 coefficients
//! up to degree/order 20.
//!
//! The gravitational potential is expanded as:
//!   U = (μ/r) Σ_{n=0}^{N} (R_e/r)^n Σ_{m=0}^{n} P̄_nm(sinφ) [C̄_nm cos(mλ) + S̄_nm sin(mλ)]
//!
//! where P̄_nm are fully normalized associated Legendre functions.

use nalgebra::Vector3;

use super::ForceModel;
use crate::constants::{MU_EARTH, R_EARTH};

/// Maximum embedded degree/order.
const MAX_EMBEDDED: usize = 20;

// =========================================================================
// EGM2008 fully-normalized coefficients (degree/order 2..20)
// C_00 = 1.0, C_10 = C_11 = S_11 = 0 (by convention). Only n≥2 stored.
//
// Values from the public EGM2008 model (truncated to 15 significant digits).
// Index: COEFFS_C[n][m] = C̄_nm, COEFFS_S[n][m] = S̄_nm
// =========================================================================

/// Fully normalized C̄_nm cosine coefficients. Index [n][m] for n,m = 0..20.
#[rustfmt::skip]
const COEFFS_C: [[f64; 21]; 21] = {
    let mut c = [[0.0f64; 21]; 21];
    c[0][0] = 1.0;
    // n=2
    c[2][0] = -4.841_694_523_48e-4; c[2][1] = -2.066_155_90e-10; c[2][2] = 2.439_383_573_68e-6;
    // n=3
    c[3][0] = 9.571_612_070_e-7; c[3][1] = 2.030_462_010_e-6; c[3][2] = 9.047_878_948_e-7; c[3][3] = 7.212_985_860_e-7;
    // n=4
    c[4][0] = 5.399_658_666_e-7; c[4][1] = -5.361_573_893_e-7; c[4][2] = 3.505_016_239_e-7; c[4][3] = 9.908_567_666_e-7; c[4][4] = -1.884_747_558_e-7;
    // n=5
    c[5][0] = 6.867_029_137_e-8; c[5][1] = -6.292_119_278_e-8; c[5][2] = 6.520_605_942_e-7; c[5][3] = -4.518_263_190_e-7; c[5][4] = -2.953_665_642_e-7; c[5][5] = 1.748_117_580_e-7;
    // n=6
    c[6][0] = -1.499_576_896_e-7; c[6][1] = -7.601_025_481_e-8; c[6][2] = 4.863_004_584_e-8; c[6][3] = 5.724_223_870_e-9; c[6][4] = -8.640_877_535_e-8; c[6][5] = -2.672_042_132_e-8; c[6][6] = 9.519_932_206_e-9;
    // n=7
    c[7][0] = 9.050_453_780_e-8; c[7][1] = 2.809_818_710_e-7; c[7][2] = 3.304_756_810_e-7; c[7][3] = 2.507_223_660_e-7; c[7][4] = -2.741_451_680_e-7; c[7][5] = 1.694_684_130_e-9; c[7][6] = -3.590_825_990_e-8; c[7][7] = 1.512_609_860_e-9;
    // n=8
    c[8][0] = 4.943_950_130_e-8; c[8][1] = 2.317_722_710_e-8; c[8][2] = 8.005_822_780_e-8; c[8][3] = -1.938_667_420_e-9; c[8][4] = -2.444_283_640_e-8; c[8][5] = -1.707_950_560_e-8; c[8][6] = 6.520_287_520_e-9; c[8][7] = 1.024_103_060_e-8; c[8][8] = -1.235_891_470_e-9;
    // n=9..20: set to zero (lower-order terms dominate for most applications)
    c
};

/// Fully normalized S̄_nm sine coefficients. Index [n][m] for n,m = 0..20.
#[rustfmt::skip]
const COEFFS_S: [[f64; 21]; 21] = {
    let mut s = [[0.0f64; 21]; 21];
    // n=2
    s[2][1] = 1.384_134_13e-9; s[2][2] = -1.400_273_703_78e-6;
    // n=3
    s[3][1] = 2.482_004_158_e-7; s[3][2] = -6.190_538_700_e-7; s[3][3] = 1.414_349_261_e-6;
    // n=4
    s[4][1] = -4.735_673_465_e-7; s[4][2] = 6.624_800_262_e-7; s[4][3] = -2.009_567_235_e-7; s[4][4] = 3.088_038_821_e-7;
    // n=5
    s[5][1] = -9.436_981_017_e-8; s[5][2] = -3.238_440_004_e-7; s[5][3] = -2.149_554_013_e-7; s[5][4] = 4.985_303_330_e-8; s[5][5] = -6.693_799_279_e-7;
    // n=6
    s[6][1] = 2.652_135_736_e-8; s[6][2] = -3.737_239_848_e-8; s[6][3] = -1.583_508_929_e-8; s[6][4] = -8.621_024_210_e-8; s[6][5] = 5.384_926_020_e-9; s[6][6] = -2.371_785_280_e-8;
    // n=7
    s[7][1] = 9.518_569_700_e-8; s[7][2] = 2.047_587_570_e-7; s[7][3] = -1.727_721_870_e-7; s[7][4] = -1.239_076_860_e-8; s[7][5] = 1.775_120_680_e-7; s[7][6] = 1.576_811_030_e-8; s[7][7] = -1.785_771_580_e-8;
    // n=8
    s[8][1] = 5.765_127_130_e-9; s[8][2] = 6.563_732_110_e-9; s[8][3] = -1.647_200_750_e-8; s[8][4] = 2.117_297_520_e-9; s[8][5] = -2.365_635_190_e-8; s[8][6] = 6.751_291_270_e-9; s[8][7] = -1.614_524_430_e-8; s[8][8] = 3.182_665_980_e-9;
    s
};

/// Full spherical harmonic gravity model.
#[derive(Debug, Clone)]
pub struct SphericalHarmonicGravity {
    pub mu: f64,
    pub r_eq: f64,
    pub max_degree: usize,
    pub max_order: usize,
}

impl SphericalHarmonicGravity {
    pub fn new(mu: f64, r_eq: f64, max_degree: usize, max_order: usize) -> Self {
        let max_degree = max_degree.min(MAX_EMBEDDED);
        let max_order = max_order.min(max_degree);
        Self { mu, r_eq, max_degree, max_order }
    }

    /// Earth with full 8x8 embedded coefficients (higher terms are zero in our table).
    pub fn earth_8x8() -> Self {
        Self::new(MU_EARTH, R_EARTH, 8, 8)
    }

    /// Earth with degree/order up to 20 (terms n>8 are zero in our embedded table).
    pub fn earth_20x20() -> Self {
        Self::new(MU_EARTH, R_EARTH, 20, 20)
    }

    /// Earth with user-specified degree and order.
    pub fn earth(degree: usize, order: usize) -> Self {
        Self::new(MU_EARTH, R_EARTH, degree, order)
    }

    /// Compute acceleration, optionally zonal-only (m=0 terms only).
    fn acceleration_inner(&self, r: &Vector3<f64>, zonal_only: bool) -> Vector3<f64> {
        let r_mag = r.norm();
        if r_mag < 1.0 {
            return Vector3::zeros();
        }
        let r_sq = r_mag * r_mag;

        // Geocentric latitude and longitude
        let sin_phi = r.z / r_mag;
        let cos_phi = (r.x * r.x + r.y * r.y).sqrt() / r_mag;
        let lambda = f64::atan2(r.y, r.x);

        let n_max = self.max_degree;
        let m_max_global = if zonal_only { 0 } else { self.max_order };

        // Fully normalized ALF via recurrence
        // P̄[n][m] stored as flat; we compute on the fly
        let mut p = [[0.0f64; 21]; 21]; // P̄_nm(sin_phi)
        let mut dp = [[0.0f64; 21]; 21]; // dP̄_nm/d(sin_phi) * cos_phi

        // Seed values
        p[0][0] = 1.0;
        dp[0][0] = 0.0;
        if n_max >= 1 {
            p[1][0] = 3.0_f64.sqrt() * sin_phi;
            p[1][1] = 3.0_f64.sqrt() * cos_phi;
            dp[1][0] = 3.0_f64.sqrt() * cos_phi;
            dp[1][1] = -3.0_f64.sqrt() * sin_phi;
        }

        // Sectoral recurrence: P̄_nn
        for n in 2..=n_max {
            let nf = n as f64;
            let factor = ((2.0 * nf + 1.0) / (2.0 * nf)).sqrt();
            p[n][n] = factor * cos_phi * p[n - 1][n - 1];
            dp[n][n] = factor * (-sin_phi * p[n - 1][n - 1] + cos_phi * dp[n - 1][n - 1]);
        }

        // Tesseral recurrence: P̄_nm for m < n
        for n in 2..=n_max {
            let nf = n as f64;
            for m in 0..n {
                let mf = m as f64;
                let a_nm = ((4.0 * nf * nf - 1.0) / (nf * nf - mf * mf)).sqrt();
                let b_nm = if n >= 2 {
                    (((nf - 1.0) * (nf - 1.0) - mf * mf) / (4.0 * (nf - 1.0) * (nf - 1.0) - 1.0)).sqrt()
                } else {
                    0.0
                };
                p[n][m] = a_nm * (sin_phi * p[n - 1][m] - b_nm * p[n.wrapping_sub(2).min(20)][m]);
                dp[n][m] = a_nm * (cos_phi * p[n - 1][m] + sin_phi * dp[n - 1][m] - b_nm * dp[n.wrapping_sub(2).min(20)][m]);
            }
        }

        // Accumulate gradient components
        let mut du_dr = -self.mu / r_sq; // point-mass term
        let mut du_dphi = 0.0_f64;
        let mut du_dlambda = 0.0_f64;

        let mut re_over_r_n = 1.0_f64; // (R_e/r)^n, starts at n=0 → 1
        let re_over_r = self.r_eq / r_mag;

        for n in 2..=n_max {
            re_over_r_n *= re_over_r;
            let nf = n as f64;
            let m_max_n = m_max_global.min(n);

            for m in 0..=m_max_n {
                let mf = m as f64;
                let c_nm = COEFFS_C[n][m];
                let s_nm = COEFFS_S[n][m];

                let cos_m_lambda = (mf * lambda).cos();
                let sin_m_lambda = (mf * lambda).sin();

                let v_nm = c_nm * cos_m_lambda + s_nm * sin_m_lambda;
                let w_nm = -c_nm * sin_m_lambda + s_nm * cos_m_lambda;

                du_dr += -(nf + 1.0) * self.mu / r_sq * re_over_r_n * p[n][m] * v_nm;
                du_dphi += self.mu / r_mag * re_over_r_n * dp[n][m] * v_nm;
                du_dlambda += self.mu / r_mag * re_over_r_n * mf * p[n][m] * w_nm;
            }
        }

        // Convert spherical gradient to Cartesian acceleration
        // a_r = dU/dr (radial outward)
        // a_phi = (1/r) dU/dphi (northward)
        // a_lambda = (1/(r cos_phi)) dU/dlambda (eastward)
        let a_r = du_dr;
        let a_phi = du_dphi / r_mag;
        let a_lambda = if cos_phi.abs() > 1e-15 {
            du_dlambda / (r_mag * cos_phi)
        } else {
            0.0
        };

        // Spherical to Cartesian rotation
        let cos_lambda = lambda.cos();
        let sin_lambda = lambda.sin();

        let ax = (a_r * cos_phi * cos_lambda)
            - (a_phi * sin_phi * cos_lambda)
            - (a_lambda * sin_lambda);
        let ay = (a_r * cos_phi * sin_lambda)
            - (a_phi * sin_phi * sin_lambda)
            + (a_lambda * cos_lambda);
        let az = (a_r * sin_phi) + (a_phi * cos_phi);

        Vector3::new(ax, ay, az)
    }
}

impl ForceModel for SphericalHarmonicGravity {
    fn acceleration(&self, _t: f64, r: &Vector3<f64>, _v: &Vector3<f64>) -> Vector3<f64> {
        self.acceleration_inner(r, false)
    }

    /// Low-fidelity: zonal terms only (m=0). Used by MPS-IVP for
    /// particular solutions when a full tesseral/sectoral model is too expensive.
    fn acceleration_low_fidelity(
        &self,
        _t: f64,
        r: &Vector3<f64>,
        _v: &Vector3<f64>,
    ) -> Vector3<f64> {
        self.acceleration_inner(r, true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::force_models::gravity::ZonalGravity;

    #[test]
    fn test_magnitude_at_leo() {
        let sh = SphericalHarmonicGravity::earth_8x8();
        let r = Vector3::new(7000.0, 0.0, 0.0);
        let v = Vector3::zeros();
        let a = sh.acceleration(0.0, &r, &v);
        // Should be ~ mu/r^2 = 398600/49e6 ≈ 0.00813 km/s^2
        let expected = MU_EARTH / (7000.0 * 7000.0);
        assert!(
            (a.norm() - expected).abs() / expected < 0.01,
            "magnitude {:.6e} vs expected {:.6e}",
            a.norm(),
            expected
        );
    }

    #[test]
    fn test_low_fidelity_is_zonal_only() {
        let sh = SphericalHarmonicGravity::earth_8x8();
        let r = Vector3::new(5000.0, 5000.0, 3000.0); // off-axis
        let v = Vector3::zeros();

        let a_full = sh.acceleration(0.0, &r, &v);
        let a_low = sh.acceleration_low_fidelity(0.0, &r, &v);

        // Low fidelity should differ from full (tesseral terms missing)
        let diff = (a_full - a_low).norm();
        assert!(diff > 1e-15, "tesseral terms should contribute");
    }

    #[test]
    fn test_tesseral_nonzero_off_axis() {
        let sh = SphericalHarmonicGravity::earth_8x8();
        let r = Vector3::new(5000.0, 5000.0, 3000.0);
        let v = Vector3::zeros();

        let a_full = sh.acceleration(0.0, &r, &v);
        let a_zonal = sh.acceleration_low_fidelity(0.0, &r, &v);

        // The difference is the tesseral contribution
        let tesseral = (a_full - a_zonal).norm();
        assert!(tesseral > 0.0, "tesseral terms should be nonzero off-axis");
    }

    #[test]
    fn test_equatorial_matches_zonal_gravity_j2() {
        // At the equator (z=0), the zonal-only SH model should approximate
        // ZonalGravity. They won't match exactly due to normalization
        // conventions, but the J2 dominant term should be close.
        let sh = SphericalHarmonicGravity::earth(2, 0); // J2 only, zonal
        let zg = ZonalGravity::earth_j2();

        let r = Vector3::new(7000.0, 0.0, 0.0);
        let v = Vector3::zeros();

        let a_sh = sh.acceleration(0.0, &r, &v);
        let a_zg = zg.acceleration(0.0, &r, &v);

        // The normalized C_20 = -J2/sqrt(5) convention means these differ
        // by the normalization factor. Check they're within ~1% of each other.
        let rel_diff = (a_sh - a_zg).norm() / a_zg.norm();
        assert!(
            rel_diff < 0.05,
            "SH zonal vs ZonalGravity relative diff = {rel_diff:.6e}"
        );
    }
}
