//! Low-precision analytical Sun and Moon positions in ECI (Earth-Centered Inertial).
//!
//! Accuracy is ~1° for the Sun and ~2° for the Moon — sufficient for
//! perturbation force evaluation but not for precise ephemeris work.

use nalgebra::Vector3;

use crate::constants::{AU_KM, OBLIQUITY_J2000};

/// Compute the Sun position in ECI coordinates (km).
///
/// Uses the low-precision solar coordinates from the Astronomical Almanac.
/// Input `t` is seconds from J2000.0 (2000 Jan 1.5 TDB).
pub fn sun_position_eci(t: f64) -> Vector3<f64> {
    let t_centuries = t / (86400.0 * 36525.0); // Julian centuries from J2000

    // Mean anomaly of the Sun (radians)
    let m = (357.5291 + 35999.0503 * t_centuries).to_radians();
    // Mean longitude of the Sun (radians)
    let l0 = (280.4664 + 36000.7698 * t_centuries).to_radians();
    // Ecliptic longitude
    let lambda = l0 + (1.9146 * m.sin() + 0.02 * (2.0 * m).sin()).to_radians();
    // Distance in AU (approx)
    let r_au = 1.00014 - 0.01671 * m.cos() - 0.00014 * (2.0 * m).cos();
    let r_km = r_au * AU_KM;

    // Ecliptic to ECI rotation through obliquity
    let eps = OBLIQUITY_J2000;
    let x = r_km * lambda.cos();
    let y = r_km * lambda.sin() * eps.cos();
    let z = r_km * lambda.sin() * eps.sin();

    Vector3::new(x, y, z)
}

/// Compute the Moon position in ECI coordinates (km).
///
/// Uses simplified lunar elements from Meeus (Astronomical Algorithms).
/// Input `t` is seconds from J2000.0 (2000 Jan 1.5 TDB).
pub fn moon_position_eci(t: f64) -> Vector3<f64> {
    let t_centuries = t / (86400.0 * 36525.0);

    // Fundamental arguments (degrees, then convert)
    let l_prime = (218.3165 + 481267.8813 * t_centuries) % 360.0; // mean longitude
    let d = (297.8502 + 445267.1115 * t_centuries) % 360.0; // mean elongation
    let m = (357.5291 + 35999.0503 * t_centuries) % 360.0; // Sun mean anomaly
    let m_prime = (134.9634 + 477198.8676 * t_centuries) % 360.0; // Moon mean anomaly
    let f = (93.2720 + 483202.0175 * t_centuries) % 360.0; // argument of latitude

    let l_prime_r = l_prime.to_radians();
    let d_r = d.to_radians();
    let m_r = m.to_radians();
    let m_prime_r = m_prime.to_radians();
    let f_r = f.to_radians();

    // Ecliptic longitude perturbations (simplified, largest terms)
    let lambda = l_prime_r
        + (6.289 * m_prime_r.sin()).to_radians()
        + (-1.274 * (2.0 * d_r - m_prime_r).sin()).to_radians()
        + (-0.658 * (2.0 * d_r).sin()).to_radians()
        + (0.214 * (2.0 * m_prime_r).sin()).to_radians()
        + (-0.186 * m_r.sin()).to_radians();

    // Ecliptic latitude (simplified)
    let beta = (5.128 * f_r.sin()).to_radians()
        + (0.281 * (m_prime_r + f_r).sin()).to_radians()
        + (-0.278 * (f_r - m_prime_r).sin()).to_radians();

    // Distance (km)
    let r_km = 385001.0
        - 20905.0 * m_prime_r.cos()
        - 3699.0 * (2.0 * d_r - m_prime_r).cos()
        - 2956.0 * (2.0 * d_r).cos();

    // Ecliptic to ECI
    let eps = OBLIQUITY_J2000;
    let x_ecl = r_km * beta.cos() * lambda.cos();
    let y_ecl = r_km * beta.cos() * lambda.sin();
    let z_ecl = r_km * beta.sin();

    let x = x_ecl;
    let y = y_ecl * eps.cos() - z_ecl * eps.sin();
    let z = y_ecl * eps.sin() + z_ecl * eps.cos();

    Vector3::new(x, y, z)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sun_distance_order_of_magnitude() {
        let r = sun_position_eci(0.0);
        let dist = r.norm();
        // Should be ~1 AU = 1.496e8 km
        assert!(dist > 1.4e8 && dist < 1.6e8, "Sun distance = {dist:.3e} km");
    }

    #[test]
    fn test_moon_distance_order_of_magnitude() {
        let r = moon_position_eci(0.0);
        let dist = r.norm();
        // Should be ~384,400 km ± 10%
        assert!(dist > 3.4e5 && dist < 4.1e5, "Moon distance = {dist:.3e} km");
    }

    #[test]
    fn test_sun_position_changes_with_time() {
        let r0 = sun_position_eci(0.0);
        let r1 = sun_position_eci(86400.0 * 30.0); // 30 days later
        let diff = (r1 - r0).norm();
        assert!(diff > 1e6, "Sun should move significantly in 30 days");
    }

    #[test]
    fn test_moon_position_changes_with_time() {
        let r0 = moon_position_eci(0.0);
        let r1 = moon_position_eci(86400.0 * 7.0); // 7 days later
        let diff = (r1 - r0).norm();
        assert!(diff > 1e4, "Moon should move significantly in 7 days");
    }
}
