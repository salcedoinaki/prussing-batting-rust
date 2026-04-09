/// Earth gravitational parameter (km³/s²) — WGS-84
pub const MU_EARTH: f64 = 398600.4418;

/// Earth mean equatorial radius (km) — WGS-84
pub const R_EARTH: f64 = 6378.137;

/// Two-pi
pub const TWO_PI: f64 = 2.0 * std::f64::consts::PI;

/// Newton iteration default tolerance (seconds)
pub const NEWTON_TOL: f64 = 1.0e-12;

/// Maximum Newton iterations
pub const NEWTON_MAX_ITER: usize = 50;

/// Earth rotation rate (rad/s) — WGS-84
pub const OMEGA_EARTH: f64 = 7.292_115_146_706_979e-5;

/// Sun gravitational parameter (km³/s²)
pub const MU_SUN: f64 = 1.327_124_400_18e11;

/// Astronomical unit (km)
pub const AU_KM: f64 = 1.496_0e8;

/// Moon gravitational parameter (km³/s²)
pub const MU_MOON: f64 = 4902.799;

/// Solar radiation pressure at 1 AU (N/m²)
pub const P_SR: f64 = 4.56e-6;

/// Obliquity of the ecliptic (radians) — J2000
pub const OBLIQUITY_J2000: f64 = 0.409_092_804_222_329;
