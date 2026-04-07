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
