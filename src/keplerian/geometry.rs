use nalgebra::Vector3;
use std::f64::consts::PI;

use crate::types::{Direction, TransferGeometry};

/// Compute all transfer-geometry quantities from the two position vectors and
/// the desired transfer direction.
pub fn compute_transfer_geometry(
    r1: &Vector3<f64>,
    r2: &Vector3<f64>,
    direction: Direction,
) -> TransferGeometry {
    let r1_mag = r1.norm();
    let r2_mag = r2.norm();

    let chord_vec = r2 - r1;
    let c = chord_vec.norm();

    let s = (r1_mag + r2_mag + c) / 2.0;

    let cos_theta = r1.dot(r2) / (r1_mag * r2_mag);
    let cross = r1.cross(r2);
    let sin_theta_unsigned = cross.norm() / (r1_mag * r2_mag);

    // Sign convention: prograde means the transfer angle is < pi when the
    // orbit normal (cross product) has a positive z-component, and > pi
    // otherwise.  Retrograde flips.
    let z_sign = if cross.z >= 0.0 { 1.0 } else { -1.0 };
    let dir_sign = match direction {
        Direction::Prograde => 1.0,
        Direction::Retrograde => -1.0,
    };
    let sin_theta = z_sign * dir_sign * sin_theta_unsigned;
    let mut theta = f64::atan2(sin_theta, cos_theta);
    if theta < 0.0 {
        theta += 2.0 * PI;
    }

    let a_min = s / 2.0;

    TransferGeometry {
        r1_mag,
        r2_mag,
        c,
        s,
        theta,
        a_min,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_geometry_coplanar() {
        let r1 = Vector3::new(7000.0, 0.0, 0.0);
        let r2 = Vector3::new(0.0, 7000.0, 0.0);
        let geom = compute_transfer_geometry(&r1, &r2, Direction::Prograde);

        assert_relative_eq!(geom.r1_mag, 7000.0, epsilon = 1e-10);
        assert_relative_eq!(geom.r2_mag, 7000.0, epsilon = 1e-10);
        assert_relative_eq!(geom.theta, PI / 2.0, epsilon = 1e-10);
    }
}
