pub mod two_body;

use nalgebra::Vector3;

/// Trait for computing accelerations in a perturbed orbital force model.
pub trait ForceModel: Send + Sync {
    /// Compute the acceleration at the given state.
    fn acceleration(&self, t: f64, r: &Vector3<f64>, v: &Vector3<f64>) -> Vector3<f64>;

    /// Lower-fidelity version for use in MPS particular-solution propagation.
    /// Defaults to the full model.
    fn acceleration_low_fidelity(
        &self,
        t: f64,
        r: &Vector3<f64>,
        v: &Vector3<f64>,
    ) -> Vector3<f64> {
        self.acceleration(t, r, v)
    }
}
