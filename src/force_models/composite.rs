//! Composite force model that sums accelerations from multiple sub-models.

use nalgebra::Vector3;

use super::ForceModel;

/// A force model that sums the accelerations from a collection of sub-models.
pub struct CompositeForceModel {
    models: Vec<Box<dyn ForceModel>>,
}

impl CompositeForceModel {
    pub fn new() -> Self {
        Self { models: Vec::new() }
    }

    /// Add a sub-model (builder pattern).
    pub fn add(mut self, model: impl ForceModel + 'static) -> Self {
        self.models.push(Box::new(model));
        self
    }
}

impl ForceModel for CompositeForceModel {
    fn acceleration(&self, t: f64, r: &Vector3<f64>, v: &Vector3<f64>) -> Vector3<f64> {
        self.models
            .iter()
            .map(|m| m.acceleration(t, r, v))
            .fold(Vector3::zeros(), |acc, a| acc + a)
    }

    fn acceleration_low_fidelity(
        &self,
        t: f64,
        r: &Vector3<f64>,
        v: &Vector3<f64>,
    ) -> Vector3<f64> {
        self.models
            .iter()
            .map(|m| m.acceleration_low_fidelity(t, r, v))
            .fold(Vector3::zeros(), |acc, a| acc + a)
    }
}

// Send + Sync is automatically satisfied since Box<dyn ForceModel> requires
// ForceModel: Send + Sync.
unsafe impl Send for CompositeForceModel {}
unsafe impl Sync for CompositeForceModel {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::force_models::two_body::TwoBody;

    #[test]
    fn test_empty_composite_returns_zero() {
        let comp = CompositeForceModel::new();
        let r = Vector3::new(7000.0, 0.0, 0.0);
        let v = Vector3::zeros();
        let a = comp.acceleration(0.0, &r, &v);
        assert!(a.norm() < 1e-30);
    }

    #[test]
    fn test_single_model_matches_direct() {
        let mu = 398600.4418;
        let tb = TwoBody::new(mu);
        let comp = CompositeForceModel::new().add(TwoBody::new(mu));

        let r = Vector3::new(7000.0, 0.0, 0.0);
        let v = Vector3::zeros();

        let a_direct = tb.acceleration(0.0, &r, &v);
        let a_comp = comp.acceleration(0.0, &r, &v);
        assert!((a_direct - a_comp).norm() < 1e-15);
    }

    #[test]
    fn test_two_models_produce_sum() {
        let mu = 398600.4418;
        let comp = CompositeForceModel::new()
            .add(TwoBody::new(mu))
            .add(TwoBody::new(mu));

        let r = Vector3::new(7000.0, 0.0, 0.0);
        let v = Vector3::zeros();

        let a_single = TwoBody::new(mu).acceleration(0.0, &r, &v);
        let a_comp = comp.acceleration(0.0, &r, &v);
        assert!((a_comp - 2.0 * a_single).norm() < 1e-15);
    }

    #[test]
    fn test_low_fidelity_delegates() {
        let comp = CompositeForceModel::new().add(TwoBody::new(398600.4418));
        let r = Vector3::new(7000.0, 0.0, 0.0);
        let v = Vector3::zeros();

        let a_full = comp.acceleration(0.0, &r, &v);
        let a_low = comp.acceleration_low_fidelity(0.0, &r, &v);
        // TwoBody default low_fidelity == full
        assert!((a_full - a_low).norm() < 1e-15);
    }
}
