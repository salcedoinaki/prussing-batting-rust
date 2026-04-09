//! Chebyshev polynomial machinery for MCPI.
//!
//! Provides CGL node generation, Chebyshev polynomial evaluation,
//! Clenshaw summation, forward/inverse DCT-based coefficient transforms,
//! and the analytic Chebyshev integration operator.

use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Chebyshev-Gauss-Lobatto (CGL) nodes
// ---------------------------------------------------------------------------

/// Generate N+1 Chebyshev-Gauss-Lobatto nodes on [-1, 1].
///
/// `tau_k = cos(k * pi / n)` for k = 0, 1, …, n.
/// Returned in *descending* order: tau_0 = 1, …, tau_n = -1.
pub fn cgl_nodes(n: usize) -> Vec<f64> {
    let nf = n as f64;
    (0..=n).map(|k| (k as f64 * PI / nf).cos()).collect()
}

// ---------------------------------------------------------------------------
// Time mapping
// ---------------------------------------------------------------------------

/// Map physical time `t ∈ [t0, tf]` to Chebyshev domain `tau ∈ [-1, 1]`.
#[inline]
pub fn time_to_tau(t: f64, t0: f64, tf: f64) -> f64 {
    (2.0 * t - (t0 + tf)) / (tf - t0)
}

/// Map Chebyshev domain `tau ∈ [-1, 1]` to physical time `t ∈ [t0, tf]`.
#[inline]
pub fn tau_to_time(tau: f64, t0: f64, tf: f64) -> f64 {
    ((tf - t0) * tau + (t0 + tf)) / 2.0
}

// ---------------------------------------------------------------------------
// Chebyshev polynomial evaluation
// ---------------------------------------------------------------------------

/// Evaluate Chebyshev polynomial T_k(tau) via the three-term recurrence.
pub fn chebyshev_t(k: usize, tau: f64) -> f64 {
    match k {
        0 => 1.0,
        1 => tau,
        _ => {
            let mut t_prev2 = 1.0;
            let mut t_prev1 = tau;
            for _ in 2..=k {
                let t_cur = 2.0 * tau * t_prev1 - t_prev2;
                t_prev2 = t_prev1;
                t_prev1 = t_cur;
            }
            t_prev1
        }
    }
}

/// Evaluate all Chebyshev polynomials T_0(tau) through T_m(tau).
/// Returns a vector of length m+1.
pub fn chebyshev_t_all(m: usize, tau: f64) -> Vec<f64> {
    let mut t = Vec::with_capacity(m + 1);
    t.push(1.0);
    if m == 0 {
        return t;
    }
    t.push(tau);
    for k in 2..=m {
        let val = 2.0 * tau * t[k - 1] - t[k - 2];
        t.push(val);
    }
    t
}

// ---------------------------------------------------------------------------
// Clenshaw summation
// ---------------------------------------------------------------------------

/// Evaluate a Chebyshev series `sum_{k=0}^{M} c_k * T_k(tau)` using Clenshaw.
pub fn clenshaw(coeffs: &[f64], tau: f64) -> f64 {
    let m = coeffs.len();
    if m == 0 {
        return 0.0;
    }
    if m == 1 {
        return coeffs[0];
    }

    let mut b_k1 = 0.0; // b_{k+1}
    let mut b_k2 = 0.0; // b_{k+2}

    for k in (1..m).rev() {
        let b_k = 2.0 * tau * b_k1 - b_k2 + coeffs[k];
        b_k2 = b_k1;
        b_k1 = b_k;
    }

    // Final step for k=0
    coeffs[0] + tau * b_k1 - b_k2
}

/// Evaluate a 3-component Chebyshev series using Clenshaw summation.
/// `coeffs[k]` = [c_k_x, c_k_y, c_k_z] for k = 0..M.
pub fn clenshaw_3d(coeffs: &[[f64; 3]], tau: f64) -> [f64; 3] {
    let mut result = [0.0; 3];
    for dim in 0..3 {
        let c: Vec<f64> = coeffs.iter().map(|row| row[dim]).collect();
        result[dim] = clenshaw(&c, tau);
    }
    result
}

// ---------------------------------------------------------------------------
// Chebyshev coefficient computation (forward transform / DCT)
// ---------------------------------------------------------------------------

/// Compute Chebyshev coefficients from function values at CGL nodes.
///
/// Given `f_j = f(tau_j)` at the N+1 CGL nodes (returned by `cgl_nodes`),
/// compute the coefficients `c_k` such that `f(tau) ≈ sum c_k T_k(tau)`.
///
/// Uses the discrete cosine transform formula:
///   c_k = (2/N) * sum_{j=0}^{N}'' f_j * T_k(tau_j)
/// where the double prime halves the first and last terms.
pub fn coefficients_from_nodes(values: &[f64], n: usize) -> Vec<f64> {
    assert!(
        values.len() == n + 1,
        "expected {} values at CGL nodes, got {}",
        n + 1,
        values.len()
    );

    let nf = n as f64;
    let mut coeffs = Vec::with_capacity(n + 1);

    for k in 0..=n {
        let mut sum = 0.0;
        for j in 0..=n {
            let t_kj = (k as f64 * j as f64 * PI / nf).cos(); // T_k(tau_j)
            let mut weight = 1.0;
            if j == 0 || j == n {
                weight = 0.5;
            }
            sum += weight * values[j] * t_kj;
        }
        let mut c_k = 2.0 / nf * sum;
        // The k=0 and k=N coefficients have a factor of 1/2 correction
        if k == 0 || k == n {
            c_k *= 0.5;
        }
        coeffs.push(c_k);
    }
    coeffs
}

/// Compute Chebyshev coefficients for a 3-component vector function.
pub fn coefficients_from_nodes_3d(values: &[[f64; 3]], n: usize) -> Vec<[f64; 3]> {
    let mut result = Vec::with_capacity(n + 1);
    let vx: Vec<f64> = values.iter().map(|v| v[0]).collect();
    let vy: Vec<f64> = values.iter().map(|v| v[1]).collect();
    let vz: Vec<f64> = values.iter().map(|v| v[2]).collect();
    let cx = coefficients_from_nodes(&vx, n);
    let cy = coefficients_from_nodes(&vy, n);
    let cz = coefficients_from_nodes(&vz, n);
    for k in 0..=n {
        result.push([cx[k], cy[k], cz[k]]);
    }
    result
}

// ---------------------------------------------------------------------------
// Chebyshev integration operator
// ---------------------------------------------------------------------------

/// Compute the *indefinite* Chebyshev integral coefficients.
///
/// Given coefficients `c_k` of `f(tau) = sum_{k=0}^{M} c_k T_k(tau)`,
/// return coefficients `d_k` of `F(tau) = integral f(tau) dtau` (in the
/// Chebyshev domain), with the constant of integration chosen so that
/// `F(-1) = 0`.
///
/// The analytic relations are:
///   d_1 = c_0 - c_2 / 2
///   d_k = (c_{k-1} - c_{k+1}) / (2k)   for k >= 2
/// with the convention that c_{M+1} = 0.
///
/// The k=1 case differs because ∫T_0 dτ = T_1 (not T_1/2).
/// d_0 is chosen to enforce `F(-1) = 0`.
pub fn integrate_chebyshev_coeffs(coeffs: &[f64]) -> Vec<f64> {
    let m = coeffs.len(); // M+1 terms  (indices 0..M)
    if m == 0 {
        return vec![];
    }

    let mut d = vec![0.0; m];

    // k=1: d_1 = c_0 - c_2/2  (because ∫T_0 dτ = T_1, full coefficient)
    if m >= 2 {
        let c_2 = if m > 2 { coeffs[2] } else { 0.0 };
        d[1] = coeffs[0] - c_2 / 2.0;
    }

    // k >= 2: d_k = (c_{k-1} - c_{k+1}) / (2k)
    for k in 2..m {
        let c_km1 = coeffs[k - 1];
        let c_kp1 = if k + 1 < m { coeffs[k + 1] } else { 0.0 };
        d[k] = (c_km1 - c_kp1) / (2.0 * k as f64);
    }

    // Choose d_0 so that F(-1) = 0.
    // F(-1) = d_0 + sum_{k=1}^{M} d_k * T_k(-1)
    // T_k(-1) = (-1)^k
    let mut sum_at_minus1 = 0.0;
    for k in 1..m {
        let sign = if k % 2 == 0 { 1.0 } else { -1.0 };
        sum_at_minus1 += d[k] * sign;
    }
    d[0] = -sum_at_minus1;

    d
}

/// Integrate 3-component Chebyshev coefficients (per-component).
pub fn integrate_chebyshev_coeffs_3d(coeffs: &[[f64; 3]]) -> Vec<[f64; 3]> {
    let cx: Vec<f64> = coeffs.iter().map(|c| c[0]).collect();
    let cy: Vec<f64> = coeffs.iter().map(|c| c[1]).collect();
    let cz: Vec<f64> = coeffs.iter().map(|c| c[2]).collect();
    let dx = integrate_chebyshev_coeffs(&cx);
    let dy = integrate_chebyshev_coeffs(&cy);
    let dz = integrate_chebyshev_coeffs(&cz);
    let m = dx.len();
    let mut result = Vec::with_capacity(m);
    for k in 0..m {
        result.push([dx[k], dy[k], dz[k]]);
    }
    result
}

// ---------------------------------------------------------------------------
// Nodal evaluation helper
// ---------------------------------------------------------------------------

/// Evaluate a 3-component Chebyshev series at a single node, using
/// precomputed `T_k(τ_j)` values.
pub fn eval_3d_at_node(coeffs: &[[f64; 3]], t_k_at_node: &[f64]) -> [f64; 3] {
    let m = coeffs.len().min(t_k_at_node.len());
    let mut result = [0.0; 3];
    for k in 0..m {
        let tk = t_k_at_node[k];
        result[0] += coeffs[k][0] * tk;
        result[1] += coeffs[k][1] * tk;
        result[2] += coeffs[k][2] * tk;
    }
    result
}

// ---------------------------------------------------------------------------
// Pre-computed operator matrices (for repeated MCPI calls with same N)
// ---------------------------------------------------------------------------

/// A cached set of operator matrices for a given number of CGL nodes.
#[derive(Debug, Clone)]
pub struct ChebyshevOperators {
    /// Number of intervals (polynomial degree). N+1 nodes.
    pub n: usize,
    /// CGL nodes in [-1, 1], length N+1.
    pub tau: Vec<f64>,
    /// T_k(tau_j) matrix: shape (N+1) x (N+1).
    /// `t_matrix[j][k]` = T_k(tau_j).
    pub t_matrix: Vec<Vec<f64>>,
}

impl ChebyshevOperators {
    /// Build operator matrices for N CGL intervals (N+1 nodes, degree N).
    pub fn new(n: usize) -> Self {
        let tau = cgl_nodes(n);
        let mut t_matrix = Vec::with_capacity(n + 1);
        for j in 0..=n {
            t_matrix.push(chebyshev_t_all(n, tau[j]));
        }
        Self { n, tau, t_matrix }
    }

    /// Compute Chebyshev coefficients from nodal values using the cached
    /// T-matrix (avoids recomputing T_k(τ_j) from scratch).
    pub fn coefficients(&self, values: &[f64]) -> Vec<f64> {
        assert!(
            values.len() == self.n + 1,
            "expected {} values at CGL nodes, got {}",
            self.n + 1,
            values.len()
        );

        let n = self.n;
        let nf = n as f64;
        let mut coeffs = Vec::with_capacity(n + 1);

        for k in 0..=n {
            let mut sum = 0.0;
            for j in 0..=n {
                let t_kj = self.t_matrix[j][k];
                let weight = if j == 0 || j == n { 0.5 } else { 1.0 };
                sum += weight * values[j] * t_kj;
            }
            let mut c_k = 2.0 / nf * sum;
            if k == 0 || k == n {
                c_k *= 0.5;
            }
            coeffs.push(c_k);
        }
        coeffs
    }
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cgl_nodes_basic() {
        let nodes = cgl_nodes(4);
        assert_eq!(nodes.len(), 5);
        assert!((nodes[0] - 1.0).abs() < 1e-15);
        assert!((nodes[4] + 1.0).abs() < 1e-15);
        // Middle node should be cos(pi/2) = 0
        assert!(nodes[2].abs() < 1e-15);
    }

    #[test]
    fn test_chebyshev_t_values() {
        // T_0(x) = 1, T_1(x) = x, T_2(x) = 2x²-1, T_3(x) = 4x³-3x
        let x: f64 = 0.5;
        assert!((chebyshev_t(0, x) - 1.0).abs() < 1e-15);
        assert!((chebyshev_t(1, x) - 0.5).abs() < 1e-15);
        assert!((chebyshev_t(2, x) - (-0.5)).abs() < 1e-15);
        assert!((chebyshev_t(3, x) - (-1.0)).abs() < 1e-15);
    }

    #[test]
    fn test_clenshaw_constant() {
        // f(tau) = 3.0 (constant)  =>  c_0 = 3, rest = 0
        let coeffs = vec![3.0, 0.0, 0.0];
        assert!((clenshaw(&coeffs, 0.5) - 3.0).abs() < 1e-15);
    }

    #[test]
    fn test_clenshaw_linear() {
        // f(tau) = 2 + 3*tau  => c_0=2, c_1=3
        let coeffs = vec![2.0, 3.0];
        let tau: f64 = 0.7;
        let expected = 2.0 + 3.0 * tau;
        assert!((clenshaw(&coeffs, tau) - expected).abs() < 1e-14);
    }

    #[test]
    fn test_coefficients_roundtrip() {
        // Sample a known polynomial at CGL nodes, compute coefficients,
        // then evaluate at an intermediate point.
        let n: usize = 8;
        let nodes = cgl_nodes(n);

        // f(tau) = 3*T_0 + 2*T_1 + 1.5*T_2 = 3 + 2*tau + 1.5*(2*tau²-1)
        let values: Vec<f64> = nodes
            .iter()
            .map(|&tau| 3.0 + 2.0 * tau + 1.5 * (2.0 * tau * tau - 1.0))
            .collect();

        let coeffs = coefficients_from_nodes(&values, n);

        // The first three coefficients should be 3.0, 2.0, 1.5.
        // The rest should be ~0.
        assert!((coeffs[0] - 3.0).abs() < 1e-12, "c0 = {}", coeffs[0]);
        assert!((coeffs[1] - 2.0).abs() < 1e-12, "c1 = {}", coeffs[1]);
        assert!((coeffs[2] - 1.5).abs() < 1e-12, "c2 = {}", coeffs[2]);
        for k in 3..=n {
            assert!(
                coeffs[k].abs() < 1e-12,
                "c{} = {} (should be 0)",
                k,
                coeffs[k]
            );
        }

        // Evaluate at tau = 0.3
        let tau: f64 = 0.3;
        let approx_val = clenshaw(&coeffs, tau);
        let exact = 3.0 + 2.0 * tau + 1.5 * (2.0 * tau * tau - 1.0);
        assert!((approx_val - exact).abs() < 1e-11);
    }

    #[test]
    fn test_integration_of_constant() {
        // f(tau) = 1  =>  integral from -1 to tau  =  tau + 1
        // In Chebyshev: c_0 = 1, rest 0
        // Integral F(tau) with F(-1) = 0  =>  F(tau) = tau + 1
        let coeffs = vec![1.0, 0.0, 0.0, 0.0];
        let d = integrate_chebyshev_coeffs(&coeffs);

        for &tau in &[-1.0, 0.0, 0.5, 1.0] {
            let f_val = clenshaw(&d, tau);
            let expected = tau + 1.0;
            assert!(
                (f_val - expected).abs() < 1e-13,
                "at tau={tau}: F={f_val}, expected={expected}"
            );
        }
    }

    #[test]
    fn test_integration_of_linear() {
        // f(tau) = tau = T_1(tau)  =>  integral from -1 to tau = (tau²-1)/2
        // In Chebyshev: c_0 = 0, c_1 = 1
        let coeffs = vec![0.0, 1.0, 0.0, 0.0];
        let d = integrate_chebyshev_coeffs(&coeffs);

        for &tau in &[-1.0, -0.5, 0.0, 0.5, 1.0] {
            let f_val = clenshaw(&d, tau);
            let expected = (tau * tau - 1.0) / 2.0;
            assert!(
                (f_val - expected).abs() < 1e-13,
                "at tau={tau}: F={f_val}, expected={expected}"
            );
        }
    }

    #[test]
    fn test_operators_construction() {
        let ops = ChebyshevOperators::new(4);
        assert_eq!(ops.tau.len(), 5);
        assert_eq!(ops.t_matrix.len(), 5);
        assert_eq!(ops.t_matrix[0].len(), 5);
        // T_0 at every node = 1
        for j in 0..5 {
            assert!((ops.t_matrix[j][0] - 1.0).abs() < 1e-15);
        }
    }
}
