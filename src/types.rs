use nalgebra::Vector3;

/// Direction of the transfer orbit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    /// Short-way transfer (0 < theta < pi)
    Prograde,
    /// Long-way transfer (pi < theta < 2*pi)
    Retrograde,
}

/// Which branch of a multi-revolution solution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Branch {
    /// The single fractional-orbit solution (N = 0)
    Fractional,
    /// Lower branch: faster transfer for the given N (a < a_min_energy for that N)
    Lower,
    /// Upper branch: slower transfer for the given N (a > a_min_energy for that N)
    Upper,
}

/// Input parameters for the Lambert solver.
#[derive(Debug, Clone)]
pub struct LambertInput {
    /// Initial position vector (km)
    pub r1: Vector3<f64>,
    /// Final position vector (km)
    pub r2: Vector3<f64>,
    /// Time of flight (seconds)
    pub tof: f64,
    /// Gravitational parameter (km³/s²)
    pub mu: f64,
    /// Transfer direction
    pub direction: Direction,
    /// Maximum number of complete revolutions to search.
    /// `None` means auto-determine N_max.
    pub max_revs: Option<u32>,
}

/// A single Lambert problem solution.
#[derive(Debug, Clone)]
pub struct LambertSolution {
    /// Departure velocity vector (km/s)
    pub v1: Vector3<f64>,
    /// Arrival velocity vector (km/s)
    pub v2: Vector3<f64>,
    /// Semi-major axis of the transfer orbit (km)
    pub a: f64,
    /// Number of complete revolutions
    pub n_revs: u32,
    /// Solution branch
    pub branch: Branch,
    /// Transfer true-anomaly angle (rad)
    pub transfer_angle: f64,
    /// Whether the solution passes feasibility checks
    pub is_feasible: bool,
}

/// Transfer geometry quantities derived from r1, r2, and direction.
#[derive(Debug, Clone, Copy)]
pub struct TransferGeometry {
    /// |r1|
    pub r1_mag: f64,
    /// |r2|
    pub r2_mag: f64,
    /// Chord length |r2 - r1|
    pub c: f64,
    /// Semi-perimeter (r1 + r2 + c) / 2
    pub s: f64,
    /// Transfer angle theta (0, 2*pi) based on direction
    pub theta: f64,
    /// Minimum-energy semi-major axis a_m = s / 2
    pub a_min: f64,
}

/// Configuration for feasibility filtering.
#[derive(Debug, Clone)]
pub struct FeasibilityConfig {
    pub check_earth_collision: bool,
    pub earth_radius: f64,
    pub max_delta_v: Option<f64>,
    pub check_escape_velocity: bool,
}

impl Default for FeasibilityConfig {
    fn default() -> Self {
        Self {
            check_earth_collision: true,
            earth_radius: 6378.137, // km
            max_delta_v: None,
            check_escape_velocity: false,
        }
    }
}

/// Configuration for the unified Lambert solver (Keplerian + optional perturbed refinement).
#[derive(Debug, Clone)]
pub struct UnifiedLambertConfig {
    /// Transfer direction.
    pub direction: Direction,
    /// Maximum complete revolutions to search. `None` = auto-determine N_max.
    pub max_revs: Option<u32>,
    /// Feasibility filtering options.
    pub feasibility: FeasibilityConfig,
    /// MCPI / perturbed solver tuning.
    pub perturbed: PerturbedConfig,
}

impl Default for UnifiedLambertConfig {
    fn default() -> Self {
        Self {
            direction: Direction::Prograde,
            max_revs: None,
            feasibility: FeasibilityConfig::default(),
            perturbed: PerturbedConfig::default(),
        }
    }
}

/// Tuning knobs for the MCPI-based perturbed solvers.
#[derive(Debug, Clone)]
pub struct PerturbedConfig {
    /// Chebyshev polynomial degree for the perturbed propagation.
    pub poly_degree: usize,
    /// Maximum Picard / MCPI iterations per propagation.
    pub max_iterations: usize,
    /// Convergence tolerance for MCPI (position, km).
    pub mcpi_tolerance: f64,
    /// Maximum outer MPS iterations (only used by MPS-IVP).
    pub max_mps_iterations: usize,
    /// Terminal position tolerance for MPS-IVP (km).
    pub mps_tolerance: f64,
    /// Relative perturbation magnitude for MPS particular solutions.
    pub perturbation_scale: f64,
    /// Use low-fidelity model for MPS particular solutions.
    pub variable_fidelity: bool,
}

impl Default for PerturbedConfig {
    fn default() -> Self {
        Self {
            poly_degree: 100,
            max_iterations: 50,
            mcpi_tolerance: 1e-12,
            max_mps_iterations: 15,
            mps_tolerance: 1e-4,
            perturbation_scale: 1e-7,
            variable_fidelity: false,
        }
    }
}

/// A perturbed Lambert solution (refined from a Keplerian warm start).
#[derive(Debug, Clone)]
pub struct PerturbedSolution {
    /// Departure velocity (km/s).
    pub v1: Vector3<f64>,
    /// Arrival velocity (km/s).
    pub v2: Vector3<f64>,
    /// Keplerian semi-major axis of the warm-start orbit (km).
    pub a_keplerian: f64,
    /// Number of complete revolutions.
    pub n_revs: u32,
    /// Solution branch.
    pub branch: Branch,
    /// Transfer true-anomaly angle from the Keplerian solution (rad).
    pub transfer_angle: f64,
    /// Which perturbed algorithm was used.
    pub algorithm: crate::perturbed::selector::PerturbedAlgorithm,
    /// Whether the perturbed solver converged.
    pub converged: bool,
}
