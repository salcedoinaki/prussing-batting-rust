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
