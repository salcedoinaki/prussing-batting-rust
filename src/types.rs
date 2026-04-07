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
