/// Direction of the transfer orbit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    /// Short-way transfer (0 < theta < pi)
    Prograde,
    /// Long-way transfer (pi < theta < 2*pi)
    Retrograde,
}
