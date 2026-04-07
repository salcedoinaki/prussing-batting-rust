/// Errors produced by the Lambert solver.
#[derive(Debug, Clone)]
pub enum LambertError {
    /// Newton iteration did not converge for a particular branch.
    NoConvergence { n_revs: u32, branch: &'static str },
    /// The input parameters are physically invalid.
    InvalidInput(String),
    /// No feasible solution exists for the given constraints.
    NoSolution,
}
