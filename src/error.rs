use std::fmt;

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

impl fmt::Display for LambertError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LambertError::NoConvergence { n_revs, branch } => {
                write!(f, "no convergence for N={n_revs} ({branch} branch)")
            }
            LambertError::InvalidInput(msg) => write!(f, "invalid input: {msg}"),
            LambertError::NoSolution => write!(f, "no feasible solution found"),
        }
    }
}

impl std::error::Error for LambertError {}
