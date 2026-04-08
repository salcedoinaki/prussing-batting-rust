use clap::Parser;
use nalgebra::Vector3;

use lambert_ult::error::LambertError;
use lambert_ult::types::{Direction, LambertInput};
use lambert_ult::solve_lambert;

/// Unified Lambert Tool — solve the two-body Lambert problem using the
/// Prussing multi-revolution algorithm.
#[derive(Parser, Debug)]
#[command(version, about)]
struct Cli {
    /// Initial position vector components x y z (km)
    #[arg(long, num_args = 3, allow_hyphen_values = true)]
    r1: Vec<f64>,

    /// Final position vector components x y z (km)
    #[arg(long, num_args = 3, allow_hyphen_values = true)]
    r2: Vec<f64>,

    /// Time of flight (seconds)
    #[arg(long)]
    tof: f64,

    /// Gravitational parameter (km³/s²) [default: Earth]
    #[arg(long, default_value_t = 398600.4418)]
    mu: f64,

    /// Transfer direction: prograde or retrograde
    #[arg(long, default_value = "prograde")]
    direction: String,

    /// Maximum number of complete revolutions to search
    #[arg(long)]
    max_revs: Option<u32>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    if cli.r1.len() != 3 || cli.r2.len() != 3 {
        return Err(Box::new(LambertError::InvalidInput(
            "r1 and r2 must each have exactly 3 components".into(),
        )));
    }

    let direction = match cli.direction.to_lowercase().as_str() {
        "prograde" | "pro" | "short" => Direction::Prograde,
        "retrograde" | "retro" | "long" => Direction::Retrograde,
        other => {
            return Err(Box::new(LambertError::InvalidInput(format!(
                "unknown direction '{other}'; use prograde or retrograde"
            ))));
        }
    };

    let input = LambertInput {
        r1: Vector3::new(cli.r1[0], cli.r1[1], cli.r1[2]),
        r2: Vector3::new(cli.r2[0], cli.r2[1], cli.r2[2]),
        tof: cli.tof,
        mu: cli.mu,
        direction,
        max_revs: cli.max_revs,
    };

    let solutions = solve_lambert(&input)?;

    println!(
        "{:<6} {:<10} {:>14} {:>14} {:>14}  {:>14} {:>14} {:>14}  {:>12}",
        "N", "Branch", "v1x", "v1y", "v1z", "v2x", "v2y", "v2z", "a (km)"
    );
    println!("{}", "-".repeat(120));

    for sol in &solutions {
        println!(
            "{:<6} {:<10} {:>14.6} {:>14.6} {:>14.6}  {:>14.6} {:>14.6} {:>14.6}  {:>12.3}",
            sol.n_revs,
            format!("{:?}", sol.branch),
            sol.v1.x,
            sol.v1.y,
            sol.v1.z,
            sol.v2.x,
            sol.v2.y,
            sol.v2.z,
            sol.a,
        );
    }

    Ok(())
}
