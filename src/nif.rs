//! Rustler NIF bindings for the Unified Lambert Tool.
//!
//! Exposes three entry points to Elixir (module
//! `Elixir.Solsticex.Rust.Solver`):
//!
//! - `run_batch_solve/5`            — Keplerian multi-rev batch
//! - `run_batch_solve_perturbed/5`  — Full three-stage pipeline with J2
//! - `run_lambert/5`                — Single-window backward-compat solve
//!
//! ## Pipeline
//!
//! Every batch window follows three stages:
//!
//! 1. **Gooding / Lancaster–Blanchard warm-start** — cheap multi-rev
//!    Keplerian solver that emits all 2·N_max + 1 candidates.
//! 2. **Heuristic filter** — discard candidates by Earth-collision, escape
//!    velocity, total Δv budget (relative to the SGP4 orbit velocities at
//!    departure / arrival), and transfer-time window.
//! 3. **Full Prussing-Batting (perturbed)** — each surviving candidate seeds
//!    the MCPI perturbed solver (TPBVP / KS-TPBVP / MPS-IVP selected via the
//!    standard algorithm selector) under the Earth J2 zonal-gravity model.
//!
//! `run_batch_solve` stops after stage 2 and picks the best Δv among the
//! surviving Keplerian candidates (the cheap path). `run_batch_solve_perturbed`
//! runs all three stages. `run_lambert` uses the Gooding + default feasibility
//! convenience helper and picks the minimum-‖v1‖ solution.

use nalgebra::Vector3;

use crate::constants::MU_EARTH;
use crate::force_models::gravity::ZonalGravity;
use crate::force_models::ForceModel;
use crate::keplerian::feasibility::filter_feasibility_with_ref;
use crate::keplerian::gooding::solve_gooding;
use crate::perturbed::selector::{select_algorithm, PerturbedAlgorithm};
use crate::solve_lambert_gooding;
use crate::types::{
    Direction, FeasibilityConfig, LambertInput, LambertSolution, PerturbedConfig,
};

// ── Heuristic thresholds (stage 2) ──────────────────────────────────────
//
// These thresholds are intentionally loose so existing Starlink fixtures
// pass; they act as sanity bounds, not tight tuning knobs.

/// Total Δv budget applied to Gooding candidates, in km/s
/// (|v1 − v1_ref| + |v2_ref − v2|).
const MAX_TOTAL_DV_KM_S: f64 = 5.0;

/// Minimum acceptable transfer time (seconds).
const MIN_TOF_S: f64 = 60.0;

/// Maximum acceptable transfer time (seconds, ~7 days).
const MAX_TOF_S: f64 = 7.0 * 86400.0;

// ── SGP4 helpers ────────────────────────────────────────────────────────

const DAYS_PER_YEAR: f64 = 365.25;
const MINUTES_PER_DAY: f64 = 1440.0;
const J2000_EPOCH_JD_TIME: f64 = 2451545.0;
const M_PER_KM: f64 = 1000.0;

/// SGP4 propagate; returns (position_km, velocity_km_s).
fn propagate_sgp4(
    constants: &sgp4::Constants,
    tle_epoch_jdtime: f64,
    target_jdtime: f64,
) -> Result<([f64; 3], [f64; 3]), String> {
    let minutes = (target_jdtime - tle_epoch_jdtime) * MINUTES_PER_DAY;
    let prediction = constants
        .propagate(sgp4::MinutesSinceEpoch(minutes))
        .map_err(|e| format!("SGP4 propagation error: {e}"))?;

    Ok((
        [prediction.position[0], prediction.position[1], prediction.position[2]],
        [prediction.velocity[0], prediction.velocity[1], prediction.velocity[2]],
    ))
}

fn tle_epoch_jdtime(elements: &sgp4::Elements) -> f64 {
    elements.epoch() * DAYS_PER_YEAR + J2000_EPOCH_JD_TIME
}

// ── Velocity helpers (km/s) ─────────────────────────────────────────────

fn vec3_sub(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn vec3_mag(v: &[f64; 3]) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

fn to_arr3(v: &Vector3<f64>) -> [f64; 3] {
    [v.x, v.y, v.z]
}

/// Total Δv for a Keplerian solution given the actual velocities at
/// departure and arrival (all in km/s).
fn solution_delta_v(sol: &LambertSolution, vel1: &[f64; 3], vel2: &[f64; 3]) -> f64 {
    let v1t = to_arr3(&sol.v1);
    let v2t = to_arr3(&sol.v2);
    let dv1 = vec3_sub(&v1t, vel1);
    let dv2 = vec3_sub(vel2, &v2t);
    vec3_mag(&dv1) + vec3_mag(&dv2)
}

/// Stage-2 heuristic filter: apply feasibility (collision, escape, Δv
/// budget against SGP4 orbit velocities) plus a transfer-time window.
///
/// Mutates `candidates` by retaining only feasible solutions.
fn apply_heuristic_filter(
    candidates: &mut Vec<LambertSolution>,
    r1: &Vector3<f64>,
    r2: &Vector3<f64>,
    vel1: &[f64; 3],
    vel2: &[f64; 3],
    tof: f64,
) {
    if tof < MIN_TOF_S || tof > MAX_TOF_S {
        candidates.clear();
        return;
    }

    let feas_config = FeasibilityConfig {
        check_earth_collision: true,
        earth_radius: 6378.137,
        max_delta_v: Some(MAX_TOTAL_DV_KM_S),
        check_escape_velocity: true,
    };
    let v1_ref = Vector3::new(vel1[0], vel1[1], vel1[2]);
    let v2_ref = Vector3::new(vel2[0], vel2[1], vel2[2]);
    filter_feasibility_with_ref(
        candidates,
        r1,
        r2,
        MU_EARTH,
        &feas_config,
        Some(&v1_ref),
        Some(&v2_ref),
    );
    candidates.retain(|s| s.is_feasible);
}

/// Stage-3 dispatch: refine a single Keplerian warm-start candidate under
/// the given force model, routing to the appropriate MCPI solver.
fn refine_perturbed(
    r1: &Vector3<f64>,
    r2: &Vector3<f64>,
    tof: f64,
    ksol: &LambertSolution,
    force_model: &dyn ForceModel,
    pc: &PerturbedConfig,
) -> (Vector3<f64>, Vector3<f64>, bool) {
    let algo = select_algorithm(ksol.transfer_angle, ksol.n_revs);
    match algo {
        PerturbedAlgorithm::McpiTpbvp => {
            let cfg = crate::perturbed::tpbvp::TpbvpConfig {
                poly_degree: pc.poly_degree,
                max_iterations: pc.max_iterations,
                tolerance: pc.mcpi_tolerance,
            };
            let res = crate::perturbed::tpbvp::solve_tpbvp(
                r1, r2, 0.0, tof, &ksol.v1, force_model, &cfg,
            );
            (res.v1, res.v2, res.converged)
        }
        PerturbedAlgorithm::McpiKsTpbvp => {
            let cfg = crate::perturbed::ks_tpbvp::KsTpbvpConfig {
                poly_degree: pc.poly_degree,
                max_iterations: pc.max_iterations,
                tolerance: pc.mcpi_tolerance,
            };
            let res = crate::perturbed::ks_tpbvp::solve_ks_tpbvp(
                r1, r2, 0.0, tof, &ksol.v1, ksol.a, force_model, &cfg,
            );
            (res.v1, res.v2, res.converged)
        }
        PerturbedAlgorithm::McpiMpsIvp => {
            let cfg = crate::perturbed::mps_ivp::MpsIvpConfig {
                poly_degree: pc.poly_degree,
                max_mcpi_iterations: pc.max_iterations,
                mcpi_tolerance: pc.mcpi_tolerance,
                max_mps_iterations: pc.max_mps_iterations,
                mps_tolerance: pc.mps_tolerance,
                perturbation_scale: pc.perturbation_scale,
                variable_fidelity: pc.variable_fidelity,
            };
            let res = crate::perturbed::mps_ivp::solve_mps_ivp(
                r1, r2, 0.0, tof, &ksol.v1, force_model, &cfg,
            );
            (res.v1, res.v2, res.converged)
        }
    }
}

// ── Convert km → m helpers (for NIF output) ─────────────────────────────

fn km_arr_to_m(a: &[f64; 3]) -> Vec<f64> {
    vec![a[0] * M_PER_KM, a[1] * M_PER_KM, a[2] * M_PER_KM]
}

fn km_vec3_to_m(v: &Vector3<f64>) -> Vec<f64> {
    vec![v.x * M_PER_KM, v.y * M_PER_KM, v.z * M_PER_KM]
}

// ── NIF result type ─────────────────────────────────────────────────────

/// Matches the existing Elixir-side WindowResult shape exactly.
type WindowResult = (
    (Vec<f64>, Vec<f64>, Vec<f64>), // (pos1_m, vel1_m/s, v1_transfer_m/s)
    (Vec<f64>, Vec<f64>, Vec<f64>), // (pos2_m, vel2_m/s, v2_transfer_m/s)
    f64,                            // delta_v_total (m/s)
    f64,                            // transfer_time (s)
);

// ── Single-window solver (Keplerian multi-rev, stages 1+2 only) ─────────

fn solve_window_keplerian(
    constants_a: &sgp4::Constants,
    epoch_jd_a: f64,
    constants_b: &sgp4::Constants,
    epoch_jd_b: f64,
    jd_departure: f64,
    jd_arrival: f64,
) -> Result<WindowResult, String> {
    let (pos1, vel1) = propagate_sgp4(constants_a, epoch_jd_a, jd_departure)?;
    let (pos2, vel2) = propagate_sgp4(constants_b, epoch_jd_b, jd_arrival)?;

    let transfer_time = (jd_arrival - jd_departure) * 86400.0;
    if transfer_time <= 0.0 {
        return Err("Non-positive transfer time".into());
    }

    let r1 = Vector3::new(pos1[0], pos1[1], pos1[2]);
    let r2 = Vector3::new(pos2[0], pos2[1], pos2[2]);

    let input = LambertInput {
        r1,
        r2,
        tof: transfer_time,
        mu: MU_EARTH,
        direction: Direction::Prograde,
        max_revs: None,
    };

    // Stage 1: Gooding warm-start.
    let mut candidates =
        solve_gooding(&input).map_err(|e| format!("Gooding error: {e}"))?;

    // Stage 2: heuristic filter (collision + escape + Δv + TOF window).
    apply_heuristic_filter(&mut candidates, &r1, &r2, &vel1, &vel2, transfer_time);

    if candidates.is_empty() {
        return Err("No feasible Keplerian candidate after heuristic filter".into());
    }

    // Pick the surviving Gooding candidate with the lowest total Δv.
    let best = candidates
        .iter()
        .min_by(|a, b| {
            solution_delta_v(a, &vel1, &vel2)
                .partial_cmp(&solution_delta_v(b, &vel1, &vel2))
                .unwrap()
        })
        .unwrap();

    let v1_transfer = to_arr3(&best.v1);
    let v2_transfer = to_arr3(&best.v2);

    let dv1 = vec3_sub(&v1_transfer, &vel1);
    let dv2 = vec3_sub(&vel2, &v2_transfer);
    let delta_v_total_km = vec3_mag(&dv1) + vec3_mag(&dv2);

    Ok((
        (km_arr_to_m(&pos1), km_arr_to_m(&vel1), km_arr_to_m(&v1_transfer)),
        (km_arr_to_m(&pos2), km_arr_to_m(&vel2), km_arr_to_m(&v2_transfer)),
        delta_v_total_km * M_PER_KM,
        transfer_time,
    ))
}

// ── Single-window solver (full three-stage pipeline with J2) ────────────

fn solve_window_perturbed(
    constants_a: &sgp4::Constants,
    epoch_jd_a: f64,
    constants_b: &sgp4::Constants,
    epoch_jd_b: f64,
    jd_departure: f64,
    jd_arrival: f64,
    force_model: &dyn ForceModel,
    pc: &PerturbedConfig,
) -> Result<WindowResult, String> {
    let (pos1, vel1) = propagate_sgp4(constants_a, epoch_jd_a, jd_departure)?;
    let (pos2, vel2) = propagate_sgp4(constants_b, epoch_jd_b, jd_arrival)?;

    let transfer_time = (jd_arrival - jd_departure) * 86400.0;
    if transfer_time <= 0.0 {
        return Err("Non-positive transfer time".into());
    }

    let r1 = Vector3::new(pos1[0], pos1[1], pos1[2]);
    let r2 = Vector3::new(pos2[0], pos2[1], pos2[2]);

    let input = LambertInput {
        r1,
        r2,
        tof: transfer_time,
        mu: MU_EARTH,
        direction: Direction::Prograde,
        max_revs: None,
    };

    // Stage 1: Gooding warm-start.
    let mut candidates =
        solve_gooding(&input).map_err(|e| format!("Gooding error: {e}"))?;

    // Stage 2: heuristic filter.
    apply_heuristic_filter(&mut candidates, &r1, &r2, &vel1, &vel2, transfer_time);

    if candidates.is_empty() {
        return Err("No feasible Gooding candidate after heuristic filter".into());
    }

    // Stage 3: perturbed refinement on each surviving candidate.
    let mut refined: Vec<(Vector3<f64>, Vector3<f64>, bool)> =
        Vec::with_capacity(candidates.len());
    for ksol in &candidates {
        refined.push(refine_perturbed(&r1, &r2, transfer_time, ksol, force_model, pc));
    }

    // Prefer converged solutions; fall back to unconverged if none converged.
    let best = refined
        .iter()
        .filter(|(_, _, conv)| *conv)
        .min_by(|a, b| {
            let dv_a = vec3_mag(&vec3_sub(&to_arr3(&a.0), &vel1))
                + vec3_mag(&vec3_sub(&vel2, &to_arr3(&a.1)));
            let dv_b = vec3_mag(&vec3_sub(&to_arr3(&b.0), &vel1))
                + vec3_mag(&vec3_sub(&vel2, &to_arr3(&b.1)));
            dv_a.partial_cmp(&dv_b).unwrap()
        })
        .or_else(|| {
            refined.iter().min_by(|a, b| {
                let dv_a = vec3_mag(&vec3_sub(&to_arr3(&a.0), &vel1))
                    + vec3_mag(&vec3_sub(&vel2, &to_arr3(&a.1)));
                let dv_b = vec3_mag(&vec3_sub(&to_arr3(&b.0), &vel1))
                    + vec3_mag(&vec3_sub(&vel2, &to_arr3(&b.1)));
                dv_a.partial_cmp(&dv_b).unwrap()
            })
        })
        .ok_or_else(|| "No perturbed solution produced".to_string())?;

    let v1_transfer = to_arr3(&best.0);
    let v2_transfer = to_arr3(&best.1);

    let dv1 = vec3_sub(&v1_transfer, &vel1);
    let dv2 = vec3_sub(&vel2, &v2_transfer);
    let delta_v_total_km = vec3_mag(&dv1) + vec3_mag(&dv2);

    Ok((
        (km_arr_to_m(&pos1), km_arr_to_m(&vel1), km_arr_to_m(&v1_transfer)),
        (km_arr_to_m(&pos2), km_arr_to_m(&vel2), km_arr_to_m(&v2_transfer)),
        delta_v_total_km * M_PER_KM,
        transfer_time,
    ))
}

// ── Common TLE parsing ──────────────────────────────────────────────────

struct ParsedTleSet {
    constants_a: sgp4::Constants,
    epoch_jd_a: f64,
    constants_b: sgp4::Constants,
    epoch_jd_b: f64,
}

fn parse_tle_pair(
    tle1_a: &str,
    tle2_a: &str,
    tle1_b: &str,
    tle2_b: &str,
) -> Result<ParsedTleSet, String> {
    let elements_a = sgp4::Elements::from_tle(None, tle1_a.as_bytes(), tle2_a.as_bytes())
        .map_err(|e| format!("TLE parse error (sat A): {e}"))?;
    let elements_b = sgp4::Elements::from_tle(None, tle1_b.as_bytes(), tle2_b.as_bytes())
        .map_err(|e| format!("TLE parse error (sat B): {e}"))?;

    let constants_a = sgp4::Constants::from_elements(&elements_a)
        .map_err(|e| format!("SGP4 constants error (sat A): {e}"))?;
    let constants_b = sgp4::Constants::from_elements(&elements_b)
        .map_err(|e| format!("SGP4 constants error (sat B): {e}"))?;

    Ok(ParsedTleSet {
        epoch_jd_a: tle_epoch_jdtime(&elements_a),
        constants_a,
        epoch_jd_b: tle_epoch_jdtime(&elements_b),
        constants_b,
    })
}

// ═══════════════════════════════════════════════════════════════════════
//  NIF entry points
// ═══════════════════════════════════════════════════════════════════════

/// Batch-solve Lambert transfers (Keplerian multi-revolution).
///
/// Runs stages 1–2 of the pipeline (Gooding warm-start + heuristic filter)
/// and returns the lowest-Δv surviving candidate per window.
#[rustler::nif(schedule = "DirtyCpu")]
pub fn run_batch_solve(
    tle1_a: String,
    tle2_a: String,
    tle1_b: String,
    tle2_b: String,
    windows: Vec<(f64, f64)>,
) -> Vec<Result<WindowResult, String>> {
    let parsed = match parse_tle_pair(&tle1_a, &tle2_a, &tle1_b, &tle2_b) {
        Ok(p) => p,
        Err(e) => return windows.iter().map(|_| Err(e.clone())).collect(),
    };

    windows
        .iter()
        .map(|&(jd1, jd2)| {
            solve_window_keplerian(
                &parsed.constants_a,
                parsed.epoch_jd_a,
                &parsed.constants_b,
                parsed.epoch_jd_b,
                jd1,
                jd2,
            )
        })
        .collect()
}

/// Batch-solve Lambert transfers with the full three-stage pipeline:
/// Gooding warm-start → heuristic filter → J2 MCPI perturbed refinement.
#[rustler::nif(schedule = "DirtyCpu")]
pub fn run_batch_solve_perturbed(
    tle1_a: String,
    tle2_a: String,
    tle1_b: String,
    tle2_b: String,
    windows: Vec<(f64, f64)>,
) -> Vec<Result<WindowResult, String>> {
    let parsed = match parse_tle_pair(&tle1_a, &tle2_a, &tle1_b, &tle2_b) {
        Ok(p) => p,
        Err(e) => return windows.iter().map(|_| Err(e.clone())).collect(),
    };

    let force_model = ZonalGravity::earth_j2();
    let pc = PerturbedConfig::default();

    windows
        .iter()
        .map(|&(jd1, jd2)| {
            solve_window_perturbed(
                &parsed.constants_a,
                parsed.epoch_jd_a,
                &parsed.constants_b,
                parsed.epoch_jd_b,
                jd1,
                jd2,
                &force_model,
                &pc,
            )
        })
        .collect()
}

/// Backward-compatible single-window Lambert solve.
///
/// Accepts positions in **meters** and time in seconds (matching the old
/// `lambert-bate` NIF). Returns transfer velocities in **m/s**.
/// Internally converts to km, runs the Gooding warm-start with default
/// feasibility filtering, and picks the minimum-‖v1‖ solution.
#[rustler::nif(schedule = "DirtyCpu")]
pub fn run_lambert(
    r1: Vec<f64>,
    r2: Vec<f64>,
    tof: f64,
    short: bool,
    _iterations: usize,
) -> Result<(Vec<f64>, Vec<f64>), String> {
    if r1.len() != 3 || r2.len() != 3 {
        return Err("r1 and r2 must each have exactly 3 elements".into());
    }

    let r1_km = Vector3::new(r1[0] / M_PER_KM, r1[1] / M_PER_KM, r1[2] / M_PER_KM);
    let r2_km = Vector3::new(r2[0] / M_PER_KM, r2[1] / M_PER_KM, r2[2] / M_PER_KM);

    // Prograde is the natural short arc when cross_z >= 0. If the caller
    // flips `short`, invert to Retrograde.
    let cross_z = r1_km.x * r2_km.y - r1_km.y * r2_km.x;
    let natural_short = cross_z >= 0.0;
    let direction = if short == natural_short {
        Direction::Prograde
    } else {
        Direction::Retrograde
    };

    let input = LambertInput {
        r1: r1_km,
        r2: r2_km,
        tof,
        mu: MU_EARTH,
        direction,
        max_revs: None,
    };

    let solutions =
        solve_lambert_gooding(&input).map_err(|e| format!("Lambert error: {e}"))?;

    let best = solutions
        .iter()
        .filter(|s| s.is_feasible)
        .min_by(|a, b| a.v1.norm().partial_cmp(&b.v1.norm()).unwrap())
        .or_else(|| {
            solutions
                .iter()
                .min_by(|a, b| a.v1.norm().partial_cmp(&b.v1.norm()).unwrap())
        })
        .ok_or_else(|| "No solution found".to_string())?;

    Ok((km_vec3_to_m(&best.v1), km_vec3_to_m(&best.v2)))
}
