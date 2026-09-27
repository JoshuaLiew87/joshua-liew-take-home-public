"""Generate, estimate, and analyse the final moving-spot scenario."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.signal import detrend

from common.io import save_json
from common.rng import make_rng_streams
from config import ScenarioConfig, scenario_config
from estimators.centroid import estimate_centroid
from estimators.gaussian_fit import estimate_gaussian
from sim.simulator import SpotSimulator
from sim.trajectory import TrajectoryGenerator


@dataclass(frozen=True)
class DynamicSummary:
    """Trajectory and disturbance-recovery metrics for one estimator."""

    estimator: str
    target_snr: float
    frame_count: int
    valid_count: int
    failure_count: int
    failure_rate: float
    lock_failure_count: int
    lock_failure_rate: float
    rms_error_px: float
    p95_error_px: float
    disturbance_axis: str
    true_frequency_hz: float
    estimated_frequency_hz: float
    frequency_abs_error_hz: float
    true_amplitude_px: float
    estimated_amplitude_px: float
    amplitude_abs_error_px: float
    spectral_noise_floor_px: float
    peak_to_noise_ratio: float


@dataclass(frozen=True)
class DynamicResult:
    """Saved-array portion of a dynamic experiment."""

    positions_xy: np.ndarray
    estimates_xy: dict[str, np.ndarray]
    saturation_counts: np.ndarray
    summaries: list[DynamicSummary]
    spectra: dict[str, tuple[np.ndarray, np.ndarray]]


def generate_dynamic_sequence(
    config: ScenarioConfig,
    *,
    target_snr: float,
    frame_count: int,
) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
    """Generate one shared trajectory and frame sequence."""

    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    streams = make_rng_streams(config.experiment.master_seed)
    simulator = SpotSimulator(config, rng_streams=streams)
    generator = TrajectoryGenerator(
        config.trajectory,
        sensor_shape=(config.sensor.height_px, config.sensor.width_px),
        position_margin_px=config.spot.position_margin_px,
        rng=streams.trajectory,
    )
    positions = generator.generate(frame_count)
    frames: list[np.ndarray] = []
    saturation_counts = np.zeros(frame_count, dtype=int)
    for index, position in enumerate(positions):
        frame = simulator.render(tuple(position), target_snr=target_snr)
        frames.append(frame.image_dn.copy())
        saturation_counts[index] = frame.metadata.saturated_pixel_count
    return positions, frames, saturation_counts


def estimate_dynamic_trajectory(
    frames: list[np.ndarray],
    config: ScenarioConfig,
    *,
    estimator: str,
    warm_start: bool = False,
) -> np.ndarray:
    """Estimate each frame without using the true trajectory."""

    if estimator not in {"centroid", "gaussian"}:
        raise ValueError("estimator must be centroid or gaussian")
    if warm_start and estimator != "gaussian":
        raise ValueError("warm_start is only supported for the Gaussian fit")
    estimates = np.full((len(frames), 2), np.nan, dtype=float)
    previous_position: tuple[float, float] | None = None
    for index, frame in enumerate(frames):
        if estimator == "centroid":
            result = estimate_centroid(frame, config)
        else:
            result = estimate_gaussian(
                frame,
                config,
                previous_position=previous_position if warm_start else None,
            )
        if result.success and result.position_xy is not None:
            estimates[index] = result.position_xy
            if warm_start:
                previous_position = result.position_xy
    return estimates


def _filled_detrended_signal(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fill invalid samples for spectral analysis and remove a linear trend."""

    signal = np.asarray(values, dtype=float)
    valid = np.isfinite(signal)
    if np.count_nonzero(valid) < 4:
        raise ValueError("too few valid samples for dynamic spectral analysis")
    indices = np.arange(signal.size, dtype=float)
    filled = np.interp(indices, indices[valid], signal[valid])
    return detrend(filled, type="linear"), valid


def _fit_sinusoid(
    signal: np.ndarray,
    sample_rate_hz: float,
    peak_frequency_hz: float,
    *,
    search_half_width_hz: float,
) -> tuple[float, float, float]:
    """Refine frequency and amplitude using sinusoidal least squares."""

    sample_count = signal.size
    time_s = np.arange(sample_count, dtype=float) / sample_rate_hz
    time_centered = time_s - np.mean(time_s)

    def solve(frequency_hz: float) -> tuple[float, np.ndarray]:
        design = np.column_stack(
            (
                np.ones(sample_count),
                time_centered,
                np.sin(2.0 * np.pi * frequency_hz * time_s),
                np.cos(2.0 * np.pi * frequency_hz * time_s),
            )
        )
        coefficients, *_ = np.linalg.lstsq(design, signal, rcond=None)
        residual = signal - design @ coefficients
        return float(np.dot(residual, residual)), coefficients

    lower = max(1.0, peak_frequency_hz - search_half_width_hz)
    upper = min(sample_rate_hz / 2.0 - 1.0, peak_frequency_hz + search_half_width_hz)
    result = minimize_scalar(
        lambda frequency: solve(float(frequency))[0],
        bounds=(lower, upper),
        method="bounded",
        options={"xatol": 1e-5},
    )
    frequency_hz = float(result.x)
    _, coefficients = solve(frequency_hz)
    amplitude_px = float(np.hypot(coefficients[2], coefficients[3]))
    return frequency_hz, amplitude_px, float(result.fun)


def recover_disturbance(
    positions_or_estimates_xy: np.ndarray,
    config: ScenarioConfig,
    *,
    axis_index: int,
    true_frequency_hz: float,
) -> tuple[float, float, float, float, np.ndarray, np.ndarray]:
    """Recover a sinusoidal component from one x/y trajectory."""

    signal, valid = _filled_detrended_signal(positions_or_estimates_xy[:, axis_index])
    window = np.hanning(signal.size)
    frequencies = np.fft.rfftfreq(signal.size, d=1.0 / config.trajectory.frame_rate_hz)
    spectrum = 2.0 * np.abs(np.fft.rfft(signal * window)) / np.sum(window)
    # The OU drift is intentionally a low-frequency process.  Exclude that
    # band from disturbance identification so slow drift cannot be mistaken
    # for the injected periodic component.
    search = (frequencies >= 20.0) & (frequencies < config.trajectory.frame_rate_hz / 2.0)
    search[0] = False
    peak_index = int(np.argmax(np.where(search, spectrum, -np.inf)))
    peak_frequency_hz = float(frequencies[peak_index])
    resolution_hz = config.trajectory.frame_rate_hz / signal.size
    estimated_frequency_hz, estimated_amplitude_px, _ = _fit_sinusoid(
        signal,
        config.trajectory.frame_rate_hz,
        peak_frequency_hz,
        search_half_width_hz=max(1.0, 2.0 * resolution_hz),
    )
    noise_region = search.copy()
    noise_region &= np.abs(frequencies - estimated_frequency_hz) > 2.0 * resolution_hz
    noise_floor_px = float(np.median(spectrum[noise_region])) if np.any(noise_region) else float("nan")
    peak_to_noise = float(estimated_amplitude_px / noise_floor_px) if noise_floor_px > 0 else float("inf")
    return (
        estimated_frequency_hz,
        estimated_amplitude_px,
        noise_floor_px,
        peak_to_noise,
        frequencies,
        spectrum,
    )


def _summarize_dynamic(
    estimator: str,
    estimates_xy: np.ndarray,
    positions_xy: np.ndarray,
    config: ScenarioConfig,
    *,
    target_snr: float,
    lock_threshold_px: float,
) -> tuple[DynamicSummary, tuple[np.ndarray, np.ndarray]]:
    errors = estimates_xy - positions_xy
    valid = np.all(np.isfinite(errors), axis=1)
    radial_errors = np.linalg.norm(errors[valid], axis=1)
    lock = valid.copy()
    valid_indices = np.flatnonzero(valid)
    lock[valid_indices[radial_errors > lock_threshold_px]] = False
    failure_count = int(np.count_nonzero(~valid))
    lock_failure_count = int(np.count_nonzero(~lock))
    if radial_errors.size:
        rms_error = float(np.sqrt(np.mean(radial_errors**2)))
        p95_error = float(np.percentile(radial_errors, 95.0))
    else:
        rms_error = p95_error = float("nan")
    axis_index = 0 if config.trajectory.disturbance_axis == "x" else 1
    estimated_frequency, estimated_amplitude, noise_floor, peak_to_noise, frequencies, spectrum = recover_disturbance(
        estimates_xy,
        config,
        axis_index=axis_index,
        true_frequency_hz=config.trajectory.disturbance_frequency_hz,
    )
    summary = DynamicSummary(
        estimator=estimator,
        target_snr=float(target_snr),
        frame_count=positions_xy.shape[0],
        valid_count=int(np.count_nonzero(valid)),
        failure_count=failure_count,
        failure_rate=float(failure_count / positions_xy.shape[0]),
        lock_failure_count=lock_failure_count,
        lock_failure_rate=float(lock_failure_count / positions_xy.shape[0]),
        rms_error_px=rms_error,
        p95_error_px=p95_error,
        disturbance_axis=config.trajectory.disturbance_axis,
        true_frequency_hz=float(config.trajectory.disturbance_frequency_hz),
        estimated_frequency_hz=estimated_frequency,
        frequency_abs_error_hz=abs(estimated_frequency - config.trajectory.disturbance_frequency_hz),
        true_amplitude_px=float(config.trajectory.disturbance_amplitude_px),
        estimated_amplitude_px=estimated_amplitude,
        amplitude_abs_error_px=abs(estimated_amplitude - config.trajectory.disturbance_amplitude_px),
        spectral_noise_floor_px=noise_floor,
        peak_to_noise_ratio=peak_to_noise,
    )
    return summary, (frequencies, spectrum)


def run_dynamic_experiment(
    config: ScenarioConfig,
    *,
    target_snr: float | None = None,
    frame_count: int | None = None,
    lock_threshold_px: float = 1.0,
    estimators: tuple[str, ...] = ("centroid", "gaussian"),
) -> DynamicResult:
    """Run the shared-frame dynamic experiment for the requested estimators."""

    config.validate()
    snr = config.experiment.target_snr if target_snr is None else float(target_snr)
    count = config.trajectory.final_frame_count if frame_count is None else int(frame_count)
    if snr <= 0 or not np.isfinite(snr) or count <= 0 or lock_threshold_px <= 0:
        raise ValueError("invalid dynamic experiment parameters")
    positions, frames, saturation_counts = generate_dynamic_sequence(
        config,
        target_snr=snr,
        frame_count=count,
    )
    estimates: dict[str, np.ndarray] = {}
    summaries: list[DynamicSummary] = []
    spectra: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for estimator in estimators:
        estimates[estimator] = estimate_dynamic_trajectory(
            frames,
            config,
            estimator=estimator,
            warm_start=False,
        )
        summary, spectrum = _summarize_dynamic(
            estimator,
            estimates[estimator],
            positions,
            config,
            target_snr=snr,
            lock_threshold_px=lock_threshold_px,
        )
        summaries.append(summary)
        spectra[estimator] = spectrum
    return DynamicResult(positions, estimates, saturation_counts, summaries, spectra)


def save_dynamic_results(
    output_dir: str | Path,
    config: ScenarioConfig,
    result: DynamicResult,
    *,
    lock_threshold_px: float,
) -> tuple[Path, Path, Path]:
    """Save per-frame trajectory data and summary metadata."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    trajectory_path = destination / "dynamic_trajectory.csv"
    estimators = tuple(result.estimates_xy)
    fieldnames = [
        "frame",
        "time_s",
        "true_x_px",
        "true_y_px",
        "saturated_pixel_count",
    ]
    for estimator in estimators:
        fieldnames.extend((f"{estimator}_x_px", f"{estimator}_y_px", f"{estimator}_error_px"))
    with trajectory_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for index, position in enumerate(result.positions_xy):
            row: dict[str, float | int] = {
                "frame": index,
                "time_s": index / config.trajectory.frame_rate_hz,
                "true_x_px": float(position[0]),
                "true_y_px": float(position[1]),
                "saturated_pixel_count": int(result.saturation_counts[index]),
            }
            for estimator in estimators:
                estimate = result.estimates_xy[estimator][index]
                row[f"{estimator}_x_px"] = float(estimate[0])
                row[f"{estimator}_y_px"] = float(estimate[1])
                row[f"{estimator}_error_px"] = float(
                    np.linalg.norm(estimate - position)
                    if np.all(np.isfinite(estimate))
                    else np.nan
                )
            writer.writerow(row)

    summary_rows = [asdict(summary) for summary in result.summaries]
    summary_json_path = destination / "dynamic_summary.json"
    summary_csv_path = destination / "dynamic_summary.csv"
    save_json(
        summary_json_path,
        {
            "configuration": config,
            "lock_threshold_px": lock_threshold_px,
            "summaries": summary_rows,
        },
    )
    with summary_csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(summary_rows[0].keys()),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(summary_rows)
    return trajectory_path, summary_json_path, summary_csv_path


def plot_dynamic_results(
    config: ScenarioConfig,
    result: DynamicResult,
    output_path: str | Path,
    *,
    show: bool = False,
) -> Path:
    """Create trajectory, error, and disturbance-spectrum diagnostics."""

    import matplotlib.pyplot as plt

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    time_s = np.arange(result.positions_xy.shape[0]) / config.trajectory.frame_rate_hz
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    colors = {"centroid": "tab:blue", "gaussian": "tab:orange"}
    true_color = "#2a9d8f"
    labels = {"centroid": "Windowed centroid", "gaussian": "Gaussian fit"}
    axis_index = 0 if config.trajectory.disturbance_axis == "x" else 1
    axis_name = "x" if axis_index == 0 else "y"

    for index, coordinate in enumerate(("x", "y")):
        axis = axes[0, index]
        # Keep the known reference trajectory visually dominant even when an
        # estimator tracks it closely.  This makes agreement readable rather
        # than allowing the noisy estimates to obscure the ground truth.
        axis.plot(
            time_s,
            result.positions_xy[:, index],
            color=true_color,
            linestyle="-",
            linewidth=0.5,
            alpha=0.9,
            label="true",
            zorder=5,
        )
        for estimator, estimates in result.estimates_xy.items():
            axis.plot(
                time_s,
                estimates[:, index],
                color=colors[estimator],
                linewidth=0.9,
                alpha=0.55,
                label=labels[estimator],
                zorder=2,
            )
        axis.set_title(f"{coordinate}-coordinate trajectory")
        axis.set_xlabel("time [s]")
        axis.set_ylabel("position [pixel]")
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize="small")

    for estimator, estimates in result.estimates_xy.items():
        errors = np.linalg.norm(estimates - result.positions_xy, axis=1)
        errors[~np.all(np.isfinite(estimates), axis=1)] = np.nan
        axes[1, 0].plot(time_s, errors, color=colors[estimator], label=labels[estimator])
    axes[1, 0].set_title("Radial trajectory error")
    axes[1, 0].set_xlabel("time [s]")
    axes[1, 0].set_ylabel("error [pixel]")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend(fontsize="small")

    for estimator, (frequencies, spectrum) in result.spectra.items():
        visible = (frequencies >= 20.0) & (frequencies <= 150.0)
        axes[1, 1].plot(
            frequencies[visible],
            spectrum[visible],
            color=colors[estimator],
            label=labels[estimator],
        )
    axes[1, 1].axvline(
        config.trajectory.disturbance_frequency_hz,
        color=true_color,
        linestyle="--",
        label="true disturbance",
    )
    axes[1, 1].set_title(f"{axis_name}-axis disturbance spectrum (20–150 Hz)")
    axes[1, 1].set_xlabel("frequency [Hz]")
    axes[1, 1].set_ylabel("amplitude [pixel]")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend(fontsize="small")
    summary_by_estimator = {summary.estimator: summary for summary in result.summaries}
    recovery_lines = [
        f"true: f={config.trajectory.disturbance_frequency_hz:.2f} Hz, "
        f"A={config.trajectory.disturbance_amplitude_px:.3f} px"
    ]
    for estimator in result.estimates_xy:
        summary = summary_by_estimator[estimator]
        recovery_lines.append(
            f"{labels[estimator]}: f={summary.estimated_frequency_hz:.3f} Hz, "
            f"A={summary.estimated_amplitude_px:.4f} px"
        )
    axes[1, 1].text(
        0.02,
        0.97,
        "\n".join(recovery_lines),
        transform=axes[1, 1].transAxes,
        va="top",
        fontsize="small",
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "0.7"},
    )
    figure.suptitle("Dynamic sub-pixel trajectory recovery")
    figure.savefig(output, dpi=160)
    if show:
        plt.show()
    plt.close(figure)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("dynamic", "ideal", "sensor"), default="dynamic")
    parser.add_argument("--snr", type=float, default=10.0)
    parser.add_argument("--frames", type=int, default=None)
    parser.add_argument("--lock-threshold-px", type=float, default=1.0)
    parser.add_argument("--estimator", choices=("both", "centroid", "gaussian"), default="both")
    parser.add_argument("--output-dir", default="results/dynamic")
    parser.add_argument("--figure", default="results/standalone/figures/dynamic_tracking.png")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.show:
        import matplotlib

        matplotlib.use("Agg")
    config = scenario_config(args.profile)
    estimators = ("centroid", "gaussian") if args.estimator == "both" else (args.estimator,)
    result = run_dynamic_experiment(
        config,
        target_snr=args.snr,
        frame_count=args.frames,
        lock_threshold_px=args.lock_threshold_px,
        estimators=estimators,
    )
    save_dynamic_results(
        args.output_dir,
        config,
        result,
        lock_threshold_px=args.lock_threshold_px,
    )
    if not args.no_plot:
        plot_dynamic_results(config, result, args.figure, show=args.show)


if __name__ == "__main__":
    main()
