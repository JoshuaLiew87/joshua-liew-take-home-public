"""Visual sanity check for the Phase 1 simulator.

Examples
--------
Display a sensor-profile frame::

    python -m experiments.inspect_simulator --profile sensor --snr 10

Save a headless diagnostic figure::

    python -m experiments.inspect_simulator --profile sensor --snr 10 \
        --output results/standalone/figures/simulator_sanity.png --no-show
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from config import scenario_config
from sim.simulator import SpotSimulator


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        choices=("ideal", "sensor", "dynamic"),
        default="sensor",
        help="simulator scenario profile",
    )
    parser.add_argument("--snr", type=float, default=10.0, help="target reference SNR")
    parser.add_argument("--x", type=float, default=15.25, help="spot x position")
    parser.add_argument("--y", type=float, default=16.40, help="spot y position")
    parser.add_argument("--sigma", type=float, default=None, help="override spot sigma")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="optional output PNG path; parent directories are created",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="do not open an interactive Matplotlib window",
    )
    return parser


def make_figure(simulator: SpotSimulator, frame) -> plt.Figure:
    """Create the four-panel simulator diagnostic figure."""

    metadata = frame.metadata
    residual = frame.sampled_charge_e - frame.expected_charge_after_prnu_e
    background_and_dark = frame.expected_background_e + frame.expected_dark_e
    figure, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)

    panels = (
        (axes[0, 0], frame.expected_signal_e, "Expected integrated spot [e-]", "magma"),
        (axes[0, 1], background_and_dark, "Expected background + dark [e-]", "viridis"),
        (axes[1, 0], frame.image_dn, "Sampled ADC frame [DN]", "gray"),
        (axes[1, 1], residual, "Charge residual [e-]", "coolwarm"),
    )
    for axis, image, title, cmap in panels:
        display = axis.imshow(image, origin="lower", cmap=cmap, interpolation="nearest")
        axis.set_title(title)
        axis.set_xlabel("x [pixel]")
        axis.set_ylabel("y [pixel]")
        axis.plot(metadata.position_xy[0], metadata.position_xy[1], "c+", ms=10, mew=1.5)
        axis.plot(
            metadata.peak_pixel_xy[0],
            metadata.peak_pixel_xy[1],
            "rx",
            ms=8,
            mew=1.5,
        )
        figure.colorbar(display, ax=axis, shrink=0.8)

    figure.suptitle(
        f"{simulator.config.name} profile | target SNR={metadata.target_snr:.3g} | "
        f"reference SNR={metadata.reference_snr:.3g} | "
        f"effective SNR={metadata.effective_snr:.3g}\n"
        f"sigma={metadata.sigma_px:.3f}px | flux={metadata.total_flux_e:.3g}e- | "
        f"saturated pixels={metadata.saturated_pixel_count}",
        fontsize=10,
    )
    return figure


def main() -> None:
    args = build_parser().parse_args()
    config = scenario_config(args.profile)
    simulator = SpotSimulator(config)
    frame = simulator.render(
        (args.x, args.y),
        target_snr=args.snr,
        sigma_px=args.sigma,
    )
    figure = make_figure(simulator, frame)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(args.output, dpi=150)
        print(f"saved {args.output}")
    if not args.no_show:
        plt.show()
    else:
        plt.close(figure)


if __name__ == "__main__":
    main()
