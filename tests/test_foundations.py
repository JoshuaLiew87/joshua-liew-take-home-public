from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from common.rng import make_rng_streams
from config import NoiseConfig, scenario_config, sigma_from_1e2_diameter
from sim.noise import convert_to_adc, make_fixed_pattern_maps, sample_poisson, sample_read_noise
from sim.psf import render_pixel_integrated_gaussian
from sim.simulator import (
    SpotSimulator,
    render_background,
    reference_snr_for_total_flux,
    total_flux_for_target_snr,
)


def test_spot_width_conversion() -> None:
    assert sigma_from_1e2_diameter(7.0) == 1.75


def test_pixel_integrated_gaussian_conserves_flux_on_wide_frame() -> None:
    image = render_pixel_integrated_gaussian(
        shape=(64, 64),
        position_xy=(31.25, 30.75),
        total_flux_e=12345.0,
        sigma_px=1.75,
    )

    assert image.shape == (64, 64)
    assert np.all(image >= 0.0)
    np.testing.assert_allclose(image.sum(), 12345.0, rtol=0.0, atol=1e-10)


def test_pixel_integrated_gaussian_is_symmetric_at_integer_center() -> None:
    image = render_pixel_integrated_gaussian(
        shape=(33, 33),
        position_xy=(16.0, 16.0),
        total_flux_e=1000.0,
        sigma_px=1.75,
    )

    # ndtr subtraction introduces tiny asymmetric round-off in the far tails.
    np.testing.assert_allclose(image, image[::-1, :], rtol=0.0, atol=3e-14)
    np.testing.assert_allclose(image, image[:, ::-1], rtol=0.0, atol=3e-14)


def test_subpixel_position_changes_expected_image() -> None:
    first = render_pixel_integrated_gaussian((32, 32), (15.1, 15.2), 1000.0, 1.75)
    second = render_pixel_integrated_gaussian((32, 32), (15.6, 15.2), 1000.0, 1.75)

    assert not np.array_equal(first, second)
    np.testing.assert_allclose(first.sum(), second.sum(), rtol=0.0, atol=1e-12)


def test_scenario_profiles_validate() -> None:
    for profile in ("ideal", "sensor", "dynamic"):
        config = scenario_config(profile)
        config.validate()


def test_rng_streams_are_reproducible_and_independent() -> None:
    first = make_rng_streams(1234)
    second = make_rng_streams(1234)

    for name in (
        "spot",
        "positions",
        "shot_noise",
        "read_noise",
        "fixed_pattern",
        "trajectory",
    ):
        first_values = getattr(first, name).normal(size=8)
        second_values = getattr(second, name).normal(size=8)
        np.testing.assert_array_equal(first_values, second_values)


def test_fixed_pattern_maps_are_persistent_and_have_expected_hot_pixels() -> None:
    rng = np.random.default_rng(5)
    maps = make_fixed_pattern_maps(
        shape=(32, 32),
        prnu_std_fraction=0.01,
        dark_current_e_per_s=50.0,
        hot_pixel_count=4,
        hot_pixel_dark_current_e_per_s=5000.0,
        rng=rng,
    )

    assert maps.prnu_gain.shape == (32, 32)
    assert maps.dark_current_e_per_s.shape == (32, 32)
    assert np.count_nonzero(maps.hot_pixel_mask) == 4
    np.testing.assert_array_equal(
        maps.dark_current_e_per_s[maps.hot_pixel_mask], 5000.0
    )


def test_poisson_and_read_noise_statistics() -> None:
    rng = np.random.default_rng(6)
    poisson_samples = np.concatenate(
        [sample_poisson(np.full((1000,), 40.0), rng) for _ in range(20)]
    )
    assert abs(float(poisson_samples.mean()) - 40.0) < 0.7
    assert abs(float(poisson_samples.var()) - 40.0) < 2.0

    read_samples = np.concatenate(
        [sample_read_noise((1000,), 3.0, rng) for _ in range(20)]
    )
    assert abs(float(read_samples.mean())) < 0.08
    assert abs(float(read_samples.std()) - 3.0) < 0.08


def test_snr_control_maps_monotonically_to_total_flux() -> None:
    config = scenario_config("ideal")
    simulator = SpotSimulator(config)
    fluxes = [
        total_flux_for_target_snr(config, simulator.maps, snr)
        for snr in (1.0, 10.0, 100.0)
    ]

    assert fluxes[0] < fluxes[1] < fluxes[2]
    for snr, flux in zip((1.0, 10.0, 100.0), fluxes):
        assert reference_snr_for_total_flux(config, simulator.maps, flux, 1.75) == pytest.approx(snr)


def test_simulator_is_reproducible_and_returns_diagnostics() -> None:
    first = SpotSimulator(scenario_config("sensor"))
    second = SpotSimulator(scenario_config("sensor"))
    position = (15.25, 16.4)
    first_frame = first.render(position, target_snr=10.0, sigma_px=1.75)
    second_frame = second.render(position, target_snr=10.0, sigma_px=1.75)

    np.testing.assert_array_equal(first_frame.image_dn, second_frame.image_dn)
    assert first_frame.metadata.target_snr == 10.0
    assert first_frame.metadata.reference_snr == pytest.approx(10.0)
    assert first_frame.metadata.effective_snr > 0.0
    assert first_frame.image_dn.shape == (32, 32)


def test_adc_quantization_and_saturation() -> None:
    config = scenario_config("sensor")
    result = convert_to_adc(
        np.array([[-10.0, 0.49, 1.51, 5000.0]]), config.sensor
    )

    np.testing.assert_array_equal(result.image_dn, np.array([[0.0, 0.0, 2.0, 4095.0]]))
    np.testing.assert_array_equal(
        result.saturated_mask, np.array([[True, False, False, True]])
    )


def test_adc_chain_respects_nonunity_electrons_per_dn() -> None:
    base = scenario_config("sensor")
    sensor = replace(
        base.sensor,
        electrons_per_dn=2.0,
        full_well_e=7000.0,
        quantization_enabled=False,
    )
    result = convert_to_adc(np.array([[6999.0, 7001.0, 8200.0]]), sensor)

    np.testing.assert_allclose(result.image_dn, np.array([[3499.5, 3500.0, 3500.0]]))
    np.testing.assert_array_equal(
        result.saturated_mask, np.array([[False, True, True]])
    )

    adc_limited_sensor = replace(sensor, full_well_e=9000.0)
    adc_limited = convert_to_adc(np.array([[8191.0]]), adc_limited_sensor)
    np.testing.assert_allclose(adc_limited.image_dn, np.array([[4095.0]]))
    np.testing.assert_array_equal(adc_limited.saturated_mask, np.array([[True]]))


def test_noise_switches_can_disable_stochastic_components() -> None:
    base = scenario_config("sensor")
    deterministic = replace(
        base,
        noise=NoiseConfig(
            photon_shot_enabled=False,
            read_noise_enabled=False,
            dark_current_enabled=False,
            background_enabled=False,
            prnu_enabled=False,
            hot_pixels_enabled=False,
        ),
        sensor=replace(
            base.sensor,
            quantization_enabled=False,
            saturation_enabled=False,
        ),
    )
    frame = SpotSimulator(deterministic).render(
        (15.25, 16.4), total_flux_e=1000.0, sigma_px=1.75
    )

    np.testing.assert_allclose(
        frame.sampled_charge_e,
        frame.expected_charge_after_prnu_e,
        rtol=0.0,
        atol=1e-12,
    )


def test_individual_noise_switches_isolate_expected_components() -> None:
    base = scenario_config("sensor")
    position = (15.25, 16.4)
    total_flux = 1000.0

    background_off = replace(
        base, noise=replace(base.noise, background_enabled=False)
    )
    background_frame = SpotSimulator(background_off).render(
        position, total_flux_e=total_flux, sigma_px=1.75
    )
    assert np.all(background_frame.expected_background_e == 0.0)

    dark_off = replace(base, noise=replace(base.noise, dark_current_enabled=False))
    dark_frame = SpotSimulator(dark_off).render(
        position, total_flux_e=total_flux, sigma_px=1.75
    )
    assert np.all(dark_frame.expected_dark_e == 0.0)

    hot_off = replace(base, noise=replace(base.noise, hot_pixels_enabled=False))
    hot_frame = SpotSimulator(hot_off).render(
        position, total_flux_e=total_flux, sigma_px=1.75
    )
    expected_baseline_dark = base.pattern.dark_current_e_per_s * base.sensor.exposure_s
    np.testing.assert_allclose(hot_frame.expected_dark_e, expected_baseline_dark)

    prnu_off = replace(base, noise=replace(base.noise, prnu_enabled=False))
    prnu_frame = SpotSimulator(prnu_off).render(
        position, total_flux_e=total_flux, sigma_px=1.75
    )
    np.testing.assert_allclose(
        prnu_frame.expected_charge_after_prnu_e,
        prnu_frame.expected_signal_e
        + prnu_frame.expected_background_e
        + prnu_frame.expected_dark_e,
        rtol=0.0,
        atol=1e-12,
    )


def test_background_gradient_has_documented_orientation() -> None:
    config = scenario_config("sensor")
    background = render_background((32, 32), config)

    assert background[16, -1] > background[16, 0]
    # The configured y-gradient is negative and image y increases downward.
    assert background[-1, 16] < background[0, 16]


def test_dark_current_scales_with_exposure() -> None:
    base = scenario_config("sensor")
    longer_exposure = replace(
        base,
        sensor=replace(base.sensor, exposure_s=2.0 * base.sensor.exposure_s),
    )
    short_frame = SpotSimulator(base).render(
        (15.25, 16.4), total_flux_e=0.0, sigma_px=1.75
    )
    long_frame = SpotSimulator(longer_exposure).render(
        (15.25, 16.4), total_flux_e=0.0, sigma_px=1.75
    )

    np.testing.assert_allclose(
        long_frame.expected_dark_e,
        2.0 * short_frame.expected_dark_e,
        rtol=0.0,
        atol=1e-12,
    )


def test_composed_pipeline_matches_poisson_statistics() -> None:
    base = scenario_config("sensor")
    poisson_only = replace(
        base,
        noise=replace(
            base.noise,
            read_noise_enabled=False,
            prnu_enabled=False,
            hot_pixels_enabled=False,
        ),
    )
    simulator = SpotSimulator(poisson_only)
    frames = [
        simulator.render((15.25, 16.4), total_flux_e=1000.0, sigma_px=1.75)
        for _ in range(1000)
    ]
    expected = frames[0].expected_charge_after_prnu_e[16, 15]
    samples = np.array([frame.sampled_charge_e[16, 15] for frame in frames])

    assert abs(float(samples.mean()) - expected) < 0.7
    # The sample variance itself has noticeable finite-N uncertainty for a
    # Poisson variable; this tolerance is deliberately wider than the mean
    # tolerance and scales with the expected variance.
    assert abs(float(samples.var()) - expected) < max(8.0, 0.2 * expected)


def test_composed_pipeline_matches_read_noise_statistics() -> None:
    base = scenario_config("sensor")
    read_only = replace(
        base,
        noise=replace(
            base.noise,
            photon_shot_enabled=False,
            dark_current_enabled=False,
            background_enabled=False,
            prnu_enabled=False,
            hot_pixels_enabled=False,
        ),
    )
    simulator = SpotSimulator(read_only)
    frames = [
        simulator.render((15.25, 16.4), total_flux_e=1000.0, sigma_px=1.75)
        for _ in range(1000)
    ]
    expected = frames[0].expected_charge_after_prnu_e[16, 15]
    samples = np.array([frame.sampled_charge_e[16, 15] for frame in frames])
    residual = samples - expected

    assert abs(float(residual.mean())) < 0.35
    assert abs(float(residual.std()) - base.sensor.read_noise_e_rms) < 0.25


def test_sensor_output_is_integer_and_within_adc_range() -> None:
    config = scenario_config("sensor")
    frame = SpotSimulator(config).render(
        (15.25, 16.4), total_flux_e=1000.0, sigma_px=1.75
    )

    assert np.issubdtype(frame.image_dn.dtype, np.integer)
    assert frame.image_dn.min() >= 0
    assert frame.image_dn.max() <= config.sensor.adc_max_dn
    assert frame.metadata.saturated_pixel_count == int(
        np.count_nonzero(frame.saturated_mask)
    )


def test_edge_spot_is_finite_and_reports_cropped_flux() -> None:
    config = scenario_config("ideal")
    frame = SpotSimulator(config).render(
        (0.0, 0.0), total_flux_e=1000.0, sigma_px=1.75
    )

    assert np.all(np.isfinite(frame.expected_signal_e))
    assert np.all(frame.expected_signal_e >= 0.0)
    assert frame.expected_signal_e.sum() < 1000.0


def test_frame_metadata_matches_rendered_frame() -> None:
    config = scenario_config("sensor")
    position = (15.25, 16.4)
    frame = SpotSimulator(config).render(
        position, target_snr=10.0, sigma_px=1.75
    )
    metadata = frame.metadata
    peak_y, peak_x = np.unravel_index(
        np.argmax(frame.expected_signal_e), frame.expected_signal_e.shape
    )

    assert metadata.position_xy == position
    assert metadata.peak_pixel_xy == (int(peak_x), int(peak_y))
    assert metadata.saturated_pixel_count == int(np.count_nonzero(frame.saturated_mask))
    assert metadata.reference_snr == pytest.approx(10.0)
