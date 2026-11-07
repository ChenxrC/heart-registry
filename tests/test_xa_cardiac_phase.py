from __future__ import annotations

import ctypes
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOLS = PROJECT_ROOT / "tools"
sys.path.insert(0, str(TOOLS))

import xa_cardiac_phase as phase
import analyze_xa_cardiac_phase as phase_cli

from xa_cardiac_phase import (
    centerline_counts_from_raw,
    confidence_status,
    coronary_motion_signal,
    estimate_anchor_frame,
    estimate_period,
    phase_confidence,
    phase_from_anchor,
    select_visible_frame_range,
)
from xa_multiseries import validate_phase_alignment_report


def test_visible_range_fills_two_frame_gap_and_keeps_reference() -> None:
    counts = [0, 1, 40, 50, 3, 2, 48, 45, 1]
    assert select_visible_frame_range(counts, reference_frame=3) == (2, 7)


def test_visible_range_falls_back_to_full_clip_when_no_signal() -> None:
    assert select_visible_frame_range([0, 0, 0], reference_frame=1) == (0, 2)


def test_centerline_counts_from_raw_counts_nonzero_pixels_per_frame() -> None:
    with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as temporary_dir:
        raw = Path(temporary_dir) / "temporal_centerlines_u8.raw"
        raw.write_bytes(bytes([0, 1, 2, 0, 0, 1, 0, 0, 0, 0, 9, 8]))

        assert centerline_counts_from_raw(raw, frame_count=3, rows=2, columns=2) == [2, 1, 2]


def test_period_estimator_recovers_eight_frame_cycle() -> None:
    signal = 1.0 - np.cos(2.0 * np.pi * np.arange(32) / 8.0)

    result = estimate_period(signal)

    assert result["status"] == "ok"
    assert abs(float(result["period_frames"]) - 8.0) <= 0.5


def test_period_estimator_rejects_short_nonperiodic_signal() -> None:
    result = estimate_period([0.1, 0.2, 0.1, 0.3])

    assert result["status"] == "insufficient"


def test_period_estimator_rejects_nonfinite_signal() -> None:
    result = estimate_period([0.0, 1.0, np.nan, 1.0, 0.0, 1.0])

    assert result["status"] == "insufficient"


@pytest.mark.parametrize(
    "signal",
    [np.zeros(32), np.full(32, 1.0e20)],
    ids=["zero", "large-offset"],
)
def test_period_estimator_rejects_constant_signal_at_any_scale(signal: np.ndarray) -> None:
    result = estimate_period(signal)

    assert result["status"] == "insufficient"
    assert "period_frames" not in result


def test_period_estimator_rejects_long_nonperiodic_quadratic_signal() -> None:
    signal = np.arange(32, dtype=np.float64) ** 2

    result = estimate_period(signal)

    assert result["status"] == "insufficient"
    assert "period_frames" not in result


@pytest.mark.parametrize("seed", [2, 710])
def test_period_estimator_rejects_long_nonperiodic_noise(seed: int) -> None:
    signal = np.random.default_rng(seed).normal(size=32)

    result = estimate_period(signal)

    assert result["status"] == "insufficient"
    assert "period_frames" not in result


@pytest.mark.parametrize("period, frame_count", [(3, 30), (16, 64)])
def test_period_estimator_recovers_candidate_range_boundaries(period: int, frame_count: int) -> None:
    signal = 1.0 - np.cos(2.0 * np.pi * np.arange(frame_count) / period)

    result = estimate_period(signal)

    assert result["status"] == "ok"
    assert float(result["period_frames"]) == pytest.approx(period, abs=0.5)


def test_period_estimator_recovers_fundamental_from_harmonic_rich_signal() -> None:
    frames = np.arange(48, dtype=np.float64)
    signal = (
        1.0
        - np.cos(2.0 * np.pi * frames / 8.0)
        + 0.40 * (1.0 - np.cos(4.0 * np.pi * frames / 8.0))
    )

    result = estimate_period(signal)

    assert result["status"] == "ok"
    assert float(result["period_frames"]) == pytest.approx(8.0, abs=0.5)


def test_phase_wraps_from_anchor() -> None:
    phase = phase_from_anchor(8, period_frames=4.0, anchor_frame=2)

    assert phase[2] == 0.0 and phase[4] == 0.5 and phase[6] == 0.0


def test_phase_rejects_nonfinite_period_and_invalid_anchor() -> None:
    with pytest.raises(ValueError, match="period_frames"):
        phase_from_anchor(8, period_frames=np.nan, anchor_frame=2)
    with pytest.raises(ValueError, match="anchor_frame"):
        phase_from_anchor(8, period_frames=4.0, anchor_frame=8)


def test_motion_signal_removes_global_translation_and_preserves_frame_shape() -> None:
    rng = np.random.default_rng(4)
    base = rng.integers(0, 256, size=(64, 64), dtype=np.uint8)
    cine = np.stack([np.roll(base, shift=frame, axis=1) for frame in range(6)])
    centerlines = np.zeros_like(cine, dtype=np.uint8)
    centerlines[:, 20:44, 31] = 1

    result = coronary_motion_signal(cine, centerlines, (0, 5))

    assert result["status"] == "ok"
    assert np.asarray(result["raw_signal"]).shape == (6,)
    assert np.asarray(result["smoothed_signal"]).shape == (6,)
    assert np.asarray(result["valid_fractions"]).shape == (6,)
    assert np.isfinite(result["raw_signal"]).all()
    assert float(np.max(result["raw_signal"])) < 0.25


def test_motion_signal_returns_insufficient_without_visible_centerline() -> None:
    cine = np.zeros((4, 32, 32), dtype=np.uint8)
    centerlines = np.zeros_like(cine)

    result = coronary_motion_signal(cine, centerlines, (0, 3))

    assert result["status"] == "insufficient"
    assert result["reason"] == "no-valid-centerline-motion"


def test_motion_signal_returns_insufficient_for_full_centerline_masks() -> None:
    rng = np.random.default_rng(7)
    cine = rng.integers(0, 256, size=(4, 31, 33), dtype=np.uint8)
    centerlines = np.ones_like(cine)

    result = coronary_motion_signal(cine, centerlines, (0, 3))

    assert result["status"] == "insufficient"
    assert result["reason"] == "stable-background-unavailable"


def test_atomic_phase_report_write_preserves_old_target_and_cleans_temp_on_replace_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "phase_alignment.json"
    target.write_text('{"previous": true}', encoding="utf-8")

    def fail_replace(source: str | Path, destination: str | Path) -> None:
        raise OSError("synthetic replace failure")

    monkeypatch.setattr(phase_cli.os, "replace", fail_replace)
    with pytest.raises(OSError, match="synthetic replace failure"):
        phase_cli.atomic_write_json(target, {"status": "insufficient"})

    assert target.read_text(encoding="utf-8") == '{"previous": true}'
    assert list(tmp_path.glob(".phase_alignment.json.*.tmp")) == []


def test_motion_signal_returns_insufficient_for_one_frame_visible_interval() -> None:
    cine = np.zeros((4, 31, 33), dtype=np.uint8)
    centerlines = np.zeros_like(cine)
    centerlines[:, 8:23, 16] = 1

    result = coronary_motion_signal(cine, centerlines, (2, 2))

    assert result["status"] == "insufficient"


def test_motion_signal_supports_odd_spatial_dimensions() -> None:
    rng = np.random.default_rng(8)
    base = rng.integers(0, 256, size=(33, 35), dtype=np.uint8)
    cine = np.stack([np.roll(base, shift=frame, axis=1) for frame in range(5)])
    centerlines = np.zeros_like(cine)
    centerlines[:, 9:24, 17] = 1

    result = coronary_motion_signal(cine, centerlines, (0, 4))

    assert result["status"] == "ok"
    assert np.asarray(result["raw_signal"]).shape == (5,)


@pytest.mark.parametrize(
    "centerlines",
    [
        np.full((2, 16, 16), "vessel", dtype=object),
        np.full((2, 16, 16), np.nan),
        np.full((2, 16, 16), np.inf),
        np.full((2, 16, 16), -1.0),
    ],
    ids=["object", "nan", "infinite", "negative"],
)
def test_motion_signal_rejects_invalid_centerline_values(centerlines: np.ndarray) -> None:
    cine = np.zeros((2, 16, 16), dtype=np.uint8)

    with pytest.raises(ValueError):
        coronary_motion_signal(cine, centerlines, (0, 1))


def test_motion_signal_validates_frame_shape_and_inclusive_range() -> None:
    cine = np.zeros((4, 16, 16), dtype=np.uint8)
    centerlines = np.zeros_like(cine)

    with pytest.raises(ValueError, match="cine"):
        coronary_motion_signal(cine[0], centerlines[0], (0, 0))
    with pytest.raises(ValueError, match="same shape"):
        coronary_motion_signal(cine, centerlines[:, :-1], (0, 3))
    with pytest.raises(ValueError, match="visible_range"):
        coronary_motion_signal(cine, centerlines, (1, 4))


def test_confidence_formula_and_status_thresholds() -> None:
    confidence = phase_confidence(
        periodicity=0.8,
        visibility=0.6,
        tracking_stability=0.4,
        observable_cycles=1.5,
    )

    assert confidence == pytest.approx(0.625)
    assert confidence_status(0.4999) == "insufficient"
    assert confidence_status(0.50) == "provisional"
    assert confidence_status(0.7999) == "provisional"
    assert confidence_status(0.80) == "automatic"


@pytest.mark.parametrize(
    "arguments",
    [
        (-0.01, 0.5, 0.5, 2.0),
        (1.01, 0.5, 0.5, 2.0),
        (0.5, -0.01, 0.5, 2.0),
        (0.5, 1.01, 0.5, 2.0),
        (0.5, 0.5, -0.01, 2.0),
        (0.5, 0.5, 1.01, 2.0),
        (0.5, 0.5, 0.5, -0.01),
    ],
)
def test_phase_confidence_rejects_out_of_range_inputs(arguments: tuple[float, ...]) -> None:
    with pytest.raises(ValueError):
        phase_confidence(*arguments)


def test_anchor_selection_uses_visible_low_motion_frame_with_adequate_tracking() -> None:
    smoothed_signal = [9.0, 8.0, 4.0, 3.0, 0.0, 2.0, 5.0, 8.0, 9.0]
    valid_fractions = [1.0] * len(smoothed_signal)
    valid_fractions[4] = 0.25

    anchor = estimate_anchor_frame(
        smoothed_signal,
        visible_range=(1, 7),
        reference_frame=4,
        period_frames=6.0,
        valid_fractions=valid_fractions,
    )

    assert anchor == 5


def test_anchor_selection_validates_period_and_reference_bounds() -> None:
    signal = [3.0, 2.0, 1.0, 2.0]

    with pytest.raises(ValueError, match="period_frames"):
        estimate_anchor_frame(signal, (0, 3), 1, np.inf)
    with pytest.raises(ValueError, match="reference_frame"):
        estimate_anchor_frame(signal, (0, 3), 4, 3.0)


@pytest.mark.parametrize(
    "invalid_fraction",
    [np.nan, np.inf, -0.01, 1.01],
    ids=["nan", "infinite", "negative", "above-one"],
)
def test_anchor_selection_rejects_invalid_tracking_fractions(invalid_fraction: float) -> None:
    signal = [3.0, 2.0, 1.0, 2.0]
    valid_fractions = [1.0, invalid_fraction, 1.0, 1.0]

    with pytest.raises(ValueError):
        estimate_anchor_frame(signal, (0, 3), 1, 3.0, valid_fractions)


def _write_phase_run_fixture(
    series_root: Path,
    series_number: int,
    *,
    visible_start: int,
    phase_offset: int,
    frame_count: int = 32,
    period: int = 8,
) -> None:
    """Write small real binary/JSON inputs with periodic local image motion."""
    run_id = f"series_{series_number:02d}"
    series_dir = series_root / run_id
    temporal_dir = series_dir / "temporal"
    dynamic_dir = series_dir / "dynamic_3d"
    temporal_dir.mkdir(parents=True)
    dynamic_dir.mkdir()

    rows = columns = 48
    rng = np.random.default_rng(100 + series_number)
    background = rng.integers(0, 96, size=(rows, columns), dtype=np.uint8)
    texture = rng.integers(112, 256, size=(24, 24), dtype=np.uint8)
    cine = np.repeat(background[None, ...], frame_count, axis=0)
    centerlines = np.zeros_like(cine)
    for frame_index in range(frame_count):
        shift = int(round(3.0 * np.sin(2.0 * np.pi * (frame_index - phase_offset) / period)))
        cine[frame_index, 12:36, 12:36] = np.roll(texture, shift=shift, axis=1)
        if frame_index >= visible_start:
            centerlines[frame_index, 16:32, 24 + shift] = 255

    cine_path = series_dir / "cine_u8.raw"
    centerline_path = temporal_dir / "temporal_centerlines_u8.raw"
    cine.tofile(cine_path)
    centerlines.tofile(centerline_path)
    manifest = {
        "schema_version": 1,
        "source_dicom": f"source-{run_id}.dcm",
        "source_record": {
            "rows": rows,
            "columns": columns,
            "frame_count": frame_count,
            "sha256": f"{series_number:064x}",
        },
        "frame_png_count": frame_count,
        "shape_frame_row_column": [frame_count, rows, columns],
        "raw_stack": {"path": "cine_u8.raw", "byte_length": cine_path.stat().st_size},
    }
    (series_dir / "series_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (dynamic_dir / "dynamic_vessel_3d.json").write_text(
        json.dumps({"schema_version": 1, "run_id": run_id, "reference_frame": visible_start + period}),
        encoding="utf-8",
    )


def test_partial_dtw_matches_shifted_sequence_and_skips_outlier() -> None:
    reference = np.asarray([[0.0], [1.0], [0.0], [-1.0], [0.0]])
    target = np.asarray([[9.0], [0.0], [1.0], [0.0], [-1.0], [0.0]])

    pairs = phase.partial_dtw(reference, target, skip_penalty=0.5)

    assert pairs[0] == (0, 1)
    assert pairs[-1] == (4, 5)
    assert all(a[0] < b[0] and a[1] < b[1] for a, b in zip(pairs, pairs[1:]))


def test_partial_dtw_returns_empty_when_all_skips_cost_less_than_any_match() -> None:
    pairs = phase.partial_dtw(
        np.asarray([[0.0]]),
        np.asarray([[1.0]]),
        skip_penalty=0.25,
    )

    assert pairs == []


def test_phase_report_rejects_group_when_partial_dtw_uses_all_skips(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as temporary_dir:
        series_root = Path(temporary_dir)
        for series_number in range(1, 7):
            _write_phase_run_fixture(
                series_root,
                series_number,
                visible_start=2 + series_number % 3,
                phase_offset=series_number,
            )
        actual_partial_dtw = phase.partial_dtw

        def all_skip_partial_dtw(
            reference_features: np.ndarray,
            target_features: np.ndarray,
            skip_penalty: float = 0.25,
        ) -> list[tuple[int, int]]:
            return actual_partial_dtw(
                reference_features,
                target_features,
                skip_penalty=1.0e-12,
            )

        monkeypatch.setattr(phase, "partial_dtw", all_skip_partial_dtw)
        report = phase.build_phase_alignment(series_root)

    assert report["status"] == "insufficient"
    assert report["frame_correspondence"] == {}
    for group_name in ("rca", "lca"):
        assert report["groups"][group_name]["status"] == "insufficient"
        assert report["groups"][group_name]["reference_run_id"] is None
        assert report["groups"][group_name]["reason"] == "no-credible-frame-correspondence"
    validate_phase_alignment_report(
        report,
        {f"series_{series_number:02d}": 32 for series_number in range(1, 7)},
    )


def test_phase_report_maps_local_dtw_indices_to_visible_frames_without_cross_group_pairs() -> None:
    with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as temporary_dir:
        series_root = Path(temporary_dir)
        _write_phase_run_fixture(series_root, 1, visible_start=2, phase_offset=0)
        _write_phase_run_fixture(series_root, 2, visible_start=5, phase_offset=2)

        report = phase.build_phase_alignment(series_root)

    assert report["status"] == "partial"
    assert set(report["runs"]) == {f"series_{index:02d}" for index in range(1, 7)}
    assert set(report["groups"]["rca"]["run_ids"]) == {"series_01", "series_02"}
    assert set(report["groups"]["lca"]["run_ids"]) == {
        "series_03", "series_04", "series_05", "series_06"
    }
    assert report["groups"]["rca"]["status"] in {"automatic", "provisional"}
    assert report["groups"]["lca"]["status"] == "insufficient"
    assert "series_07" not in json.dumps(report)

    correspondence = next(iter(report["frame_correspondence"].values()))
    assert correspondence["group"] == "rca"
    assert {correspondence["reference_run_id"], correspondence["target_run_id"]} == {
        "series_01", "series_02"
    }
    starts = {
        run_id: report["runs"][run_id]["visible_frame_range"][0]
        for run_id in ("series_01", "series_02")
    }
    for reference_frame, target_frame in correspondence["pairs"]:
        assert reference_frame >= starts[correspondence["reference_run_id"]]
        assert target_frame >= starts[correspondence["target_run_id"]]


def test_phase_report_keeps_malformed_runs_insufficient_and_cli_still_writes_output() -> None:
    with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as temporary_dir:
        series_root = Path(temporary_dir) / "series"
        series_root.mkdir()
        malformed = series_root / "series_01"
        malformed.mkdir()
        (malformed / "series_manifest.json").write_text("{not-json", encoding="utf-8")
        output = Path(temporary_dir) / "out" / "phase_alignment.json"

        completed = subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "analyze_xa_cardiac_phase.py"),
                "--series-root",
                str(series_root),
                "--output",
                str(output),
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

        assert completed.returncode == 0, completed.stderr
        report = json.loads(output.read_text(encoding="utf-8"))
        assert report["status"] == "insufficient"
        assert report["runs"]["series_01"]["status"] == "insufficient"
        assert report["runs"]["series_01"]["error"]
        assert report["runs"]["series_06"]["reason"] == "missing-run-directory"


def test_phase_report_isolates_real_manifest_hash_failure_to_one_run() -> None:
    if sys.platform != "win32":
        pytest.skip("This regression uses Windows exclusive file-sharing semantics")

    with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as temporary_dir:
        series_root = Path(temporary_dir)
        for series_number in range(1, 7):
            _write_phase_run_fixture(
                series_root,
                series_number,
                visible_start=2 + series_number % 3,
                phase_offset=series_number,
            )

        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateFileW.argtypes = [
            wintypes.LPCWSTR,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.LPVOID,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.HANDLE,
        ]
        kernel32.CreateFileW.restype = wintypes.HANDLE
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL
        locked_manifest = series_root / "series_01" / "series_manifest.json"
        handle = kernel32.CreateFileW(
            str(locked_manifest),
            0x80000000,
            0,
            None,
            3,
            0x80,
            None,
        )
        assert handle != ctypes.c_void_p(-1).value
        try:
            report = phase.build_phase_alignment(series_root)
        finally:
            assert kernel32.CloseHandle(handle)

    assert set(report["runs"]) == {f"series_{number:02d}" for number in range(1, 7)}
    failed = report["runs"]["series_01"]
    assert failed["status"] == "insufficient"
    assert failed["reason"] == "invalid-run-input"
    assert "PermissionError" in failed["error"]
    for series_number in range(2, 7):
        run = report["runs"][f"series_{series_number:02d}"]
        assert run["visible_frame_range"] is not None
        assert set(run["inputs"]) >= {
            "manifest", "cine", "temporal_centerlines", "dynamic_vessel_3d"
        }
