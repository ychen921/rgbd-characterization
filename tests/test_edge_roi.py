"""Tests for Scene 04 edge ROI configuration and masks."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.preprocessing.edge_roi import (
    EDGE_SCENE_TYPE,
    EdgeBleedingConfig,
    EdgeInvalidConfig,
    EdgeReferenceConfig,
    EdgeROIConfig,
    EdgeTransitionConfig,
    Line2D,
    build_roi_mask,
    load_edge_roi_config,
    save_edge_roi_config,
    validate_edge_roi_config,
)
from src.preprocessing.roi import RectROI


def _valid_config() -> EdgeROIConfig:
    return EdgeROIConfig(
        name="scene04_edge_d050",
        source_experiment="scene04_edge_d050_r01",
        source_frame_index=4,
        foreground_roi=RectROI(
            x=0,
            y=1,
            width=2,
            height=6,
        ),
        background_roi=RectROI(
            x=6,
            y=1,
            width=2,
            height=6,
        ),
        edge_roi=RectROI(
            x=3,
            y=1,
            width=2,
            height=6,
        ),
        nominal_edge=Line2D(
            p1=(4.0, 1.0),
            p2=(4.0, 7.0),
        ),
        foreground_side="left",
        distance_bin_px=2.0,
        max_edge_distance_px=30.0,
        reference=EdgeReferenceConfig(
            minimum_tolerance_mm=10.0,
            mad_scale=3.0,
        ),
        transition=EdgeTransitionConfig(
            high_probability=0.9,
            low_probability=0.1,
        ),
    )


def test_edge_roi_config_accepts_valid_configuration() -> None:
    config = _valid_config()

    validate_edge_roi_config(config, image_shape=(10, 10))

    assert config.name == "scene04_edge_d050"
    assert config.edge_roi.pixel_count == 12
    assert config.nominal_edge.p1 == (4.0, 1.0)


def test_save_load_edge_roi_config_round_trip(
    tmp_path: Path,
) -> None:
    path = tmp_path / "config" / "roi" / "scene04_edge_d050.yaml"
    expected = _valid_config()

    save_edge_roi_config(path, expected)
    loaded = load_edge_roi_config(path)

    assert loaded == expected

    with path.open("r", encoding="utf-8") as stream:
        document = yaml.safe_load(stream)

    assert list(document) == [
        "name",
        "scene_type",
        "source",
        "foreground_roi",
        "background_roi",
        "edge_roi",
        "nominal_edge",
        "analysis",
    ]
    assert document["scene_type"] == EDGE_SCENE_TYPE
    assert document["source"] == {
        "experiment": "scene04_edge_d050_r01",
        "frame_index": 4,
    }
    assert document["nominal_edge"] == {
        "type": "line",
        "p1": [4.0, 1.0],
        "p2": [4.0, 7.0],
        "foreground_side": "left",
    }
    assert document["analysis"]["reference"] == {
        "minimum_tolerance_mm": 10.0,
        "mad_scale": 3.0,
        "minimum_valid_ratio": 0.9,
        "minimum_valid_count": 100,
    }
    assert document["analysis"]["bleeding"] == {
        "probability_threshold": 0.05,
    }
    assert document["analysis"]["invalid"] == {
        "ratio_threshold": 0.5,
    }


def test_save_edge_roi_config_rejects_existing_file(
    tmp_path: Path,
) -> None:
    path = tmp_path / "scene04.yaml"
    path.write_text("existing: true\n", encoding="utf-8")

    with pytest.raises(FileExistsError):
        save_edge_roi_config(path, _valid_config())

    assert path.read_text(encoding="utf-8") == "existing: true\n"


def test_load_edge_roi_config_allows_trailing_empty_document(
    tmp_path: Path,
) -> None:
    path = tmp_path / "scene04.yaml"
    save_edge_roi_config(path, _valid_config())
    with path.open("a", encoding="utf-8") as stream:
        stream.write("---\n")

    assert load_edge_roi_config(path) == _valid_config()


def test_load_edge_roi_config_applies_new_threshold_defaults(
    tmp_path: Path,
) -> None:
    path = tmp_path / "scene04.yaml"
    save_edge_roi_config(path, _valid_config())
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    reference = document["analysis"]["reference"]
    reference.pop("minimum_valid_ratio")
    reference.pop("minimum_valid_count")
    document["analysis"].pop("bleeding")
    document["analysis"].pop("invalid")
    path.unlink()
    path.write_text(
        yaml.safe_dump(document, sort_keys=False),
        encoding="utf-8",
    )

    loaded = load_edge_roi_config(path)

    assert loaded.reference.minimum_valid_ratio == 0.9
    assert loaded.reference.minimum_valid_count == 100
    assert loaded.bleeding == EdgeBleedingConfig(
        probability_threshold=0.05
    )
    assert loaded.invalid == EdgeInvalidConfig(
        ratio_threshold=0.5
    )


def test_load_edge_roi_config_rejects_wrong_scene_type(
    tmp_path: Path,
) -> None:
    path = tmp_path / "scene04.yaml"
    save_edge_roi_config(path, _valid_config())
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    document["scene_type"] = "planar"
    path.unlink()
    path.write_text(
        yaml.safe_dump(document, sort_keys=False),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="scene_type"):
        load_edge_roi_config(path)


@pytest.mark.parametrize(
    ("section", "expected_message"),
    [
        ("source", "source"),
        ("foreground_roi", "foreground_roi"),
        ("nominal_edge", "nominal_edge"),
        ("analysis", "analysis"),
    ],
)
def test_load_edge_roi_config_rejects_missing_section(
    tmp_path: Path,
    section: str,
    expected_message: str,
) -> None:
    path = tmp_path / "scene04.yaml"
    save_edge_roi_config(path, _valid_config())
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    document.pop(section)
    path.unlink()
    path.write_text(
        yaml.safe_dump(document, sort_keys=False),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=expected_message):
        load_edge_roi_config(path)


def test_load_edge_roi_config_rejects_wrong_rectangle_type(
    tmp_path: Path,
) -> None:
    path = tmp_path / "scene04.yaml"
    save_edge_roi_config(path, _valid_config())
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    document["edge_roi"]["type"] = "ellipse"
    path.unlink()
    path.write_text(
        yaml.safe_dump(document, sort_keys=False),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="edge_roi type"):
        load_edge_roi_config(path)


@pytest.mark.parametrize(
    ("field_name", "roi", "expected_message"),
    [
        (
            "foreground_roi",
            RectROI(x=9, y=0, width=2, height=1),
            "foreground_roi exceeds image width",
        ),
        (
            "background_roi",
            RectROI(x=0, y=9, width=1, height=2),
            "background_roi exceeds image height",
        ),
        (
            "edge_roi",
            RectROI(x=9, y=0, width=2, height=1),
            "edge_roi exceeds image width",
        ),
    ],
)
def test_validate_edge_roi_config_rejects_out_of_bounds_roi(
    field_name: str,
    roi: RectROI,
    expected_message: str,
) -> None:
    config = replace(_valid_config(), **{field_name: roi})

    with pytest.raises(ValueError, match=expected_message):
        validate_edge_roi_config(config, image_shape=(10, 10))


def test_validate_edge_roi_config_rejects_line_missing_edge_roi() -> None:
    config = replace(
        _valid_config(),
        nominal_edge=Line2D(
            p1=(0.0, 8.0),
            p2=(2.0, 8.0),
        ),
        foreground_side="negative",
    )

    with pytest.raises(ValueError, match="does not intersect"):
        validate_edge_roi_config(config, image_shape=(10, 10))


@pytest.mark.parametrize(
    "foreground_side",
    ["above", "", "foreground"],
)
def test_edge_roi_config_rejects_unknown_foreground_side(
    foreground_side: str,
) -> None:
    with pytest.raises(ValueError, match="foreground_side"):
        replace(
            _valid_config(),
            foreground_side=foreground_side,
        )


def test_edge_roi_config_rejects_left_right_for_horizontal_line() -> None:
    with pytest.raises(ValueError, match="horizontal"):
        replace(
            _valid_config(),
            nominal_edge=Line2D(
                p1=(3.0, 4.0),
                p2=(5.0, 4.0),
            ),
            foreground_side="left",
        )


@pytest.mark.parametrize(
    ("constructor", "expected_message"),
    [
        (
            lambda: EdgeReferenceConfig(
                minimum_tolerance_mm=0.0,
                mad_scale=3.0,
            ),
            "minimum_tolerance_mm",
        ),
        (
            lambda: EdgeReferenceConfig(
                minimum_tolerance_mm=10.0,
                mad_scale=-1.0,
            ),
            "mad_scale",
        ),
        (
            lambda: EdgeReferenceConfig(
                minimum_tolerance_mm=10.0,
                mad_scale=3.0,
                minimum_valid_ratio=1.1,
            ),
            "minimum_valid_ratio",
        ),
        (
            lambda: EdgeReferenceConfig(
                minimum_tolerance_mm=10.0,
                mad_scale=3.0,
                minimum_valid_count=0,
            ),
            "minimum_valid_count",
        ),
        (
            lambda: EdgeBleedingConfig(
                probability_threshold=-0.1,
            ),
            "probability_threshold",
        ),
        (
            lambda: EdgeInvalidConfig(
                ratio_threshold=1.1,
            ),
            "ratio_threshold",
        ),
        (
            lambda: EdgeTransitionConfig(
                high_probability=1.0,
                low_probability=0.1,
            ),
            "high_probability",
        ),
        (
            lambda: EdgeTransitionConfig(
                high_probability=0.1,
                low_probability=0.9,
            ),
            "less than",
        ),
    ],
)
def test_analysis_config_rejects_invalid_parameters(
    constructor: object,
    expected_message: str,
) -> None:
    with pytest.raises(ValueError, match=expected_message):
        constructor()


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("distance_bin_px", 0.0),
        ("distance_bin_px", np.inf),
        ("max_edge_distance_px", -1.0),
        ("max_edge_distance_px", True),
    ],
)
def test_edge_roi_config_rejects_invalid_distance_parameters(
    field_name: str,
    value: object,
) -> None:
    with pytest.raises(ValueError, match=field_name):
        replace(_valid_config(), **{field_name: value})


def test_edge_roi_config_requires_symmetric_whole_distance_bins() -> None:
    with pytest.raises(ValueError, match="integer multiple"):
        replace(
            _valid_config(),
            distance_bin_px=4.0,
            max_edge_distance_px=30.0,
        )


@pytest.mark.parametrize(
    ("p1", "p2", "expected_message"),
    [
        ((1.0,), (2.0, 3.0), "two coordinates"),
        ((np.nan, 1.0), (2.0, 3.0), "finite"),
        ((1.0, 2.0), (1.0, 2.0), "non-zero length"),
    ],
)
def test_line2d_rejects_invalid_endpoints(
    p1: object,
    p2: object,
    expected_message: str,
) -> None:
    with pytest.raises(ValueError, match=expected_message):
        Line2D(p1=p1, p2=p2)


def test_build_roi_mask_returns_expected_boolean_mask() -> None:
    roi = RectROI(x=1, y=2, width=3, height=2)

    mask = build_roi_mask((5, 6), roi)

    expected = np.zeros((5, 6), dtype=bool)
    expected[2:4, 1:4] = True
    np.testing.assert_array_equal(mask, expected)
    assert mask.dtype == np.bool_
    assert np.count_nonzero(mask) == roi.pixel_count


@pytest.mark.parametrize(
    ("image_shape", "expected_message"),
    [
        ((5,), "height, width"),
        ((0, 5), "height"),
        ((5, -1), "width"),
        ((5, True), "width"),
    ],
)
def test_build_roi_mask_rejects_invalid_image_shape(
    image_shape: object,
    expected_message: str,
) -> None:
    with pytest.raises(ValueError, match=expected_message):
        build_roi_mask(
            image_shape,
            RectROI(x=0, y=0, width=1, height=1),
        )


def test_build_roi_mask_rejects_out_of_bounds_roi() -> None:
    with pytest.raises(ValueError, match="image width"):
        build_roi_mask(
            (5, 5),
            RectROI(x=4, y=0, width=2, height=1),
        )
