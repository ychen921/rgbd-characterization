"""Integration tests for the Phase 0 alignment recorder preflight."""

import os
from pathlib import Path
import subprocess

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RECORDER = PROJECT_ROOT / "scripts" / "record_alignment_experiment.sh"


def install_fake_ros2(bin_dir: Path) -> Path:
    """Install a deterministic ros2 stand-in and return its command log."""
    bin_dir.mkdir()
    command_log = bin_dir / "ros2.log"
    fake_ros2 = bin_dir / "ros2"
    fake_ros2.write_text(
        """#!/usr/bin/env bash
set -euo pipefail

printf '%s\\n' "$*" >> "${FAKE_ROS2_LOG}"

if [[ "$1 $2" == "topic list" ]]; then
    cat <<'EOF'
/camera/color/image_raw
/camera/color/camera_info
/camera/depth/image_raw
/camera/depth/camera_info
/camera/depth/image_unaligned
/tf_static
EOF
    exit 0
fi

if [[ "$1 $2" == "param dump" ]]; then
    cat <<'EOF'
/camera/camera:
  ros__parameters:
    depth_registration: true
    align_mode: SW
    align_target_stream: COLOR
    time_domain: global
EOF
    exit 0
fi

if [[ "$1 $2" == "param get" ]]; then
    parameter="$4"
    case "${parameter}" in
        depth_registration)
            printf 'Boolean value is: %s\\n' "${FAKE_REGISTRATION:-True}"
            ;;
        align_mode)
            printf 'String value is: SW\\n'
            ;;
        align_target_stream)
            printf 'String value is: COLOR\\n'
            ;;
        time_domain)
            printf 'String value is: global\\n'
            ;;
        depth_precision)
            printf 'String value is: %s\\n' "${FAKE_DEPTH_PRECISION-1mm}"
            ;;
        *)
            printf 'Unknown parameter: %s\\n' "${parameter}" >&2
            exit 1
            ;;
    esac
    exit 0
fi

if [[ "$1 $2" == "topic echo" ]]; then
    topic="$3"
    field=""
    shift 3
    while (( $# > 0 )); do
        if [[ "$1" == "--field" ]]; then
            field="$2"
            break
        fi
        shift
    done

    case "${field}" in
        height)
            printf 'data: 720\\n'
            ;;
        width)
            printf 'data: 1280\\n'
            ;;
        encoding)
            if [[ "${topic}" == "/camera/color/image_raw" ]]; then
                printf "data: 'rgb8'\\n"
            else
                printf "data: '16UC1'\\n"
            fi
            ;;
        header.frame_id)
            printf "data: 'camera_color_optical_frame'\\n"
            ;;
        *)
            exit 1
            ;;
    esac
    exit 0
fi

if [[ "$1 $2" == "bag record" ]]; then
    output_dir=""
    shift 2
    while (( $# > 0 )); do
        if [[ "$1" == "-o" ]]; then
            output_dir="$2"
            break
        fi
        shift
    done
    mkdir -p "${output_dir}"
    : > "${output_dir}/metadata.yaml"
    exit 0
fi

printf 'Unsupported fake ros2 invocation: %s\\n' "$*" >&2
exit 1
""",
        encoding="utf-8",
    )
    fake_ros2.chmod(0o755)
    return command_log


def recorder_environment(tmp_path: Path, command_log: Path) -> dict[str, str]:
    """Return an isolated recorder environment using the fake ros2 command."""
    environment = os.environ.copy()
    environment.update(
        {
            "BAG_ROOT": str(tmp_path / "bags"),
            "RESULTS_ROOT": str(tmp_path / "results"),
            "FAKE_ROS2_LOG": str(command_log),
            "PATH": f"{command_log.parent}:{environment['PATH']}",
        }
    )
    return environment


def recorder_command(*extra_args: str) -> list[str]:
    """Build a valid d100-center recorder command."""
    return [
        str(RECORDER),
        "--distance-mm",
        "1000",
        "--position",
        "center",
        "--yaw-deg",
        "0",
        "--repeat",
        "1",
        *extra_args,
    ]


def test_warn_preflight_records_only_discovered_topics(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    command_log = install_fake_ros2(bin_dir)
    environment = recorder_environment(tmp_path, command_log)

    completed = subprocess.run(
        recorder_command("--duration", "1"),
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert completed.returncode == 0, completed.stderr
    experiment_name = "scene05_alignment_d100_center_yaw00_r01"
    experiment_dir = tmp_path / "bags" / experiment_name
    assert (experiment_dir / "rosbag" / "metadata.yaml").is_file()

    preflight = yaml.safe_load(
        (experiment_dir / "preflight.yaml").read_text(encoding="utf-8")
    )
    assert preflight["overall_result"] == "warn"
    assert preflight["evidence"]["registration_value"] == "True"
    assert preflight["evidence"]["wrapper_align_mode"] == "SW"
    assert preflight["evidence"]["color"]["width"] == 1280
    assert preflight["evidence"]["aligned_depth"]["height"] == 720
    assert preflight["evidence"]["recorded_topics"] == [
        "/camera/color/image_raw",
        "/camera/color/camera_info",
        "/camera/depth/image_raw",
        "/camera/depth/camera_info",
        "/tf_static",
        "/camera/depth/image_unaligned",
    ]

    experiment = yaml.safe_load(
        (experiment_dir / "experiment.yaml").read_text(encoding="utf-8")
    )
    assert experiment["registration"] == {
        "enabled": True,
        "aligned_depth_topic": "/camera/depth/image_raw",
        "mode": "sdk",
        "wrapper_align_mode": "SW",
        "aligned_to": "color",
        "coordinate_convention": "color_pixel_grid",
    }
    assert experiment["color"]["width"] == 1280
    assert experiment["depth"]["height"] == 720
    assert experiment["depth"]["precision"] == "1mm"

    post_recording = yaml.safe_load(
        (experiment_dir / "post_recording.yaml").read_text(encoding="utf-8")
    )
    assert post_recording["status"] == "success"
    assert "/diagnostics" not in post_recording["recorded_topics"]

    bag_invocation = next(
        line
        for line in command_log.read_text(encoding="utf-8").splitlines()
        if line.startswith("bag record ")
    )
    assert "/camera/depth/image_unaligned" in bag_invocation
    assert "/diagnostics" not in bag_invocation


def test_disabled_registration_fails_without_formal_directory(
    tmp_path: Path,
) -> None:
    bin_dir = tmp_path / "bin"
    command_log = install_fake_ros2(bin_dir)
    environment = recorder_environment(tmp_path, command_log)
    environment["FAKE_REGISTRATION"] = "False"

    completed = subprocess.run(
        recorder_command(),
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert completed.returncode == 1
    experiment_name = "scene05_alignment_d100_center_yaw00_r01"
    assert not (tmp_path / "bags" / experiment_name).exists()

    report_path = (
        tmp_path
        / "results"
        / "preflight"
        / f"{experiment_name}_preflight.yaml"
    )
    report = yaml.safe_load(report_path.read_text(encoding="utf-8"))
    assert report["overall_result"] == "fail"
    registration_check = next(
        check for check in report["checks"] if check["name"] == "registration_enabled"
    )
    assert registration_check["status"] == "fail"
    assert "depth_registration:=true" in registration_check["detail"]

    assert not any(
        line.startswith("bag record ")
        for line in command_log.read_text(encoding="utf-8").splitlines()
    )


def test_unresolved_depth_precision_fails_without_recording(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    command_log = install_fake_ros2(bin_dir)
    environment = recorder_environment(tmp_path, command_log)
    environment["FAKE_DEPTH_PRECISION"] = ""

    completed = subprocess.run(
        recorder_command(),
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert completed.returncode == 1
    experiment_name = "scene05_alignment_d100_center_yaw00_r01"
    assert not (tmp_path / "bags" / experiment_name).exists()

    report_path = (
        tmp_path
        / "results"
        / "preflight"
        / f"{experiment_name}_preflight.yaml"
    )
    report = yaml.safe_load(report_path.read_text(encoding="utf-8"))
    depth_check = next(
        check for check in report["checks"] if check["name"] == "depth_contract"
    )
    assert depth_check["status"] == "fail"
    assert "depth_precision:=1mm" in depth_check["detail"]
    assert not any(
        line.startswith("bag record ")
        for line in command_log.read_text(encoding="utf-8").splitlines()
    )
