#!/usr/bin/env bash

set -euo pipefail

# ============================================================
# Orbbec sensor characterization experiment recorder
# ============================================================

WORKSPACE_DIR="${WORKSPACE_DIR:-$HOME/dev/orbbec_ws}"
BAG_ROOT="${BAG_ROOT:-${WORKSPACE_DIR}/bags}"

SENSOR_MODEL="Orbbec Gemini 335L"
SENSOR_SERIAL="unknown"

SCENE=""
MATERIAL=""
DISTANCE_M=""
ANGLE_DEG=""
CAMERA_HEIGHT_M=""
TARGET_WIDTH_M=""
TARGET_HEIGHT_M=""
DURATION_SEC=30
REPEAT_ID=1
NOTES=""

usage() {
    cat <<EOF
Usage:
  $0 [options]

Required:
  --scene NAME
  --material NAME

Options:
  --distance M
  --angle DEG
  --camera-height M
  --target-width M
  --target-height M
  --duration SEC
  --repeat ID
  --notes TEXT

Example:
  $0 \\
    --scene scene01_distance \\
    --material white \\
    --distance 1.0 \\
    --angle 0 \\
    --camera-height 0.85 \\
    --target-width 0.30 \\
    --target-height 0.50 \\
    --duration 30 \\
    --repeat 1
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --scene)
            SCENE="$2"
            shift 2
            ;;
        --material)
            MATERIAL="$2"
            shift 2
            ;;
        --distance)
            DISTANCE_M="$2"
            shift 2
            ;;
        --angle)
            ANGLE_DEG="$2"
            shift 2
            ;;
        --camera-height)
            CAMERA_HEIGHT_M="$2"
            shift 2
            ;;
        --target-width)
            TARGET_WIDTH_M="$2"
            shift 2
            ;;
        --target-height)
            TARGET_HEIGHT_M="$2"
            shift 2
            ;;
        --duration)
            DURATION_SEC="$2"
            shift 2
            ;;
        --repeat)
            REPEAT_ID="$2"
            shift 2
            ;;
        --notes)
            NOTES="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            usage
            exit 1
            ;;
    esac
done

# ============================================================
# Validation
# ============================================================

if [[ -z "${SCENE}" ]]; then
    echo "ERROR: --scene is required"
    exit 1
fi

if [[ -z "${MATERIAL}" ]]; then
    echo "ERROR: --material is required"
    exit 1
fi

if ! [[ "${DURATION_SEC}" =~ ^[0-9]+$ ]] || [[ "${DURATION_SEC}" -le 0 ]]; then
    echo "ERROR: --duration must be a positive integer"
    exit 1
fi

if ! [[ "${REPEAT_ID}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: --repeat must be an integer"
    exit 1
fi

command -v ros2 >/dev/null 2>&1 || {
    echo "ERROR: ros2 command not found"
    exit 1
}

# ============================================================
# Experiment ID
# ============================================================

distance_tag="dna"

if [[ -n "${DISTANCE_M}" ]]; then
    distance_cm="$(
        python3 -c \
        "print(round(float('${DISTANCE_M}') * 100))"
    )"

    printf -v distance_tag "d%03d" "${distance_cm}"
fi

printf -v repeat_tag "r%02d" "${REPEAT_ID}"

EXPERIMENT_ID="${SCENE}_${MATERIAL}_${distance_tag}_${repeat_tag}"

EXPERIMENT_DIR="${BAG_ROOT}/${EXPERIMENT_ID}"
ROSBAG_DIR="${EXPERIMENT_DIR}/rosbag"
EXPERIMENT_YAML="${EXPERIMENT_DIR}/experiment.yaml"

if [[ -e "${EXPERIMENT_DIR}" ]]; then
    echo "ERROR: experiment already exists:"
    echo "  ${EXPERIMENT_DIR}"
    exit 1
fi

mkdir -p "${EXPERIMENT_DIR}"

# ============================================================
# Experiment metadata
# ============================================================

TIMESTAMP="$(date --iso-8601=seconds)"

cat > "${EXPERIMENT_YAML}" <<EOF
scene_id: "${EXPERIMENT_ID}"

sensor:
  model: "${SENSOR_MODEL}"
  serial_number: "${SENSOR_SERIAL}"

target:
  material: "${MATERIAL}"
  width_m: ${TARGET_WIDTH_M:-null}
  height_m: ${TARGET_HEIGHT_M:-null}

setup:
  distance_m: ${DISTANCE_M:-null}
  incidence_angle_deg: ${ANGLE_DEG:-null}
  camera_height_m: ${CAMERA_HEIGHT_M:-null}

environment:
  indoor: true
  lighting: office
  sunlight: false

recording:
  timestamp: "${TIMESTAMP}"
  duration_sec: ${DURATION_SEC}
  repeat_id: ${REPEAT_ID}

notes: "${NOTES}"
EOF

echo
echo "Experiment"
echo "--------------------------------------------------"
echo "ID       : ${EXPERIMENT_ID}"
echo "Directory: ${EXPERIMENT_DIR}"
echo "Duration : ${DURATION_SEC} sec"
echo
echo "Metadata:"
cat "${EXPERIMENT_YAML}"
echo

# ============================================================
# ROS topics
# ============================================================

TOPICS=(
    /camera/depth/image_raw
    /camera/depth/camera_info
    /camera/depth/metadata
    /camera/depth/points
    /camera/color/image_raw
    /camera/color/camera_info
    /camera/device_status
    /camera/depth_filter_status
    /camera/depth_filters/status
    /diagnostics
)

# ============================================================
# rosbag recording
# ============================================================

echo
echo "Starting rosbag recording..."
echo

set +e

timeout \
    --signal=SIGINT \
    --kill-after=10s \
    "${DURATION_SEC}s" \
    ros2 bag record \
        -o "${ROSBAG_DIR}" \
        "${TOPICS[@]}"

RECORD_EXIT_CODE=$?

set -e

# timeout returns 124 when duration expires normally.
if [[ "${RECORD_EXIT_CODE}" -ne 0 ]] &&
   [[ "${RECORD_EXIT_CODE}" -ne 124 ]]; then

    echo
    echo "ERROR: rosbag recording failed"
    echo "Exit code: ${RECORD_EXIT_CODE}"
    exit "${RECORD_EXIT_CODE}"
fi

# ============================================================
# Result validation
# ============================================================

if [[ ! -f "${ROSBAG_DIR}/metadata.yaml" ]]; then
    echo
    echo "ERROR: rosbag metadata.yaml was not generated"
    exit 1
fi

echo
echo "Recording completed"
echo "--------------------------------------------------"
echo "Experiment : ${EXPERIMENT_ID}"
echo "Metadata   : ${EXPERIMENT_YAML}"
echo "Rosbag     : ${ROSBAG_DIR}"