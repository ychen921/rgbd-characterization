#!/usr/bin/env bash

set -euo pipefail

# Record a formal scene05 RGB-depth alignment experiment only after the
# running camera satisfies the documented data and device contract.

RGBD_CHARACTERIZATION_ROOT="${RGBD_CHARACTERIZATION_ROOT:-$(
    cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd
)}"
BAG_ROOT="${BAG_ROOT:-${RGBD_CHARACTERIZATION_ROOT}/bags}"
RESULTS_ROOT="${RESULTS_ROOT:-${RGBD_CHARACTERIZATION_ROOT}/results}"

DISTANCE_MM=""
POSITION=""
YAW_DEG=""
REPEAT_ID=""
DURATION_SEC=5
CAMERA_NODE="/camera/camera"
REGISTRATION_PARAM="depth_registration"
REGISTRATION_MODE="sdk"
EXPECTED_WIDTH=1280
EXPECTED_HEIGHT=720
EXPECTED_COLOR_ENCODING="rgb8"
EXPECTED_DEPTH_ENCODING="16UC1"
TOPIC_TIMEOUT_SEC=10

COLOR_TOPIC="/camera/color/image_raw"
COLOR_INFO_TOPIC="/camera/color/camera_info"
DEPTH_TOPIC="/camera/depth/image_raw"
DEPTH_INFO_TOPIC="/camera/depth/camera_info"
TF_STATIC_TOPIC="/tf_static"

DEPTH_UNIT="mm"
DEPTH_INVALID_VALUES="[0, 65535]"
TIMESTAMP_SOURCE="message_header"
TIMESTAMP_UNIT="ns"
COORDINATE_CONVENTION="color_pixel_grid"

usage() {
    cat <<'EOF'
Usage:
  scripts/record_alignment_experiment.sh [options]

Start the camera first with depth registration enabled:

  ros2 launch orbbec_camera gemini_330_series.launch.py \
    depth_registration:=true

Required:
  --distance-mm MM
  --position POSITION       center | top_left | top_right | bottom_left | bottom_right
  --yaw-deg DEG
  --repeat ID

Options:
  --duration SEC            Recording duration (default: 5)
  --camera-node NAME        Running camera node (default: /camera/camera)
  --registration-param NAME Registration parameter (default: depth_registration)
  --registration-mode MODE  device | sdk | ros_node (default: sdk)
  --expected-width PX       Expected color/aligned-depth width (default: 1280)
  --expected-height PX      Expected color/aligned-depth height (default: 720)
  --topic-timeout SEC       Per-field topic timeout (default: 10)
  -h, --help

Example:
  scripts/record_alignment_experiment.sh \
    --distance-mm 1000 \
    --position center \
    --yaw-deg 0 \
    --repeat 1 \
    --duration 5
EOF
}

error() {
    printf 'ERROR: %s\n' "$*" >&2
}

require_option_value() {
    local option="$1"
    local remaining="$2"
    if (( remaining < 2 )); then
        error "${option} requires a value"
        usage >&2
        exit 2
    fi
}

while (( $# > 0 )); do
    case "$1" in
        --distance-mm)
            require_option_value "$1" "$#"
            DISTANCE_MM="$2"
            shift 2
            ;;
        --position)
            require_option_value "$1" "$#"
            POSITION="$2"
            shift 2
            ;;
        --yaw-deg)
            require_option_value "$1" "$#"
            YAW_DEG="$2"
            shift 2
            ;;
        --repeat)
            require_option_value "$1" "$#"
            REPEAT_ID="$2"
            shift 2
            ;;
        --duration)
            require_option_value "$1" "$#"
            DURATION_SEC="$2"
            shift 2
            ;;
        --camera-node)
            require_option_value "$1" "$#"
            CAMERA_NODE="$2"
            shift 2
            ;;
        --registration-param)
            require_option_value "$1" "$#"
            REGISTRATION_PARAM="$2"
            shift 2
            ;;
        --registration-mode)
            require_option_value "$1" "$#"
            REGISTRATION_MODE="$2"
            shift 2
            ;;
        --expected-width)
            require_option_value "$1" "$#"
            EXPECTED_WIDTH="$2"
            shift 2
            ;;
        --expected-height)
            require_option_value "$1" "$#"
            EXPECTED_HEIGHT="$2"
            shift 2
            ;;
        --topic-timeout)
            require_option_value "$1" "$#"
            TOPIC_TIMEOUT_SEC="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            error "unknown argument: $1"
            usage >&2
            exit 2
            ;;
    esac
done

validate_positive_integer() {
    local value="$1"
    local label="$2"
    if ! [[ "${value}" =~ ^[0-9]+$ ]] || (( value <= 0 )); then
        error "${label} must be a positive integer"
        exit 2
    fi
}

for required_name in DISTANCE_MM POSITION YAW_DEG REPEAT_ID; do
    if [[ -z "${!required_name}" ]]; then
        error "--$(printf '%s' "${required_name}" | tr '[:upper:]_' '[:lower:]-') is required"
        exit 2
    fi
done

validate_positive_integer "${DISTANCE_MM}" "--distance-mm"
validate_positive_integer "${REPEAT_ID}" "--repeat"
validate_positive_integer "${DURATION_SEC}" "--duration"
validate_positive_integer "${EXPECTED_WIDTH}" "--expected-width"
validate_positive_integer "${EXPECTED_HEIGHT}" "--expected-height"
validate_positive_integer "${TOPIC_TIMEOUT_SEC}" "--topic-timeout"

if (( DISTANCE_MM % 10 != 0 )); then
    error "--distance-mm must be a multiple of 10 for the dCCC naming convention"
    exit 2
fi

if ! [[ "${YAW_DEG}" =~ ^-?[0-9]+$ ]]; then
    error "--yaw-deg must be an integer"
    exit 2
fi

case "${POSITION}" in
    center|top_left|top_right|bottom_left|bottom_right) ;;
    *)
        error "--position must be center, top_left, top_right, bottom_left, or bottom_right"
        exit 2
        ;;
esac

case "${REGISTRATION_MODE}" in
    device|sdk|ros_node) ;;
    *)
        error "--registration-mode must be device, sdk, or ros_node"
        exit 2
        ;;
esac

if [[ "${CAMERA_NODE}" != /* ]]; then
    error "--camera-node must be an absolute ROS node name"
    exit 2
fi

if [[ -z "${REGISTRATION_PARAM}" ]]; then
    error "--registration-param must not be empty"
    exit 2
fi

distance_cm=$((DISTANCE_MM / 10))
printf -v distance_tag 'd%03d' "${distance_cm}"
printf -v repeat_tag 'r%02d' "${REPEAT_ID}"

if (( YAW_DEG == 0 )); then
    yaw_tag="yaw00"
elif (( YAW_DEG > 0 )); then
    printf -v yaw_tag 'yawp%02d' "${YAW_DEG}"
else
    yaw_abs=$((-YAW_DEG))
    printf -v yaw_tag 'yawm%02d' "${yaw_abs}"
fi

EXPERIMENT_ID="scene05_alignment_${distance_tag}_${POSITION}_${yaw_tag}_${repeat_tag}"
EXPERIMENT_DIR="${BAG_ROOT}/${EXPERIMENT_ID}"
ROSBAG_DIR="${EXPERIMENT_DIR}/rosbag"
FAILURE_REPORT_DIR="${RESULTS_ROOT}/preflight"
FAILURE_REPORT="${FAILURE_REPORT_DIR}/${EXPERIMENT_ID}_preflight.yaml"

REQUIRED_TOPICS=(
    "${COLOR_TOPIC}"
    "${COLOR_INFO_TOPIC}"
    "${DEPTH_TOPIC}"
    "${DEPTH_INFO_TOPIC}"
    "${TF_STATIC_TOPIC}"
)

OPTIONAL_TOPICS=(
    "/camera/depth/image_unaligned"
    "/camera/depth_to_color"
    "/diagnostics"
    "/camera/device_status"
)

CHECK_NAMES=()
CHECK_STATUSES=()
CHECK_DETAILS=()
WARNINGS=()
ERRORS=()
RECORDED_TOPICS=()
FAILURE_COUNT=0
WARNING_COUNT=0

add_check() {
    local name="$1"
    local status="$2"
    local detail="$3"
    CHECK_NAMES+=("${name}")
    CHECK_STATUSES+=("${status}")
    CHECK_DETAILS+=("${detail}")

    case "${status}" in
        fail)
            ERRORS+=("${name}: ${detail}")
            FAILURE_COUNT=$((FAILURE_COUNT + 1))
            ;;
        warn)
            WARNINGS+=("${name}: ${detail}")
            WARNING_COUNT=$((WARNING_COUNT + 1))
            ;;
    esac
}

yaml_quote() {
    local value="$1"
    value="${value//$'\n'/ }"
    value="${value//\'/\'\'}"
    printf "'%s'" "${value}"
}

write_yaml_list() {
    local indent="$1"
    shift
    local item
    if (( $# == 0 )); then
        printf '%s[]\n' "${indent}"
        return
    fi
    for item in "$@"; do
        printf '%s- ' "${indent}"
        yaml_quote "${item}"
        printf '\n'
    done
}

overall_result() {
    if (( FAILURE_COUNT > 0 )); then
        printf 'fail'
    elif (( WARNING_COUNT > 0 )); then
        printf 'warn'
    else
        printf 'pass'
    fi
}

write_preflight() {
    local output_path="$1"
    local result
    result="$(overall_result)"

    {
        printf 'schema_version: 1\n'
        printf 'experiment: '
        yaml_quote "${EXPERIMENT_ID}"
        printf '\n'
        printf 'overall_result: %s\n' "${result}"
        printf 'checks:\n'
        local index
        for ((index = 0; index < ${#CHECK_NAMES[@]}; index++)); do
            printf '  - name: '
            yaml_quote "${CHECK_NAMES[index]}"
            printf '\n    status: %s\n    detail: ' "${CHECK_STATUSES[index]}"
            yaml_quote "${CHECK_DETAILS[index]}"
            printf '\n'
        done
        printf 'warnings:\n'
        write_yaml_list '  ' "${WARNINGS[@]}"
        printf 'errors:\n'
        write_yaml_list '  ' "${ERRORS[@]}"
        printf 'evidence:\n'
        printf '  camera_node: '
        yaml_quote "${CAMERA_NODE}"
        printf '\n  registration_parameter: '
        yaml_quote "${REGISTRATION_PARAM}"
        printf '\n  registration_value: '
        yaml_quote "${REGISTRATION_VALUE:-unavailable}"
        printf '\n  registration_mode: '
        yaml_quote "${REGISTRATION_MODE}"
        printf '\n  wrapper_align_mode: '
        yaml_quote "${ALIGN_MODE_VALUE:-unavailable}"
        printf '\n  align_target_stream: '
        yaml_quote "${ALIGN_TARGET_VALUE:-unavailable}"
        printf '\n  coordinate_convention: '
        yaml_quote "${COORDINATE_CONVENTION}"
        printf '\n  color:\n'
        printf '    topic: '
        yaml_quote "${COLOR_TOPIC}"
        printf '\n    width: %s\n    height: %s\n    encoding: ' "${COLOR_WIDTH:-null}" "${COLOR_HEIGHT:-null}"
        yaml_quote "${COLOR_ENCODING:-unavailable}"
        printf '\n    frame_id: '
        yaml_quote "${COLOR_FRAME_ID:-unavailable}"
        printf '\n  aligned_depth:\n'
        printf '    topic: '
        yaml_quote "${DEPTH_TOPIC}"
        printf '\n    width: %s\n    height: %s\n    encoding: ' "${DEPTH_WIDTH:-null}" "${DEPTH_HEIGHT:-null}"
        yaml_quote "${DEPTH_ENCODING:-unavailable}"
        printf '\n    frame_id: '
        yaml_quote "${DEPTH_FRAME_ID:-unavailable}"
        printf '\n    unit: %s\n    invalid_values: %s\n' "${DEPTH_UNIT}" "${DEPTH_INVALID_VALUES}"
        printf '  timestamps:\n'
        printf '    source: %s\n    unit: %s\n    clock_domain: ' "${TIMESTAMP_SOURCE}" "${TIMESTAMP_UNIT}"
        yaml_quote "${TIME_DOMAIN_VALUE:-unavailable}"
        printf '\n  expected_image_size:\n'
        printf '    width: %s\n    height: %s\n' "${EXPECTED_WIDTH}" "${EXPECTED_HEIGHT}"
        printf '  recorded_topics:\n'
        write_yaml_list '    ' "${RECORDED_TOPICS[@]}"
    } > "${output_path}"
}

trim_scalar() {
    local value="$1"
    value="${value#data:}"
    value="${value%%$'\n'*}"
    value="${value#${value%%[![:space:]]*}}"
    value="${value%${value##*[![:space:]]}}"
    if [[ "${value}" == \"*\" ]] || [[ "${value}" == \'*\' ]]; then
        value="${value:1:${#value}-2}"
    fi
    printf '%s' "${value}"
}

PARAM_OUTPUT=""
PARAM_VALUE=""
read_parameter() {
    local parameter="$1"
    if ! PARAM_OUTPUT="$(ros2 param get "${CAMERA_NODE}" "${parameter}" 2>&1)"; then
        PARAM_VALUE=""
        return 1
    fi
    PARAM_VALUE="${PARAM_OUTPUT##*: }"
    PARAM_VALUE="$(trim_scalar "${PARAM_VALUE}")"
    [[ -n "${PARAM_VALUE}" ]]
}

FIELD_OUTPUT=""
FIELD_VALUE=""
read_topic_field() {
    local topic="$1"
    local field="$2"
    if ! FIELD_OUTPUT="$(
        timeout "${TOPIC_TIMEOUT_SEC}" \
            ros2 topic echo "${topic}" --once --field "${field}" 2>&1
    )"; then
        FIELD_VALUE=""
        return 1
    fi
    FIELD_VALUE="$(trim_scalar "${FIELD_OUTPUT}")"
    [[ -n "${FIELD_VALUE}" ]]
}

topic_available() {
    local wanted="$1"
    local topic
    while IFS= read -r topic; do
        if [[ "${topic}" == "${wanted}" ]]; then
            return 0
        fi
    done <<< "${TOPIC_LIST:-}"
    return 1
}

STAGING_DIR="$(mktemp -d "${TMPDIR:-/tmp}/rgbd-alignment-preflight.XXXXXX")"
cleanup_staging() {
    rm -rf -- "${STAGING_DIR}"
}
trap cleanup_staging EXIT

for command_name in ros2 timeout; do
    if command -v "${command_name}" >/dev/null 2>&1; then
        add_check "command_${command_name}" pass "${command_name} is available"
    else
        add_check "command_${command_name}" fail "${command_name} command was not found"
    fi
done

if [[ -e "${EXPERIMENT_DIR}" ]]; then
    add_check experiment_directory fail "experiment already exists: ${EXPERIMENT_DIR}"
else
    add_check experiment_directory pass "formal experiment path is available"
fi

TOPIC_LIST=""
if command -v ros2 >/dev/null 2>&1; then
    if TOPIC_LIST="$(ros2 topic list 2>&1)"; then
        add_check topic_discovery pass "ROS topic discovery succeeded"
    else
        add_check topic_discovery fail "ROS topic discovery failed: ${TOPIC_LIST}"
        TOPIC_LIST=""
    fi
else
    add_check topic_discovery fail "ROS topic discovery was skipped because ros2 is unavailable"
fi

for topic in "${REQUIRED_TOPICS[@]}"; do
    if topic_available "${topic}"; then
        add_check "required_topic_${topic}" pass "required topic is available"
        RECORDED_TOPICS+=("${topic}")
    else
        add_check "required_topic_${topic}" fail "required topic is missing"
    fi
done

for topic in "${OPTIONAL_TOPICS[@]}"; do
    if topic_available "${topic}"; then
        add_check "optional_topic_${topic}" pass "optional topic is available"
        RECORDED_TOPICS+=("${topic}")
    else
        add_check "optional_topic_${topic}" warn "optional topic is unavailable and will not be recorded"
    fi
done

CAMERA_PARAMS_PATH="${STAGING_DIR}/camera_params.yaml"
: > "${STAGING_DIR}/param_dump_error.txt"
if command -v ros2 >/dev/null 2>&1 && \
   ros2 param dump "${CAMERA_NODE}" > "${CAMERA_PARAMS_PATH}" 2> "${STAGING_DIR}/param_dump_error.txt"; then
    add_check camera_parameter_dump pass "camera parameters were captured"
else
    param_error="$(<"${STAGING_DIR}/param_dump_error.txt")"
    add_check camera_parameter_dump fail "camera parameter dump failed: ${param_error:-no diagnostic output}"
fi

REGISTRATION_VALUE=""
if command -v ros2 >/dev/null 2>&1 && read_parameter "${REGISTRATION_PARAM}"; then
    REGISTRATION_VALUE="${PARAM_VALUE}"
    if [[ "${REGISTRATION_VALUE}" =~ ^([Tt]rue|1)$ ]]; then
        add_check registration_enabled pass "runtime ${REGISTRATION_PARAM}=${REGISTRATION_VALUE}"
    else
        add_check registration_enabled fail \
            "runtime ${REGISTRATION_PARAM}=${REGISTRATION_VALUE}; start the camera with depth_registration:=true"
    fi
else
    add_check registration_enabled fail \
        "could not read ${REGISTRATION_PARAM}: ${PARAM_OUTPUT:-no diagnostic output}"
fi

ALIGN_MODE_VALUE=""
if command -v ros2 >/dev/null 2>&1 && read_parameter align_mode; then
    ALIGN_MODE_VALUE="${PARAM_VALUE}"
    case "${ALIGN_MODE_VALUE^^}" in
        SW|HW)
            add_check registration_mode pass \
                "registration.mode=${REGISTRATION_MODE}, runtime align_mode=${ALIGN_MODE_VALUE}"
            ;;
        *)
            add_check registration_mode fail "runtime align_mode is unresolved: ${ALIGN_MODE_VALUE}"
            ;;
    esac
else
    add_check registration_mode fail "could not read align_mode: ${PARAM_OUTPUT:-no diagnostic output}"
fi

ALIGN_TARGET_VALUE=""
if command -v ros2 >/dev/null 2>&1 && read_parameter align_target_stream; then
    ALIGN_TARGET_VALUE="${PARAM_VALUE}"
    if [[ "${ALIGN_TARGET_VALUE^^}" == "COLOR" ]]; then
        add_check align_target_stream pass "runtime align_target_stream=${ALIGN_TARGET_VALUE}"
    else
        add_check align_target_stream fail \
            "runtime align_target_stream=${ALIGN_TARGET_VALUE}; expected COLOR"
    fi
else
    add_check align_target_stream fail \
        "could not read align_target_stream: ${PARAM_OUTPUT:-no diagnostic output}"
fi

TIME_DOMAIN_VALUE=""
if command -v ros2 >/dev/null 2>&1 && read_parameter time_domain; then
    TIME_DOMAIN_VALUE="${PARAM_VALUE}"
    if [[ -n "${TIME_DOMAIN_VALUE}" ]]; then
        add_check timestamp_contract pass \
            "source=${TIMESTAMP_SOURCE}, unit=${TIMESTAMP_UNIT}, clock_domain=${TIME_DOMAIN_VALUE}"
    else
        add_check timestamp_contract fail "runtime time_domain is empty"
    fi
else
    add_check timestamp_contract fail "could not read time_domain: ${PARAM_OUTPUT:-no diagnostic output}"
fi

DEPTH_PRECISION_VALUE="unavailable"
if command -v ros2 >/dev/null 2>&1 && read_parameter depth_precision; then
    DEPTH_PRECISION_VALUE="${PARAM_VALUE}"
    if [[ "${DEPTH_PRECISION_VALUE,,}" == "1mm" ]]; then
        add_check depth_contract pass \
            "unit=${DEPTH_UNIT}, invalid_values=${DEPTH_INVALID_VALUES}, runtime depth_precision=${DEPTH_PRECISION_VALUE}"
    elif [[ -z "${DEPTH_PRECISION_VALUE}" ]]; then
        DEPTH_PRECISION_VALUE="wrapper_default"
        add_check depth_contract warn \
            "unit=${DEPTH_UNIT} and invalid_values=${DEPTH_INVALID_VALUES} are documented; runtime depth_precision uses the wrapper default"
    else
        add_check depth_contract fail \
            "runtime depth_precision=${DEPTH_PRECISION_VALUE}; expected 1mm"
    fi
else
    add_check depth_contract warn \
        "unit=${DEPTH_UNIT} and invalid_values=${DEPTH_INVALID_VALUES} are documented, but depth_precision could not be read"
fi

read_image_contract() {
    local prefix="$1"
    local topic="$2"
    local field variable_name
    if ! topic_available "${topic}"; then
        for field in height width encoding; do
            variable_name="${prefix}_${field^^}"
            printf -v "${variable_name}" '%s' ""
            add_check "${prefix,,}_${field}" fail \
                "field check was skipped because ${topic} is unavailable"
        done
        return
    fi
    for field in height width encoding; do
        variable_name="${prefix}_${field^^}"
        if command -v ros2 >/dev/null 2>&1 && command -v timeout >/dev/null 2>&1 && \
           read_topic_field "${topic}" "${field}"; then
            printf -v "${variable_name}" '%s' "${FIELD_VALUE}"
            add_check "${prefix,,}_${field}" pass "${topic} ${field}=${FIELD_VALUE}"
        else
            printf -v "${variable_name}" '%s' ""
            add_check "${prefix,,}_${field}" fail \
                "could not read ${field} from ${topic}: ${FIELD_OUTPUT:-no diagnostic output}"
        fi
    done
}

read_camera_info_dimensions() {
    local prefix="$1"
    local topic="$2"
    local field variable_name
    if ! topic_available "${topic}"; then
        for field in height width; do
            variable_name="${prefix}_${field^^}"
            printf -v "${variable_name}" '%s' ""
            add_check "${prefix,,}_${field}" fail \
                "field check was skipped because ${topic} is unavailable"
        done
        return
    fi
    for field in height width; do
        variable_name="${prefix}_${field^^}"
        if command -v ros2 >/dev/null 2>&1 && command -v timeout >/dev/null 2>&1 && \
           read_topic_field "${topic}" "${field}"; then
            printf -v "${variable_name}" '%s' "${FIELD_VALUE}"
            add_check "${prefix,,}_${field}" pass "${topic} ${field}=${FIELD_VALUE}"
        else
            printf -v "${variable_name}" '%s' ""
            add_check "${prefix,,}_${field}" fail \
                "could not read ${field} from ${topic}: ${FIELD_OUTPUT:-no diagnostic output}"
        fi
    done
}

COLOR_HEIGHT=""
COLOR_WIDTH=""
COLOR_ENCODING=""
DEPTH_HEIGHT=""
DEPTH_WIDTH=""
DEPTH_ENCODING=""
COLOR_INFO_HEIGHT=""
COLOR_INFO_WIDTH=""
DEPTH_INFO_HEIGHT=""
DEPTH_INFO_WIDTH=""
COLOR_FRAME_ID=""
DEPTH_FRAME_ID=""

read_image_contract COLOR "${COLOR_TOPIC}"
read_image_contract DEPTH "${DEPTH_TOPIC}"
read_camera_info_dimensions COLOR_INFO "${COLOR_INFO_TOPIC}"
read_camera_info_dimensions DEPTH_INFO "${DEPTH_INFO_TOPIC}"

if topic_available "${COLOR_TOPIC}" && \
   command -v ros2 >/dev/null 2>&1 && command -v timeout >/dev/null 2>&1 && \
   read_topic_field "${COLOR_TOPIC}" "header.frame_id"; then
    COLOR_FRAME_ID="${FIELD_VALUE}"
    add_check color_frame_id pass "${COLOR_TOPIC} frame_id=${COLOR_FRAME_ID}"
else
    add_check color_frame_id fail \
        "could not read frame_id from ${COLOR_TOPIC}: ${FIELD_OUTPUT:-no diagnostic output}"
fi

if topic_available "${DEPTH_TOPIC}" && \
   command -v ros2 >/dev/null 2>&1 && command -v timeout >/dev/null 2>&1 && \
   read_topic_field "${DEPTH_TOPIC}" "header.frame_id"; then
    DEPTH_FRAME_ID="${FIELD_VALUE}"
    add_check depth_frame_id pass "${DEPTH_TOPIC} frame_id=${DEPTH_FRAME_ID}"
else
    add_check depth_frame_id fail \
        "could not read frame_id from ${DEPTH_TOPIC}: ${FIELD_OUTPUT:-no diagnostic output}"
fi

if [[ "${COLOR_WIDTH}" == "${EXPECTED_WIDTH}" && \
      "${COLOR_HEIGHT}" == "${EXPECTED_HEIGHT}" ]]; then
    add_check color_expected_dimensions pass \
        "color dimensions are ${EXPECTED_WIDTH}x${EXPECTED_HEIGHT}"
else
    add_check color_expected_dimensions fail \
        "color dimensions are ${COLOR_WIDTH:-unknown}x${COLOR_HEIGHT:-unknown}; expected ${EXPECTED_WIDTH}x${EXPECTED_HEIGHT}"
fi

if [[ "${DEPTH_WIDTH}" == "${EXPECTED_WIDTH}" && \
      "${DEPTH_HEIGHT}" == "${EXPECTED_HEIGHT}" ]]; then
    add_check aligned_depth_expected_dimensions pass \
        "aligned depth dimensions are ${EXPECTED_WIDTH}x${EXPECTED_HEIGHT}"
else
    add_check aligned_depth_expected_dimensions fail \
        "aligned depth dimensions are ${DEPTH_WIDTH:-unknown}x${DEPTH_HEIGHT:-unknown}; expected ${EXPECTED_WIDTH}x${EXPECTED_HEIGHT}"
fi

if [[ -n "${COLOR_WIDTH}" && "${COLOR_WIDTH}" == "${DEPTH_WIDTH}" && \
      -n "${COLOR_HEIGHT}" && "${COLOR_HEIGHT}" == "${DEPTH_HEIGHT}" ]]; then
    add_check aligned_depth_color_pixel_grid pass \
        "RGB and aligned depth share a ${COLOR_WIDTH}x${COLOR_HEIGHT} pixel grid"
else
    add_check aligned_depth_color_pixel_grid fail \
        "RGB and aligned depth dimensions are incompatible"
fi

if [[ -n "${COLOR_FRAME_ID}" && "${COLOR_FRAME_ID}" == "${DEPTH_FRAME_ID}" ]]; then
    add_check aligned_depth_frame_id pass \
        "RGB and aligned depth use frame_id=${COLOR_FRAME_ID}"
else
    add_check aligned_depth_frame_id fail \
        "RGB frame_id=${COLOR_FRAME_ID:-unknown}, aligned depth frame_id=${DEPTH_FRAME_ID:-unknown}"
fi

if [[ "${COLOR_ENCODING}" == "${EXPECTED_COLOR_ENCODING}" ]]; then
    add_check color_encoding_contract pass "color encoding=${COLOR_ENCODING}"
else
    add_check color_encoding_contract fail \
        "color encoding=${COLOR_ENCODING:-unknown}; expected ${EXPECTED_COLOR_ENCODING}"
fi

if [[ "${DEPTH_ENCODING}" == "${EXPECTED_DEPTH_ENCODING}" ]]; then
    add_check depth_encoding_contract pass "aligned depth encoding=${DEPTH_ENCODING}"
else
    add_check depth_encoding_contract fail \
        "aligned depth encoding=${DEPTH_ENCODING:-unknown}; expected ${EXPECTED_DEPTH_ENCODING}"
fi

if [[ "${COLOR_INFO_WIDTH}" == "${COLOR_WIDTH}" && \
      "${COLOR_INFO_HEIGHT}" == "${COLOR_HEIGHT}" ]]; then
    add_check color_camera_info_dimensions pass "color CameraInfo matches the color image"
else
    add_check color_camera_info_dimensions fail "color CameraInfo dimensions do not match the color image"
fi

if [[ "${DEPTH_INFO_WIDTH}" == "${DEPTH_WIDTH}" && \
      "${DEPTH_INFO_HEIGHT}" == "${DEPTH_HEIGHT}" ]]; then
    add_check depth_camera_info_dimensions pass "depth CameraInfo matches the aligned depth image"
else
    add_check depth_camera_info_dimensions fail \
        "depth CameraInfo dimensions do not match the aligned depth image"
fi

PREFLIGHT_PATH="${STAGING_DIR}/preflight.yaml"
write_preflight "${PREFLIGHT_PATH}"

if (( FAILURE_COUNT > 0 )); then
    mkdir -p "${FAILURE_REPORT_DIR}"
    if [[ -e "${FAILURE_REPORT}" ]]; then
        failure_timestamp="$(date +%Y%m%dT%H%M%S)"
        FAILURE_REPORT="${FAILURE_REPORT_DIR}/${EXPERIMENT_ID}_preflight_${failure_timestamp}.yaml"
    fi
    mv "${PREFLIGHT_PATH}" "${FAILURE_REPORT}"
    printf 'Preflight: FAIL\n'
    printf 'Report: %s\n' "${FAILURE_REPORT}"
    printf 'Formal experiment directory was not created.\n'
    exit 1
fi

mkdir -p "${EXPERIMENT_DIR}"
mv "${PREFLIGHT_PATH}" "${EXPERIMENT_DIR}/preflight.yaml"
mv "${CAMERA_PARAMS_PATH}" "${EXPERIMENT_DIR}/camera_params.yaml"

recording_timestamp="$(date --iso-8601=seconds)"
cat > "${EXPERIMENT_DIR}/experiment.yaml" <<EOF
schema_version: 1
experiment:
  name: ${EXPERIMENT_ID}
  type: rgb_depth_alignment
  scene: 5
  repeat: ${REPEAT_ID}
geometry:
  camera_to_foreground_mm: ${DISTANCE_MM}
  image_position: ${POSITION}
  yaw_deg: ${YAW_DEG}
registration:
  enabled: true
  aligned_depth_topic: ${DEPTH_TOPIC}
  mode: ${REGISTRATION_MODE}
  wrapper_align_mode: ${ALIGN_MODE_VALUE}
  aligned_to: color
  coordinate_convention: ${COORDINATE_CONVENTION}
color:
  topic: ${COLOR_TOPIC}
  width: ${COLOR_WIDTH}
  height: ${COLOR_HEIGHT}
  encoding: ${COLOR_ENCODING}
depth:
  topic: ${DEPTH_TOPIC}
  width: ${DEPTH_WIDTH}
  height: ${DEPTH_HEIGHT}
  encoding: ${DEPTH_ENCODING}
  unit: ${DEPTH_UNIT}
  invalid_values: ${DEPTH_INVALID_VALUES}
timestamps:
  source: ${TIMESTAMP_SOURCE}
  unit: ${TIMESTAMP_UNIT}
  clock_domain: ${TIME_DOMAIN_VALUE}
recording:
  requested_at: ${recording_timestamp}
  duration_sec: ${DURATION_SEC}
EOF

printf 'Preflight: %s\n' "$(overall_result | tr '[:lower:]' '[:upper:]')"
printf 'Experiment: %s\n' "${EXPERIMENT_ID}"
printf 'Recording %s verified topics for %s seconds...\n' "${#RECORDED_TOPICS[@]}" "${DURATION_SEC}"

ROSBAG_PID=""
stop_rosbag() {
    if [[ -n "${ROSBAG_PID}" ]] && kill -0 "${ROSBAG_PID}" 2>/dev/null; then
        kill -SIGINT "${ROSBAG_PID}" 2>/dev/null || true
        wait "${ROSBAG_PID}" 2>/dev/null || true
    fi
}
handle_recording_signal() {
    stop_rosbag
    exit 130
}
trap cleanup_staging EXIT
trap handle_recording_signal INT TERM

set +e
ros2 bag record -o "${ROSBAG_DIR}" "${RECORDED_TOPICS[@]}" &
ROSBAG_PID=$!
set -e

sleep "${DURATION_SEC}"
if kill -0 "${ROSBAG_PID}" 2>/dev/null; then
    kill -SIGINT "${ROSBAG_PID}" 2>/dev/null || true
fi

set +e
wait "${ROSBAG_PID}"
RECORD_EXIT_CODE=$?
set -e
ROSBAG_PID=""

recording_status="success"
if [[ "${RECORD_EXIT_CODE}" -ne 0 && "${RECORD_EXIT_CODE}" -ne 130 ]]; then
    recording_status="failed"
fi
if [[ ! -f "${ROSBAG_DIR}/metadata.yaml" ]]; then
    recording_status="failed"
fi

cat > "${EXPERIMENT_DIR}/post_recording.yaml" <<EOF
schema_version: 1
experiment: ${EXPERIMENT_ID}
status: ${recording_status}
requested_duration_sec: ${DURATION_SEC}
recorder_exit_status: ${RECORD_EXIT_CODE}
bag_path: ${ROSBAG_DIR}
recorded_topics:
EOF
for topic in "${RECORDED_TOPICS[@]}"; do
    printf '  - %s\n' "${topic}" >> "${EXPERIMENT_DIR}/post_recording.yaml"
done

if [[ "${recording_status}" != "success" ]]; then
    error "rosbag recording failed; see ${EXPERIMENT_DIR}/post_recording.yaml"
    exit 1
fi

printf 'Recording completed.\n'
printf 'Experiment directory: %s\n' "${EXPERIMENT_DIR}"
