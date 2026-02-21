#!/usr/bin/env bash
set -euo pipefail

criterion_dir="${1:-target/criterion}"
output_file="${2:-${criterion_dir}/multistream-summary.md}"

mkdir -p "$(dirname "${output_file}")"

declare -a CASES=(
  "1ch_10ms_32000"
  "1ch_10ms_96000"
  "1ch_20ms_96000"
  "2ch_10ms_32000"
  "2ch_10ms_96000"
  "2ch_20ms_96000"
  "6ch_10ms_32000"
  "6ch_10ms_96000"
  "6ch_20ms_96000"
)

median_point_estimate() {
  local file="$1"
  jq -r '.median.point_estimate' "${file}"
}

ns_to_ms() {
  local ns="$1"
  awk -v x="${ns}" 'BEGIN { printf "%.3f", x/1000000.0 }'
}

ratio_r_over_c() {
  local rust_ns="$1"
  local c_ns="$2"
  awk -v r="${rust_ns}" -v c="${c_ns}" 'BEGIN { if (c == 0) { print "n/a" } else { printf "%.2f", r/c } }'
}

{
  echo "# Multistream Benchmark Summary"
  echo
  echo "Generated from Criterion outputs in \`${criterion_dir}\`."
  echo

  for op in encode decode; do
    if [[ "${op}" == "encode" ]]; then
      op_title="Encode"
    else
      op_title="Decode"
    fi
    echo "## ${op_title}"
    echo
    echo "| Scenario | Rust | C | Ratio (Rust/C) |"
    echo "|---|---:|---:|---:|"

    rows=0
    for case in "${CASES[@]}"; do
      rust_file="${criterion_dir}/multistream_${op}_cmp/rust/${case}/new/estimates.json"
      c_file="${criterion_dir}/multistream_${op}_cmp/c/${case}/new/estimates.json"

      if [[ ! -f "${rust_file}" || ! -f "${c_file}" ]]; then
        continue
      fi

      rust_ns="$(median_point_estimate "${rust_file}")"
      c_ns="$(median_point_estimate "${c_file}")"
      rust_ms="$(ns_to_ms "${rust_ns}")"
      c_ms="$(ns_to_ms "${c_ns}")"
      ratio="$(ratio_r_over_c "${rust_ns}" "${c_ns}")"

      echo "| ${case} | ${rust_ms} ms | ${c_ms} ms | ${ratio}x |"
      rows=$((rows + 1))
    done

    if [[ ${rows} -eq 0 ]]; then
      echo "| (no comparable rust/c rows found) | - | - | - |"
    fi
    echo
  done
} >"${output_file}"

echo "wrote ${output_file}"
