#!/usr/bin/env bash
set -euo pipefail

criterion_dir="${1:-target/criterion}"
output_file="${2:-${criterion_dir}/projection-summary.md}"

mkdir -p "$(dirname "${output_file}")"

median_point_estimate() {
  local file="$1"
  jq -r '.median.point_estimate' "${file}"
}

ns_to_ms() {
  local ns="$1"
  awk -v x="${ns}" 'BEGIN { printf "%.3f", x/1000000.0 }'
}

ns_to_us() {
  local ns="$1"
  awk -v x="${ns}" 'BEGIN { printf "%.3f", x/1000.0 }'
}

ratio_r_over_c() {
  local rust_ns="$1"
  local c_ns="$2"
  awk -v r="${rust_ns}" -v c="${c_ns}" 'BEGIN { if (c == 0) { print "n/a" } else { printf "%.2f", r/c } }'
}

list_cases() {
  local dir="$1"
  if [[ ! -d "${dir}" ]]; then
    return 0
  fi
  find "${dir}" -mindepth 1 -maxdepth 1 -type d -exec basename {} \; | LC_ALL=C sort -u
}

{
  echo "# Projection Benchmark Summary"
  echo
  echo "Generated from Criterion outputs in \`${criterion_dir}\`."
  echo

  for op in encode decode; do
    if [[ "${op}" == "encode" ]]; then
      group="projection_encode_cmp"
      op_title="Encode"
    else
      group="projection_decode_cmp"
      op_title="Decode"
    fi

    echo "## ${op_title}"
    echo
    echo "| Scenario | Rust | C | Ratio (Rust/C) |"
    echo "|---|---:|---:|---:|"

    rust_cases="$(list_cases "${criterion_dir}/${group}/rust")"
    c_cases="$(list_cases "${criterion_dir}/${group}/c")"
    rows=0

    while IFS= read -r case; do
      if [[ -z "${case}" ]]; then
        continue
      fi
      if ! grep -Fxq "${case}" <<<"${c_cases}"; then
        continue
      fi

      rust_file="${criterion_dir}/${group}/rust/${case}/new/estimates.json"
      c_file="${criterion_dir}/${group}/c/${case}/new/estimates.json"
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
    done <<<"${rust_cases}"

    if [[ ${rows} -eq 0 ]]; then
      echo "| (no comparable rust/c rows found) | - | - | - |"
    fi
    echo
  done

  echo "## Matrix Apply (Rust)"
  echo
  echo "| Scenario | Rust |"
  echo "|---|---:|"
  rows=0
  while IFS= read -r case; do
    if [[ -z "${case}" ]]; then
      continue
    fi
    rust_file="${criterion_dir}/projection_matrix_apply/${case}/new/estimates.json"
    if [[ ! -f "${rust_file}" ]]; then
      continue
    fi

    rust_ns="$(median_point_estimate "${rust_file}")"
    rust_us="$(ns_to_us "${rust_ns}")"
    echo "| ${case} | ${rust_us} us |"
    rows=$((rows + 1))
  done <<<"$(list_cases "${criterion_dir}/projection_matrix_apply")"

  if [[ ${rows} -eq 0 ]]; then
    echo "| (no rows found) | - |"
  fi
  echo
} >"${output_file}"

echo "wrote ${output_file}"
