#!/usr/bin/env bash
set -euo pipefail

UPSTREAM_ROOT="${1:-../libopus/opus}"
VENDORED_ROOT="libopus-sys/opus"

if [ ! -d "$UPSTREAM_ROOT" ]; then
  echo "error: upstream root not found: $UPSTREAM_ROOT" >&2
  exit 2
fi

tmp_refs="$(mktemp)"
trap 'rm -f "$tmp_refs"' EXIT

is_generated_dnn_artifact() {
  local p="$1"
  if printf "%s" "$p" | rg -q '^dnn/.+_data\.(c|h)$'; then
    return 0
  fi
  if [ "$p" = "dnn/dred_rdovae_constants.h" ]; then
    return 0
  fi
  return 1
}

rg -n "Upstream C:" src tests > "$tmp_refs"

fail=0
while IFS=: read -r file line text; do
  ref="$(printf "%s" "$text" | sed -E 's/^.*Upstream C:[[:space:]]*//; s/[`“”"]//g; s/[[:space:]]+$//; s/[.]$//')"
  IFS=',' read -r -a parts <<< "$ref"
  for p in "${parts[@]}"; do
    p="$(printf "%s" "$p" | xargs)"
    p="$(printf "%s" "$p" | sed -E 's/[[:space:]]*\(.*\)$//')"

    if ! printf "%s" "$p" | rg -q '[A-Za-z0-9_./-]+\.(c|h)'; then
      continue
    fi

    path="$(printf "%s" "$p" | sed -E 's/^([^:]+\.(c|h)).*$/\1/')"
    sym=""
    if printf "%s" "$p" | rg -q ':[A-Za-z_][A-Za-z0-9_]*'; then
      sym="$(printf "%s" "$p" | sed -E 's/^.*:([A-Za-z_][A-Za-z0-9_]*).*$/\1/')"
    fi

    full="$UPSTREAM_ROOT/$path"
    if [ ! -f "$full" ]; then
      # Some DNN model/data artifacts are generated and are not guaranteed to
      # exist in a shallow upstream checkout or in vendored sources.
      if is_generated_dnn_artifact "$path"; then
        if [ -f "$VENDORED_ROOT/$path" ]; then
          full="$VENDORED_ROOT/$path"
        else
          # Treat missing generated artifacts as valid references.
          continue
        fi
      else
        echo "MISSING_FILE $file:$line -> $p"
        fail=1
        continue
      fi
    fi

    if [ -n "$sym" ] && ! rg -n "(^|[^A-Za-z0-9_])${sym}([^A-Za-z0-9_]|$)" "$full" >/dev/null; then
      echo "MISSING_SYMBOL $file:$line -> $p"
      fail=1
    fi
  done
done < "$tmp_refs"

# Enforce style: `Upstream C:` must be the last non-empty line in its
# contiguous comment block.
while IFS= read -r file; do
  if ! awk -v file="$file" '
function is_comment(line) {
  return line ~ /^[[:space:]]*\/\/[\/!]?[[:space:]]*/
}
function payload(line, s) {
  s = line
  sub(/^[[:space:]]*\/\/[\/!]?[[:space:]]*/, "", s)
  return s
}
{
  lines[NR] = $0
}
END {
  i = 1
  while (i <= NR) {
    if (!is_comment(lines[i])) {
      i++
      continue
    }
    start = i
    while (i <= NR && is_comment(lines[i])) i++
    end = i - 1

    has_ref = 0
    last_ref = 0
    for (k = start; k <= end; k++) {
      if (index(lines[k], "Upstream C:") > 0) {
        has_ref = 1
        last_ref = k
      }
    }
    if (!has_ref) continue

    for (k = last_ref + 1; k <= end; k++) {
      if (payload(lines[k]) ~ /[^[:space:]]/) {
        printf("REF_NOT_LAST %s:%d -> %s\n", file, last_ref, lines[last_ref])
        bad = 1
        break
      }
    }
  }
  exit bad
}
  ' "$file"; then
    fail=1
  fi
done < <(rg -l "Upstream C:" src tests)

if [ "$fail" -ne 0 ]; then
  exit 1
fi

echo "Upstream C reference check passed."
