#!/usr/bin/env bash
set -euo pipefail

UPSTREAM_ROOT="${1:-../libopus/opus}"

if [ ! -d "$UPSTREAM_ROOT" ]; then
  echo "error: upstream root not found: $UPSTREAM_ROOT" >&2
  exit 2
fi

tmp_refs="$(mktemp)"
trap 'rm -f "$tmp_refs"' EXIT

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
      echo "MISSING_FILE $file:$line -> $p"
      fail=1
      continue
    fi

    if [ -n "$sym" ] && ! rg -n "(^|[^A-Za-z0-9_])${sym}([^A-Za-z0-9_]|$)" "$full" >/dev/null; then
      echo "MISSING_SYMBOL $file:$line -> $p"
      fail=1
    fi
  done
done < "$tmp_refs"

if [ "$fail" -ne 0 ]; then
  exit 1
fi

echo "Upstream C reference check passed."
