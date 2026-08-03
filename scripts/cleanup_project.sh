#!/usr/bin/env bash
# Cleanup helper for the repo.
# Safe by default: dry-run unless --apply is provided.
# Compatible with macOS default bash (3.2).

set -euo pipefail
IFS=$'\n\t'

APPLY=0
for arg in "$@"; do
  case "$arg" in
    --apply) APPLY=1 ;;
    -h|--help)
      echo "Usage: scripts/cleanup_project.sh [--apply]"
      echo "Dry-run by default; pass --apply to actually delete."
      exit 0
      ;;
    *)
      echo "Unknown arg: $arg" >&2
      exit 2
      ;;
  esac
done

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "Repo root: $ROOT_DIR"
if [[ "$APPLY" -eq 1 ]]; then
  echo "Mode: APPLY (will delete files)"
else
  echo "Mode: DRY-RUN (no deletions)"
fi

# What we consider generated/temporary in this repo:
# - venvs
# - caches
# - python bytecode
# - optional SUMO artifacts (can be regenerated)
# IMPORTANT: output/ is *tracked* for demo purposes, so we do NOT delete it.

TARGET_DIRS=(
  ".venv"
  "venv"
  "ENV"
  "cache"
  "sumo"
)

# Collect __pycache__ dirs anywhere.
TMP_LIST="$(mktemp)"
find . -type d -name "__pycache__" -print > "$TMP_LIST" || true

print_item() {
  local p="$1"
  if [[ -e "$p" ]]; then
    echo "  - $p"
  fi
}

DO_DELETE=()
for d in "${TARGET_DIRS[@]}"; do
  [[ -e "$d" ]] && DO_DELETE+=("$d")
done

while IFS= read -r p; do
  [[ -n "$p" ]] && DO_DELETE+=("$p")
done < "$TMP_LIST"
rm -f "$TMP_LIST"

# Also remove .DS_Store files
TMP_DS="$(mktemp)"
find . -name ".DS_Store" -print > "$TMP_DS" || true
while IFS= read -r p; do
  [[ -n "$p" ]] && DO_DELETE+=("$p")
done < "$TMP_DS"
rm -f "$TMP_DS"

# De-duplicate list (portable-ish)
TMP_UNIQ="$(mktemp)"
printf "%s\n" "${DO_DELETE[@]}" | awk 'NF' | sort -u > "$TMP_UNIQ"

if [[ ! -s "$TMP_UNIQ" ]]; then
  echo "Nothing to clean."
  rm -f "$TMP_UNIQ"
  exit 0
fi

echo "Will remove:" 
while IFS= read -r p; do
  print_item "$p"
done < "$TMP_UNIQ"

if [[ "$APPLY" -ne 1 ]]; then
  echo
  echo "Dry-run complete. Re-run with --apply to delete."
  rm -f "$TMP_UNIQ"
  exit 0
fi

echo
echo "Deleting…"
while IFS= read -r p; do
  if [[ -d "$p" ]]; then
    rm -rf "$p"
  elif [[ -f "$p" ]]; then
    rm -f "$p"
  fi
done < "$TMP_UNIQ"
rm -f "$TMP_UNIQ"

echo "Cleanup complete."
