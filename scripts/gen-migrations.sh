#!/usr/bin/env bash
# gen-migrations.sh — regenerate migrations/ from the canonical schema.
#
# The canonical schema is memory/fleet-memory-schema-kimi.sql — the SINGLE
# source of truth. The migration files in migrations/ are PRODUCED by this
# script and must never be hand-edited (a hand edit would be silently
# overwritten on the next regeneration).
#
# Output:
#   migrations/0001_registry.sql       §1 pragmas + §2 registry tables
#   migrations/0002_index_template.sql §1 pragmas + §3 index template
#                                       (@DIMS@ placeholder, substituted by
#                                       the indexer at build time)
#
# Section blocks are extracted VERBATIM (comments included) by line range:
# each §N header is preceded by a `-- ====` bar that belongs to it, and the
# section ends just before the bar that precedes the next § header.
# §4 (reference queries) is documentation, not DDL, and is not migrated.
#
# Usage: scripts/gen-migrations.sh [path-to-canonical.sql]
#   or:  FLEET_SCHEMA=... scripts/gen-migrations.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CANONICAL="${1:-${FLEET_SCHEMA:-$HOME/.openclaw/workspace/memory/fleet-memory-schema-kimi.sql}}"
OUT_DIR="$REPO_DIR/migrations"

if [[ ! -f "$CANONICAL" ]]; then
    echo "error: canonical schema not found: $CANONICAL" >&2
    exit 1
fi

mkdir -p "$OUT_DIR"
SHA256="$(sha256sum "$CANONICAL" | cut -d' ' -f1)"

# Locate the §N header lines (1-based). grep handles the UTF-8 § cleanly.
mapfile -t MARKS < <(grep -n '^-- §[0-9]' "$CANONICAL" | cut -d: -f1)
if [[ ${#MARKS[@]} -lt 4 ]]; then
    echo "error: expected at least 4 section markers (§1..§4) in $CANONICAL" >&2
    exit 1
fi
L1=${MARKS[0]} L2=${MARKS[1]} L3=${MARKS[2]} L4=${MARKS[3]}

# section N prints lines (LN - 1) .. (L(N+1) - 2): leading bar, header, body.
section() { # $1 = start header line, $2 = next header line
    sed -n "$(( $1 - 1 )), $(( $2 - 2 ))p" "$CANONICAL"
}

emit_banner() {
    local sections="$1"
    cat <<EOF
-- ============================================================================
-- GENERATED FILE — DO NOT EDIT.
-- Produced by scripts/gen-migrations.sh from the canonical schema:
--   $CANONICAL
-- Canonical sha256: $SHA256
-- Sections: $sections
-- Any manual change here will be overwritten on regeneration.
-- ============================================================================

EOF
}

{
    emit_banner "§1 pragmas + §2 registry (fleet-memory.db)"
    section "$L1" "$L2"
    section "$L2" "$L3"
} > "$OUT_DIR/0001_registry.sql"

{
    emit_banner "§1 pragmas + §3 index template (@DIMS@ substituted at build time)"
    section "$L1" "$L2"
    section "$L3" "$L4"
} > "$OUT_DIR/0002_index_template.sql"

# Sanity checks: @DIMS@ must survive in 0002, key tables must be present.
grep -q '@DIMS@' "$OUT_DIR/0002_index_template.sql" || {
    echo "error: @DIMS@ placeholder missing from 0002" >&2; exit 1; }
for t in embedding_providers index_registry reindex_runs reindex_checkpoints \
         creative_works work_subjects work_renders work_text_fts agent_decisions decision_links; do
    grep -q "CREATE TABLE IF NOT EXISTS $t\|CREATE VIRTUAL TABLE IF NOT EXISTS $t" "$OUT_DIR/0001_registry.sql" \
        || { echo "error: table $t missing from 0001" >&2; exit 1; }
done
for t in index_meta documents chunks vec_chunks; do
    grep -q "CREATE TABLE IF NOT EXISTS $t\|CREATE VIRTUAL TABLE IF NOT EXISTS $t" "$OUT_DIR/0002_index_template.sql" \
        || { echo "error: table $t missing from 0002" >&2; exit 1; }
done

echo "generated:"
echo "  $OUT_DIR/0001_registry.sql       ($(wc -l < "$OUT_DIR/0001_registry.sql") lines)"
echo "  $OUT_DIR/0002_index_template.sql ($(wc -l < "$OUT_DIR/0002_index_template.sql") lines)"
echo "canonical sha256: $SHA256"
