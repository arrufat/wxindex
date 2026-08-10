#!/bin/sh
# Refresh the data archive and, if anything changed, commit it.
# Usage: ./archive-and-commit.sh [SENSOR...]
set -eu

cd "$(dirname -- "$0")/.."
python3 archive.py "$@"

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "not a git repo, skipping commit"
elif git diff --quiet -- data; then
    echo "archive unchanged, nothing to commit"
else
    git add data
    git commit -q -m "Archive data update $(date -u +%Y-%m-%d)" -- data
    echo "committed: $(git log -1 --oneline)"
fi
