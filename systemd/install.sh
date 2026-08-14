#!/bin/sh
# Install the weekly weatherindex archival timer as systemd user units.
# Usage: ./install.sh [SENSOR...]     (default: LEBL RKSS)
set -eu

unit_dir="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
project_dir="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
sensors="${*:-LEBL RKSS}"

mkdir -p "$unit_dir"
sed -e "s|@PROJECT_DIR@|$project_dir|" -e "s|@SENSORS@|$sensors|" \
    "$project_dir/systemd/wxindex-archive.service" \
    > "$unit_dir/wxindex-archive.service"
cp "$project_dir/systemd/wxindex-archive.timer" "$unit_dir/"

systemctl --user daemon-reload
systemctl --user enable --now wxindex-archive.timer
echo "Installed. Archiving [$sensors] weekly; next runs:"
systemctl --user list-timers wxindex-archive.timer --no-pager
