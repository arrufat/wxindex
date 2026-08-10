#!/bin/sh
# Install the weekly weatherindex archival timer as systemd user units.
# Usage: ./install.sh [SENSOR...]     (default: LEBL RKSS)
set -eu

unit_dir="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
project_dir="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
sensors="${*:-LEBL RKSS}"

mkdir -p "$unit_dir"
sed -e "s|@PROJECT_DIR@|$project_dir|" -e "s|@SENSORS@|$sensors|" \
    "$project_dir/systemd/weatherindex-archive.service" \
    > "$unit_dir/weatherindex-archive.service"
cp "$project_dir/systemd/weatherindex-archive.timer" "$unit_dir/"

systemctl --user daemon-reload
systemctl --user enable --now weatherindex-archive.timer
echo "Installed. Archiving [$sensors] weekly; next runs:"
systemctl --user list-timers weatherindex-archive.timer --no-pager
