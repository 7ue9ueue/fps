#!/bin/bash
# Powers this VM off after 30 minutes idle, so a forgotten VM stops costing money. Idle means: no SSH
# connection, no live vm.py run, no Docker container, and a 1-minute load under 0.5. A systemd timer
# runs the check each minute; a stopped VM keeps its disk and starts again with
# `gcloud compute instances start <vm>`.
#
#   sudo bash idle_shutdown.sh install   install or update it (also the way to change IDLE_LIMIT)
#   idle_shutdown.sh status              why the VM is busy, or how long it has been idle
#
# From the Mac, on every VM:
#   gcloud compute ssh <vm> --zone=<zone> --command='cat > /tmp/idle_shutdown.sh && sudo bash /tmp/idle_shutdown.sh install' < tools/idle_shutdown.sh
set -u
LIMIT=${IDLE_LIMIT:-1800}  # seconds
SINCE=/run/idle-shutdown.since  # /run is cleared on boot: a fresh boot gets the full grace period

busy() {
    ss -Htn state established '( sport = :22 )' | grep -q . && { echo 'an SSH connection'; return; }
    for pid in /home/*/hpc_library/*.run-*/pid; do
        [ -e "$pid" ] && [ ! -e "${pid%pid}exit" ] && kill -0 "$(cat "$pid")" 2>/dev/null &&
            { echo "the vm.py run ${pid%/pid}"; return; }
    done
    [ -n "$(docker ps -q 2>/dev/null)" ] && { echo 'a Docker container'; return; }
    awk '{ exit !($1 >= 0.5) }' /proc/loadavg && { echo "load $(cut -d' ' -f1 /proc/loadavg)"; return; }
    return 1
}

case ${1-check} in
install)
    install -m 755 "${BASH_SOURCE[0]}" /usr/local/sbin/idle-shutdown
    cat > /etc/systemd/system/idle-shutdown.service <<EOF
[Unit]
Description=Power off after ${LIMIT}s idle
[Service]
Type=oneshot
Environment=IDLE_LIMIT=$LIMIT
ExecStart=/usr/local/sbin/idle-shutdown check
EOF
    cat > /etc/systemd/system/idle-shutdown.timer <<EOF
[Unit]
Description=Check each minute whether the VM is idle
[Timer]
OnBootSec=1min
OnUnitActiveSec=1min
AccuracySec=10s
[Install]
WantedBy=timers.target
EOF
    systemctl daemon-reload
    systemctl enable --now idle-shutdown.timer
    echo "idle-shutdown installed on $(hostname): power off after ${LIMIT}s idle" ;;
status)
    if why=$(busy); then echo "busy: $why"
    elif [ -s $SINCE ]; then echo "idle for $(( $(date +%s) - $(cat $SINCE) ))s of ${LIMIT}s"
    else echo 'idle (the timer has not seen it yet)'; fi ;;
check)
    if busy > /dev/null; then rm -f $SINCE; exit 0; fi
    now=$(date +%s)
    [ -s $SINCE ] || echo "$now" > $SINCE
    if (( now - $(cat $SINCE) >= LIMIT )); then
        logger -t idle-shutdown "idle for ${LIMIT}s; powering off"
        systemctl poweroff
    fi ;;
*) echo "usage: $0 [install|status|check]" >&2; exit 2 ;;
esac
