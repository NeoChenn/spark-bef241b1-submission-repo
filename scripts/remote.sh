#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

ISAAC_IMAGE="nvcr.io/nvidia/isaac-sim:6.2.0"
URL="https://rig.zeecloud.org"

has_token() {
  [[ -f .env ]] && grep -Eq '^TUNNEL_TOKEN=.+' .env
}

row() { printf '%-5s %-10s %s\n' "$1" "$2" "$3"; }

cmd_check() {
  local fail=0 driver
  driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || true)
  if [[ $driver == 580.* ]]; then row PASS driver "$driver"; else row FAIL driver "${driver:-nvidia-smi failed} (want 580.x)"; fail=1; fi

  if docker info >/dev/null 2>&1; then row PASS docker "works without sudo"; else row FAIL docker "docker info failed (docker group?)"; fail=1; fi

  if nvidia-ctk cdi list 2>/dev/null | grep -qx 'nvidia.com/gpu=all'; then row PASS cdi "nvidia.com/gpu=all"; else row FAIL cdi "nvidia.com/gpu=all missing (needs host fix: ask first)"; fail=1; fi

  if docker image inspect "$ISAAC_IMAGE" >/dev/null 2>&1; then row PASS image "$ISAAC_IMAGE present"; else row WARN image "$ISAAC_IMAGE missing: first up will pull ~27 GB"; fi

  local port
  for port in 8080 9000; do
    if ! ss -ltn 2>/dev/null | awk -v p=":$port" '$4 ~ p"$" {found=1} END {exit !found}'; then row PASS "port $port" "free"
    elif docker compose ps --status running -q web 2>/dev/null | grep -q .; then row PASS "port $port" "in use (rig web running)"
    else row FAIL "port $port" "in use by another process"; fail=1; fi
  done

  if has_token; then row PASS tunnel "TUNNEL_TOKEN set"; else row WARN tunnel "no TUNNEL_TOKEN in .env: tunnel will not start"; fi
  return $fail
}

cmd_up() {
  mkdir -p logs && chmod 777 logs
  docker compose up -d --build web sim
  if has_token; then docker compose --profile public up -d tunnel; fi
  echo
  echo "started. next: scripts/remote.sh status   (first start compiles shaders: black frames for several minutes)"
  if has_token; then echo "public: $URL"; else echo "no tunnel token: view with  ssh -N -L 8080:127.0.0.1:8080 watson  then http://localhost:8080"; fi
}

cmd_down() {
  docker compose --profile public down
}

cmd_logs() {
  docker compose --profile public logs -f --tail 100 "${1:-sim}"
}

cmd_status() {
  docker compose --profile public ps
  echo
  curl -s --max-time 3 localhost:8080/api/health || echo "web not answering on :8080"
  echo
  docker compose logs --no-log-prefix sim 2>/dev/null | grep 'fps=' | tail -1 || true
}

cmd_frame() {
  local out=${1:-logs/frame.jpg}
  mkdir -p "$(dirname "$out")"
  curl -sf --max-time 3 -o "$out" localhost:8080/api/frame.jpg && echo "saved $out" || { echo "no frame yet"; return 1; }
}

case "${1:-}" in
  check) cmd_check ;;
  up) cmd_up ;;
  down) cmd_down ;;
  logs) shift; cmd_logs "$@" ;;
  status) cmd_status ;;
  frame) shift; cmd_frame "$@" ;;
  *) echo "usage: $0 check|up|down|logs [svc]|status|frame [path]"; exit 2 ;;
esac
