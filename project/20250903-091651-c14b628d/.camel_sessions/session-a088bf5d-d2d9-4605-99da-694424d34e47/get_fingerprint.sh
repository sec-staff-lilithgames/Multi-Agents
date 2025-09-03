#!/usr/bin/env bash
set -euo pipefail

if ! command -v adb >/dev/null 2>&1; then
  echo "ERROR: adb not found in PATH" >&2
  exit 1
fi

serial=$(adb devices | awk '$2=="device" {print $1; exit}')
if [ -z "${serial:-}" ]; then
  echo "ERROR: no connected device in 'device' state" >&2
  adb devices -l || true
  exit 2
fi

fp=$(adb -s "$serial" shell getprop ro.build.fingerprint | tr -d '')
echo "$fp" > fingerprint.txt
echo "FINGERPRINT: $fp"
