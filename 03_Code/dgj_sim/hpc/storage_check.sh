#!/bin/bash
# READ-ONLY storage check (2026-11-04, supervisor's email on /project file limits and permissions).
# Changes nothing: lists quotas, your folders in def-cbravo and rrg-cbravo, their permissions,
# folders you own that are not 700, and file counts of your scratch run folders.
set -u
echo "=== host: $(hostname -s)   user: $USER ==="
echo "=== quota report (usage and file counts) ==="
timeout 120 diskusage_report 2>/dev/null || echo "(diskusage_report not available)"
for G in def-cbravo rrg-cbravo; do
  D="/project/$G/$USER"
  echo "=== /project/$G/$USER ==="
  if [[ -d "$D" ]]; then
    stat -c '%A  %U  %n' "$D"
    echo "files: $(timeout 300 find "$D" -type f 2>/dev/null | wc -l)   size: $(timeout 300 du -sh "$D" 2>/dev/null | cut -f1)"
    echo "top-level entries:"; ls -la "$D" | head -30
  else
    echo "does not exist"
  fi
done
echo "=== folders you own under /project/def-cbravo and /project/rrg-cbravo that are NOT 700 (depth <= 4) ==="
for G in def-cbravo rrg-cbravo; do
  timeout 300 find "/project/$G" -maxdepth 4 -user "$USER" -type d ! -perm 700 -printf '%m  %p\n' 2>/dev/null | head -50
done
echo "=== home and scratch top level ==="
stat -c '%A  %n' "$HOME" "$HOME/scratch" 2>/dev/null
echo "=== scratch run folders: files per folder (largest first) ==="
R="$HOME/scratch/ai-trading-runs"
if [[ -d "$R" ]]; then
  for d in "$R"/*/; do printf '%8d  %s\n' "$(find "$d" -type f 2>/dev/null | wc -l)" "$(basename "$d")"; done | sort -rn | head -40
  echo "total files in ai-trading-runs: $(find "$R" -type f 2>/dev/null | wc -l)   size: $(du -sh "$R" 2>/dev/null | cut -f1)"
fi
echo "=== code checkouts and logs in scratch (file counts) ==="
for d in "$HOME"/scratch/AI-agent-trading-*/; do printf '%8d  %s\n' "$(find "$d" -type f 2>/dev/null | wc -l)" "$(basename "$d")"; done
echo DONE-CHECK
