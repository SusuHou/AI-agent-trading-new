#!/bin/bash
#SBATCH --job-name=backupruns
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=06:00:00
#SBATCH --account=def-cbravo_cpu
#SBATCH --output=logs/%x_%j.out
# Back up ~/scratch/ai-trading-runs to /project/rrg-cbravo/$USER (supervisor's email, 2026-11-04):
# ONE compressed archive per run folder (few files on /project), private folder (700), originals in scratch
# untouched. Each archive is verified (its file count must equal the folder's) and checksummed.
# Skips folders whose verified archive already exists, so it is safe to rerun.
set -uo pipefail
SRC="$HOME/scratch/ai-trading-runs"
DEST="/project/rrg-cbravo/$USER/backup_ai-trading-runs"
mkdir -p -m 700 "$DEST"
REPORT="$DEST/BACKUP_REPORT.txt"
echo "backup started $(date -u --iso-8601=seconds) on $(hostname -s)" >> "$REPORT"
ok=0; bad=0; skipped=0
for d in "$SRC"/*/; do
  name=$(basename "$d"); arc="$DEST/$name.tar.gz"
  nfiles=$(find "$d" -type f | wc -l)
  if [[ -f "$arc" ]] && grep -q "^VERIFIED $name " "$REPORT"; then skipped=$((skipped+1)); continue; fi
  tar -czf "$arc.part" -C "$SRC" "$name" && mv "$arc.part" "$arc"
  chmod 600 "$arc"
  nin=$(tar -tzf "$arc" | grep -v '/$' | wc -l)
  if [[ "$nin" -eq "$nfiles" ]]; then
    echo "VERIFIED $name files=$nfiles size=$(du -h "$arc" | cut -f1) sha256=$(sha256sum "$arc" | cut -d' ' -f1)" >> "$REPORT"
    ok=$((ok+1))
  else
    echo "MISMATCH $name folder=$nfiles archive=$nin" >> "$REPORT"; bad=$((bad+1))
  fi
done
echo "backup finished $(date -u --iso-8601=seconds): verified $ok, mismatched $bad, skipped (already done) $skipped" >> "$REPORT"
chmod 600 "$REPORT"
