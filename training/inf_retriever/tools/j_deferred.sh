#!/bin/bash
# Deferred Session J runs (user, 2026-10-15): E1 / E2 / E5 on plain Hungarian (j_multi_seq, j_multi_detach,
# j_multi_seq_detach, j_multi_fednorm; off/off = round 1's clean_v3_abl_qampari_multi). They must not compete
# with planned work, so this waits until ALL of these hold for an hour:
#   - every J run in j_runs.log (other than these four) and every F run in tools/f_queue.sh logged RUN DONE / RUN FAILED;
#   - every checkpoint in the J full-eval queue has test scores;
#   - no clean_v3_j_* or clean_v3_f_* job is queued or running;
#   - the hold file results/phaseC/j_deferred.hold does not exist (user, 2026-10-17: F is planning more runs;
#     Session F deletes it once all of F's planned runs are done).
# Then it closes the old J eval queue with END (its workers exit), launches the four runs (tools/j_launch.sh),
# attaches cheap-eval drivers (j_drivers.sh), and starts a new eval queue (full_queue_qampari_clean_v3_j2.txt,
# workers jc / jd) with its own j_select follower.
#   setsid nohup bash training/inf_retriever/tools/j_deferred.sh > results/phaseC/j_deferred.out 2>&1 < /dev/null &
set -u
cd /scratch/hc3337/projects/autoregressive
T=training/inf_retriever/tools
LOG=results/phaseC/phaseC.log
Q1=results/phaseC/full_queue_qampari_clean_v3_j.txt
Q2=results/phaseC/full_queue_qampari_clean_v3_j2.txt
NEW=(j_multi_seq j_multi_detach j_multi_seq_detach j_multi_fednorm)
HOLD=results/phaseC/j_deferred.hold
f_runs() { grep -oE '^  "C_[A-Za-z0-9_]+' $T/f_queue.sh | tr -d ' "' | sed 's/^/clean_v3_f_qampari_/'; }
log() { echo "[$(date '+%F %T')] [j_deferred] $*" | tee -a $LOG; }
ended() { grep -qE "\[$1\] RUN (DONE|FAILED)" $LOG 2>/dev/null; }
ready() {
  local r t ck k
  [ -e $HOLD ] && { WAIT="hold file $HOLD present (F's planned runs not finished)"; return 1; }
  for r in $(awk '{print $3}' results/phaseC/j_runs.log | sort -u) $(f_runs); do
    case $r in clean_v3_j_qampari_multi_*) continue ;; esac
    ended $r || { WAIT="$r not done"; return 1; }
  done
  while read -r t ck k; do
    [ -z "$t" ] || [ "$t" = END ] && continue
    grep -q MRecall results/phaseC/full/qampari/$t/test/eval_metrics.txt 2>/dev/null || { WAIT="full eval $t not scored"; return 1; }
  done < $Q1
  squeue -h -u $USER -o %j | grep -qE 'clean_v3_[jf]_' && { WAIT="clean_v3_j/f jobs still queued"; return 1; }
  return 0
}
log "waiting for all planned J and F runs before launching ${NEW[*]}"
since=""; last=""
while true; do
  WAIT=""
  if ready; then
    [ -z "$since" ] && since=$(date +%s) && log "everything planned looks done; launching if it stays so for 1 h"
    [ $(( $(date +%s) - since )) -ge 3600 ] && break
  else
    since=""; [ "$WAIT" != "$last" ] && log "still waiting: $WAIT"; last=$WAIT
  fi
  sleep 600
done
grep -qx END $Q1 || echo END >> $Q1
log "closed $Q1 with END; launching ${NEW[*]}"
bash $T/j_launch.sh "${NEW[@]}" || { log "ERROR: j_launch failed"; exit 1; }
runs=(); for r in "${NEW[@]}"; do runs+=(clean_v3_j_qampari_${r#j_}); done
bash $T/j_drivers.sh "${runs[@]}"
touch $Q2
for w in jc jd; do
  QUEUE=$Q2 DEV_TAG=cleanv2 PER_STEP=1 WORKER=$w setsid nohup bash $T/full_eval_queue.sh qampari \
    > results/phaseC/fullq_j_$w.out 2>&1 < /dev/null &
done
QUEUE=$Q2 setsid nohup bash $T/j_select.sh "${runs[@]}" > results/phaseC/j_select2.out 2>&1 < /dev/null &
log "launched; drivers, workers jc/jd and j_select (queue $Q2) started"
