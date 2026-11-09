# shellcheck shell=bash
# When is a run_train_full.sh run complete? Sourced by check_runs.sh and
# run_retrain_full.sh, so the report and the resubmission cannot disagree.
#
# Complete means what the sweep's results need: a whole ChromBPNet checkpoint
# and a whole metrics JSON. The pipeline writes the metrics after the model and
# its test-set predictions, then goes on to footprints, contribution scores,
# TF-MoDISco and the report, none of which summarize.sh reads.
#
# A "Completed run" line in a Slurm log is reported, but not required. Runs
# have finished every step and still never printed it (their script was edited
# while they ran, so bash executed a fragment of the new version), or been
# stopped by the time limit during contribution scoring with their evaluation
# written. Requiring the line made run_retrain_full.sh delete and retrain
# those folds, results and all.

# metrics_complete FILE: the evaluation's metrics JSON exists and is whole.
metrics_complete() {
  local file="$1"
  [[ -s "${file}" ]] || return 1
  grep -q '"counts_metrics"' "${file}" && grep -q '"profile_metrics"' "${file}" || return 1
  [[ "$(tr -d ' \t\r\n' < "${file}" | tail -c 1)" == "}" ]]
}

# checkpoint_complete FILE: a torch.save checkpoint (a zip archive) that was
# written to the end. A save interrupted by a full disk leaves a file with no
# end-of-archive record; zipfile.is_zipfile looks for it. Without python3 only
# the size is checked.
checkpoint_complete() {
  local file="$1"
  [[ -s "${file}" ]] || return 1
  command -v python3 >/dev/null 2>&1 || return 0
  python3 -c 'import sys, zipfile; sys.exit(0 if zipfile.is_zipfile(sys.argv[1]) else 1)' "${file}"
}

# run_complete FOLD_DIR: the run in <arm>/<ASSAY>/<ENCID>/fold_<N> is complete.
run_complete() {
  checkpoint_complete "$1/model/models/chrombpnet.pt" &&
    metrics_complete "$1/model/evaluation/chrombpnet_metrics.json"
}

# logged_complete ARM ENCID FOLD LOG_DIR: some train or retrain log for
# exactly this run printed "Completed run". Prints that log's name.
logged_complete() {
  local arm="$1" encid="$2" fold="$3" logs="$4" out
  for out in "${logs}"/dnase_rc_train.*.out "${logs}"/dnase_rc_retrain.*.out; do
    [[ -e "${out}" ]] || continue
    grep -q "^Completed run" "${out}" 2>/dev/null || continue
    [[ "$(grep -m1 '^DATA_AUG:' "${out}" | awk '{print $2}')" == "${arm}" ]] || continue
    [[ "$(grep -m1 '^ENCID:'    "${out}" | awk '{print $2}')" == "${encid}" ]] || continue
    [[ "$(grep -m1 '^FOLD_NUM:' "${out}" | awk '{print $2}')" == "${fold}" ]] || continue
    basename "${out}"
    return 0
  done
  return 1
}
