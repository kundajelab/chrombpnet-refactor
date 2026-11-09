# shellcheck shell=bash
# The site configuration every blog 2026-015 shell script reads: paths, environments and
# cluster settings, from site.env (copy site.env.example to site.env and fill it in, or point
# BLOG_SITE_ENV at another file). Nothing here names a path of any one site.
#
# Each script finds this folder and sources this file with the same three lines:
#
#   if [[ -z "${BLOG_SCRIPTS:-}" ]]; then
#     if [[ -n "${SLURM_JOB_ID:-}" ]]; then BLOG_SCRIPTS="$(cd "${SLURM_SUBMIT_DIR}/.." && pwd)"
#     else BLOG_SCRIPTS="$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)"; fi
#   fi
#   source "${BLOG_SCRIPTS}/_site.sh"
#
# Under Slurm the script runs from a spool copy, so a job is submitted from its own folder
# (or with BLOG_SCRIPTS set).

if [[ -z "${BLOG_SCRIPTS:-}" || ! -f "${BLOG_SCRIPTS}/_site.sh" ]]; then
  echo "_site.sh: BLOG_SCRIPTS must be this folder (analysis/blog-2026-015); got '${BLOG_SCRIPTS:-}'" >&2
  exit 2
fi
export BLOG_SCRIPTS

if [[ -z "${_BLOG_SITE_LOADED:-}" ]]; then
  _blog_site_env="${BLOG_SITE_ENV:-${BLOG_SCRIPTS}/site.env}"
  if [[ ! -f "${_blog_site_env}" ]]; then
    echo "No site configuration at ${_blog_site_env}: copy ${BLOG_SCRIPTS}/site.env.example" \
         "to it (or set BLOG_SITE_ENV) and fill it in." >&2
    exit 2
  fi
  # A value already in the environment (VAR=... ./script.sh, or exported) wins over site.env's,
  # so one run can point somewhere else without editing the file.
  declare -A _blog_environment=()
  while IFS= read -r -d '' _blog_pair; do
    _blog_name="${_blog_pair%%=*}"
    [[ "${_blog_name}" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] && _blog_environment["${_blog_name}"]="${_blog_pair#*=}"
  done < <(env -0)
  # shellcheck disable=SC1090
  source "${_blog_site_env}"
  for _blog_name in "${!_blog_environment[@]}"; do
    if [[ "${!_blog_name-}" != "${_blog_environment[${_blog_name}]}" ]]; then
      printf -v "${_blog_name}" '%s' "${_blog_environment[${_blog_name}]}"
    fi
  done
  unset _blog_environment _blog_pair _blog_name
  export BLOG_SITE_ENV="${_blog_site_env}"
  _BLOG_SITE_LOADED=1
fi

# This repository, unless site.env names another checkout.
: "${CHROMBPNET_REPO:=$(cd "${BLOG_SCRIPTS}/../.." && pwd)}"
export CHROMBPNET_REPO

# require_site NAME...: stop, naming every one that site.env leaves empty.
require_site() {
  local name missing=()
  for name in "$@"; do
    [[ -n "${!name:-}" ]] || missing+=("${name}")
  done
  if (( ${#missing[@]} )); then
    echo "$(basename "${0}"): set ${missing[*]} in ${BLOG_SITE_ENV}" >&2
    exit 2
  fi
}

# site_hook NAME: run site.env's function site_NAME, if it defines one (module loads, library
# paths and the like). A site without that need defines nothing and the call does nothing.
site_hook() {
  if declare -F "site_$1" >/dev/null; then
    "site_$1"
  fi
}
