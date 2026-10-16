#!/usr/bin/env bash
# Scan files for secrets / private data before committing.
# Usage: scripts/check_secrets.sh [path...]   (default: staged files if in git, else all files)
# Extra private terms (names, hostnames) can be listed one per line in .secret_terms (gitignored via *secret*).
set -u
if [ $# -gt 0 ]; then FILES=("$@")
elif git rev-parse --git-dir >/dev/null 2>&1 && [ -n "$(git diff --cached --name-only)" ]; then mapfile -t FILES < <(git diff --cached --name-only --diff-filter=ACM)
else mapfile -t FILES < <(find . -type f -not -path './.git/*'); fi
PAT='ghp_[A-Za-z0-9]{20,}|github_pat_|gho_|hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9]{20,}|xox[bap]-|AKIA[0-9A-Z]{16}|-----BEGIN [A-Z ]*PRIVATE KEY'
PAT="$PAT|\b100\.(6[4-9]|[7-9][0-9]|1[01][0-9]|12[0-7])\.[0-9]{1,3}\.[0-9]{1,3}\b|\b(10|192\.168|172\.(1[6-9]|2[0-9]|3[01]))\.[0-9]{1,3}\.[0-9]{1,3}\b"
PAT="$PAT|[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}|/home/[a-z]|/Users/[A-Za-z]"
[ -f .secret_terms ] && while read -r t; do [ -n "$t" ] && PAT="$PAT|$t"; done < .secret_terms
fail=0
for f in "${FILES[@]}"; do
  [ -f "$f" ] || continue; case "${f#./}" in scripts/check_secrets.sh|.secret_terms) continue;; esac
  if grep -nIiE "$PAT" "$f" | grep -v 'noreply.github.com' ; then echo "^^ in $f"; fail=1; fi
done
if command -v gitleaks >/dev/null; then gitleaks detect --no-banner --redact -s . || fail=1; else echo "(gitleaks not installed; regex scan only)"; fi
[ $fail -eq 0 ] && echo "check_secrets: clean" || { echo "check_secrets: FOUND ISSUES"; exit 1; }
