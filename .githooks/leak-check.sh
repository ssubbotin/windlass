#!/bin/sh
# Shared by pre-commit and commit-msg. This repository is public; names of
# private hosts, addresses and projects must not reach it. The patterns live in
# .git/info/forbidden-patterns (one extended regex per line, case-insensitive,
# '#' comments allowed) because listing them in a tracked file would publish
# exactly what they protect.
#
# Usage: leak-check.sh files         scan every staged file, whole content
#        leak-check.sh message FILE  scan a commit message
PATTERNS="$(git rev-parse --git-common-dir)/info/forbidden-patterns"
if [ ! -s "$PATTERNS" ]; then
  echo "leak-check: $PATTERNS is missing; nothing checked" >&2
  exit 0
fi
RE=$(mktemp) || exit 1
trap 'rm -f "$RE"' EXIT
# A blank line in a -f pattern file matches everything, so drop blanks and comments.
grep -vE '^[[:space:]]*(#|$)' "$PATTERNS" > "$RE"
[ -s "$RE" ] || exit 0

found=0
case "$1" in
  files)
    # Whole staged content, not just added lines: a file that already carries a
    # match is a leak the moment it is committed again. -a scans binaries too.
    for f in $(git diff --cached --name-only --diff-filter=ACMR); do
      if echo "$f" | grep -qiEf "$RE"; then
        echo "leak-check: file name matches a forbidden pattern: $f" >&2; found=1
      fi
      hits=$(git show ":$f" | grep -naiEf "$RE" | cut -c1-160)
      if [ -n "$hits" ]; then
        echo "leak-check: $f" >&2
        echo "$hits" | sed 's/^/    /' >&2
        found=1
      fi
    done
    ;;
  message)
    hits=$(grep -v '^#' "$2" | grep -niEf "$RE")
    if [ -n "$hits" ]; then
      echo "leak-check: commit message" >&2
      echo "$hits" | sed 's/^/    /' >&2
      found=1
    fi
    ;;
esac
if [ "$found" -ne 0 ]; then
  echo "leak-check: refusing; move machine details to CLAUDE.local.md (untracked)" >&2
  exit 1
fi
exit 0
