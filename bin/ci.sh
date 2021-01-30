#!/usr/bin/env bash
set -euo pipefail

echo '# Lint'
bash ./bin/lint.sh
echo

echo '# Test'
bash ./bin/test.sh
echo