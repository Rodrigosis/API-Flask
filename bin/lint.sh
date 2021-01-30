#!/usr/bin/env bash
set -euo pipefail

#| Code style with flake8
echo "running flake8"
flake8 --count --max-line-length=119 --show-source --statistics --doctests wine/ tests/

#| Typing check with mypy
echo "running mypy"