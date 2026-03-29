#!/usr/bin/env bash
set -euo pipefail

# One-time cleanup before the first public GitHub push.
# This removes tracked generated or private artifacts from the Git index
# while keeping local files on disk.

git rm -r --cached --ignore-unmatch \
  claude-skill-creator \
  .codex \
  .pytest_cache \
  .vscode \
  __pycache__ \
  scripts/__pycache__ \
  src/__pycache__ \
  test/__pycache__ \
  docx \
  release \
  reports \
  tmp \
  skills/tmp \
  skills/minor-detection-v*

git add .gitignore README.md requirements.txt src scripts skills test demo_inputs minor-detection data

if [ -f app_minor_detection.py ]; then
  git add app_minor_detection.py
fi

if [ -f app_formal.py ]; then
  git add app_formal.py
fi

echo "Public GitHub release index cleanup staged."
echo "Next steps:"
echo "  1. git status --short"
echo "  2. git commit -m 'Prepare public GitHub release'"
echo "  3. git remote add origin <YOUR_GITHUB_REPO_URL>"
echo "  4. git push -u origin main"
