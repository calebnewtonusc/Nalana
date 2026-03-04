#!/usr/bin/env bash
# Run this once after: gh auth login
# Creates the private repo on GitHub (if it doesn't exist) and pushes everything.

set -e

REPO="calebnewtonusc/nalana"

# Check if repo already exists on GitHub
if gh repo view "$REPO" >/dev/null 2>&1; then
	echo "Repo $REPO already exists — skipping creation"
else
	echo "Creating private GitHub repo: $REPO"
	gh repo create "$REPO" \
		--private \
		--description "The world's first universal voice-to-3D AI — Nalana v1"
fi

# Ensure remote is set correctly
if git remote get-url origin >/dev/null 2>&1; then
	git remote set-url origin "https://github.com/$REPO.git"
else
	git remote add origin "https://github.com/$REPO.git"
fi

# Push all branches and tags
git push -u origin main

echo ""
echo "Nalana is live at: https://github.com/$REPO"
