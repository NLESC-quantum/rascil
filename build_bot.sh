#!/usr/bin/env bash

git clone https://github.com/dependabot/dependabot-script.git &
PROC_ID=$!

wait $PROC_ID

BUNDLE_GEMFILE="./dependabot-script/Gemfile" bundle install -j $(nproc) --path vendor