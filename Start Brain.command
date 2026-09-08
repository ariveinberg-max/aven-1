#!/bin/zsh
cd "${0:A:h}"
if [[ ! -x .venv/bin/python ]]; then
  echo 'Run the setup instructions in README.md first.'
  read
  exit 1
fi
exec .venv/bin/python server.py --open
