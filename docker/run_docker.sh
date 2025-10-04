#!/bin/bash

cd $(dirname ${BASH_SOURCE[0]})

if [ ! -f .bash_history ]; then
  touch .bash_history
fi

docker compose run --rm --name object_detection object_detection
