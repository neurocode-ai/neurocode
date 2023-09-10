#!/bin/bash
# Remove the Cpython created binaries from __pycache__ 
# recursively from the root directory. Use with care,
# will invoke `rm -rf` and remove everything named *pycache*.

function traverse() {
  for file in "$1"/*
  do
    if [ ! -d "${file}" ]; then
      continue
    else
      if [[ "${file}" == *"pycache"* ]]; then
        rm -rf "${file}"
      else
        traverse "${file}"
      fi
    fi
  done
}

echo "Revoming all __pycache__ directories from $PWD"
traverse "."
