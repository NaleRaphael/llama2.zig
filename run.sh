#!/usr/bin/env bash
./zig-out/bin/run \
    ./llama2.c/stories15M.bin \
    -t 0.8 \
    -n 256 \
    -i "One day, Lily met a Shoggoth"

