#!/usr/bin/env bash
THIS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
DIR_PROJ_ROOT=`realpath ${THIS_DIR}/..`

cd ${DIR_PROJ_ROOT}

mkdir -p third_party

pushd third_party > /dev/null
git clone https://github.com/zig-gamedev/zig-gamedev
cd zig-gamedev
# Use ztracy v0.10.0 for Zig 0.11.0 because of the API changes in build.zig
git checkout 0feff64d0069c25cfc13601cbfb3fe4c4ed100ea
popd > /dev/null

