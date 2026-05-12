#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  ./build.sh [debug|release|asan] [run]
  ./build.sh clean
  ./build.sh --help

Examples:
  ./build.sh
  ./build.sh release
  ./build.sh asan run
  CXX=clang++ ./build.sh debug
  BUILD_DIR=out TARGET_NAME=forward_research ./build.sh release

Environment variables:
  CXX           C++ compiler (default: g++)
  BUILD_DIR     build output directory (default: ./build)
  TARGET_NAME   output binary name (default: forward_example)
  STD           C++ standard (default: c++17)
  NATIVE        set to 1 to add -march=native
  VERBOSE       set to 1 to print the full compile command
EOF
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${BUILD_DIR:-$ROOT_DIR/build}"
TARGET_NAME="${TARGET_NAME:-forward_example}"
STD="${STD:-c++17}"
COMPILER="${CXX:-g++}"
NATIVE="${NATIVE:-0}"
VERBOSE="${VERBOSE:-0}"

MODE="debug"
RUN_AFTER_BUILD=0

for arg in "$@"; do
    case "$arg" in
        debug|release|asan)
            MODE="$arg"
            ;;
        run)
            RUN_AFTER_BUILD=1
            ;;
        clean)
            echo "[build] removing $BUILD_DIR"
            rm -rf "$BUILD_DIR"
            exit 0
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "[build] unknown argument: $arg" >&2
            usage >&2
            exit 1
            ;;
    esac
done

COMMON_FLAGS=(
    "-std=$STD"
    -Wall
    -Wextra
    -pedantic
)

if [[ "$NATIVE" == "1" ]]; then
    COMMON_FLAGS+=(-march=native)
fi

case "$MODE" in
    debug)
        MODE_FLAGS=(-O0 -g3 -fno-omit-frame-pointer)
        ;;
    release)
        MODE_FLAGS=(-O2 -DNDEBUG)
        ;;
    asan)
        MODE_FLAGS=(-O1 -g3 -fsanitize=address,undefined -fno-omit-frame-pointer)
        ;;
esac

SOURCES=(
    "$ROOT_DIR/forward.cpp"
    "$ROOT_DIR/main.cpp"
)

mkdir -p "$BUILD_DIR"
OUTPUT_PATH="$BUILD_DIR/$TARGET_NAME"

CMD=(
    "$COMPILER"
    "${COMMON_FLAGS[@]}"
    "${MODE_FLAGS[@]}"
    "${SOURCES[@]}"
    -o
    "$OUTPUT_PATH"
)

echo "[build] mode      : $MODE"
echo "[build] compiler  : $COMPILER"
echo "[build] output    : $OUTPUT_PATH"

if [[ "$VERBOSE" == "1" ]]; then
    printf '[build] command   :'
    printf ' %q' "${CMD[@]}"
    printf '\n'
fi

"${CMD[@]}"

echo "[build] success"

if [[ "$RUN_AFTER_BUILD" == "1" ]]; then
    echo "[run] $OUTPUT_PATH"
    "$OUTPUT_PATH"
fi
