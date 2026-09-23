#!/usr/bin/env bash
# IKA-99: build the target-cpu / integer-roll arms side by side, one target dir each.
#
#     rust/experiments/ika99_build.sh <out-dir> [arm ...]
#
# Arms (default: all four):
#   base      the release profile as it ships (x86-64 baseline)
#   v3        -C target-cpu=x86-64-v3 (AVX2, BMI2, FMA)
#   native    -C target-cpu=native (znver5 on this machine: AVX-512)
#   introll   --features int-rolls (the 16 rolls without the f64 round trip)
#
# Each arm lands at <out-dir>/pokeuraou-damage-<arm>[.exe]. The out dir is the caller's so
# that nothing here writes rust/target, which the node bridge and other sessions read.
set -euo pipefail

rust="$(cd "$(dirname "$0")/.." && pwd)"
out="${1:?usage: ika99_build.sh <out-dir> [arm ...]}"
shift
if [ "$#" -eq 0 ]; then
    set -- base v3 native introll
fi
mkdir -p "$out"
exe=""
case "$(uname -s)" in
    MINGW* | MSYS* | CYGWIN*) exe=".exe" ;;
esac

for arm in "$@"; do
    flags=""
    features=()
    case "$arm" in
        base) ;;
        v3) flags="-C target-cpu=x86-64-v3" ;;
        native) flags="-C target-cpu=native" ;;
        introll) features=(--features int-rolls) ;;
        *)
            echo "unknown arm: $arm" >&2
            exit 2
            ;;
    esac
    echo "== $arm  RUSTFLAGS='$flags' ${features[*]:-}"
    RUSTFLAGS="$flags" cargo build --release --quiet \
        --manifest-path "$rust/Cargo.toml" \
        --target-dir "$out/target-$arm" "${features[@]}"
    cp "$out/target-$arm/release/pokeuraou-damage$exe" "$out/pokeuraou-damage-$arm$exe"
done
ls -l "$out"/pokeuraou-damage-*
