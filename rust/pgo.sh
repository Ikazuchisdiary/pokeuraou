#!/usr/bin/env bash
# IKA-100: a profile-guided build of pokeuraou-damage on this machine. Off by default --
# nothing reads what this writes unless it is pointed at with POKEURAOU_RUST_NODE_BIN.
#
#     PYTHON=<interpreter> rust/pgo.sh <out-dir> [turns.json] [games-dir] [train-positions]
#
# Why it looks like this. The host toolchain is stable-x86_64-pc-windows-gnu, whose
# standard library ships without `profiler_builtins`, and the machine has no C toolchain to
# build one. The x86_64-pc-windows-gnullvm standard library does ship it, so the build is
# for that target (`rustup target add x86_64-pc-windows-gnullvm`) -- and since its linker
# is meant to be an llvm-mingw clang that is not here either, experiments/pgo-link/ turns
# rust-mingw's link-only gcc into one. llvm-profdata comes from `rustup component add
# llvm-tools`, the same LLVM as rustc, so it reads the .profraw this rustc writes.
#
# Three binaries land in <out-dir>:
#   pokeuraou-damage-llvm.exe     gnullvm, no PGO: the control, so the target change alone
#                                 is measured apart from the profile
#   pokeuraou-damage-pgogen.exe   instrumented; writes <out-dir>/profraw/*.profraw
#   pokeuraou-damage-pgo.exe      built with the merged profile
#
# Training: every case of turns.json once, then the first [train-positions] (36) recorded
# nodes of [games-dir] at width 12 through the node process (tools/bench_ika99.py train).
# tools/bench_ika99.py check/bench measure on the positions after those, never on them.
set -euo pipefail

rust="$(cd "$(dirname "$0")" && pwd)"
repo="$(cd "$rust/.." && pwd)"
out="${1:?usage: pgo.sh <out-dir> [turns.json] [games-dir] [train-positions]}"
turns="${2:-$rust/turns.json}"
games="${3:-$repo/data/ika73/w12}"
train="${4:-36}"
python="${PYTHON:-python}"
target=x86_64-pc-windows-gnullvm

sysroot="$(rustc --print sysroot)"
host_lib="$sysroot/lib/rustlib/x86_64-pc-windows-gnu"
if ! ls "$sysroot/lib/rustlib/$target/lib/"libprofiler_builtins-*.rlib >/dev/null 2>&1; then
    echo "no profiler_builtins for $target: rustup target add $target" >&2
    exit 1
fi
profdata="$host_lib/bin/llvm-profdata.exe"
if [ ! -x "$profdata" ]; then
    echo "no llvm-profdata: rustup component add llvm-tools" >&2
    exit 1
fi
export IKA100_GCC="$host_lib/bin/self-contained/x86_64-w64-mingw32-gcc.exe"
export IKA100_MINGW_LIB="$host_lib/lib/self-contained"

mkdir -p "$out"
# rustc and the binaries are Windows programs: give them Windows paths.
outw="$(cygpath -m "$out")"
rustc -O -o "$out/gnullvm_link.exe" "$rust/experiments/pgo-link/gnullvm_link.rs"
export CARGO_TARGET_X86_64_PC_WINDOWS_GNULLVM_LINKER="$outw/gnullvm_link.exe"

build() {
    local name="$1" flags="$2"
    echo "== $name  RUSTFLAGS='$flags'"
    RUSTFLAGS="$flags" cargo build --release --quiet \
        --manifest-path "$rust/Cargo.toml" --target "$target" --target-dir "$out/target-$name"
    cp "$out/target-$name/$target/release/pokeuraou-damage.exe" "$out/pokeuraou-damage-$name.exe"
}

build llvm ""
rm -rf "$out/profraw"
build pgogen "-Cprofile-generate=$outw/profraw"

format="$(head -c 4096 "$turns" | grep -o '"format_id": *"[a-z0-9]*"' | head -1 | sed 's/.*"\([a-z0-9]*\)"$/\1/')"
reg="$repo/configs/regulations/$format.json"
echo "== training: $turns ($format)"
# In a scratch directory: `turns` writes refused.json into its cwd.
mkdir -p "$out/train-cwd"
(cd "$out/train-cwd" && "$out/pokeuraou-damage-pgogen.exe" turns "$reg" "$turns" | tail -1)
echo "== training: $train nodes of $games"
PYTHONPATH="$repo/src" "$python" "$repo/tools/bench_ika99.py" train \
    --arm "pgogen=$outw/pokeuraou-damage-pgogen.exe" \
    --games-dir "$games" --train-positions "$train"

"$profdata" merge -o "$outw/pgo.profdata" "$outw/profraw"
ls -l "$out/profraw" "$out/pgo.profdata"
build pgo "-Cprofile-use=$outw/pgo.profdata"
ls -l "$out"/pokeuraou-damage-*.exe
