//! IKA-381: builds HiGHS for the port's LP (`src/lp.rs`).
//!
//! The source is `vendor/HiGHS`, the submodule pinned to the commit scipy 1.18.1 builds
//! (`scipy/HiGHS` 4f96ee8, HiGHS 1.12.0), so the port's LP is the solver
//! `equilibrium._lp` calls, handed the same model with the same options. Built with cmake
//! (the `cmake` crate) as a static library; the few C API functions the port calls are
//! declared by hand in `lp.rs`, so no bindgen (and no libclang) is needed.
//!
//! What a machine needs: cmake, and a C++ compiler for the target. On Linux the system g++.
//! On Windows (`x86_64-pc-windows-gnu`) a MinGW-w64 g++ of the release the Rust toolchain's
//! self-contained linker ships (MinGW-Builds 14.2.0, posix-seh, msvcrt), so the objects link
//! against that toolchain's libstdc++; `CXX_x86_64_pc_windows_gnu` names it (rust/README.md).

use std::path::PathBuf;

fn main() {
    let root = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let source = root.join("..").join("vendor").join("HiGHS");
    if !source.join("CMakeLists.txt").exists() {
        panic!(
            "vendor/HiGHS is empty: `git submodule update --init vendor/HiGHS` \
             (the port's LP is HiGHS 1.12.0, IKA-381)"
        );
    }
    println!("cargo:rerun-if-changed=../vendor/HiGHS/Version.txt");
    println!("cargo:rerun-if-changed=build.rs");

    let target = std::env::var("TARGET").unwrap();
    let mut config = cmake::Config::new(&source);
    if target.contains("pc-windows-gnu") {
        // cmake-rs names the compiler to cmake everywhere but on Windows, where cmake would
        // take the first one on PATH; name the one `cc` resolves (`CXX_<target>`), and the
        // resource compiler beside it (the library carries a version resource on Windows).
        let c = cc::Build::new().get_compiler();
        let cxx = cc::Build::new().cpp(true).get_compiler();
        let slashes = |p: &std::path::Path| p.to_string_lossy().replace('\\', "/");
        config.define("CMAKE_C_COMPILER", slashes(c.path()));
        config.define("CMAKE_CXX_COMPILER", slashes(cxx.path()));
        if let Some(dir) = cxx.path().parent() {
            let windres = dir.join("windres.exe");
            if windres.exists() {
                config.define("CMAKE_RC_COMPILER", slashes(&windres));
            }
        }
    }
    let installed = config
        .profile("Release")
        .define("FAST_BUILD", "ON")
        .define("BUILD_SHARED_LIBS", "OFF")
        .define("BUILD_TESTING", "OFF")
        .define("BUILD_EXAMPLES", "OFF")
        .define("BUILD_CXX_EXE", "OFF")
        .define("ZLIB", "OFF")
        .define("CMAKE_INTERPROCEDURAL_OPTIMIZATION", "FALSE")
        .build();
    println!("cargo:rustc-link-search=native={}/lib", installed.display());
    println!("cargo:rustc-link-search=native={}/lib64", installed.display());
    println!("cargo:rustc-link-lib=static=highs");

    if target.contains("pc-windows-gnu") {
        // Static, so the exe needs no libstdc++-6.dll beside it; from the self-contained
        // linker's own lib directory, the same MinGW-Builds release the compiler above is.
        println!("cargo:rustc-link-lib=static=stdc++");
        println!("cargo:rustc-link-lib=static=pthread");
    } else if target.contains("apple") {
        println!("cargo:rustc-link-lib=dylib=c++");
    } else {
        println!("cargo:rustc-link-lib=dylib=stdc++");
    }
}
