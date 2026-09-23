//! IKA-100: a linker for `x86_64-pc-windows-gnullvm` out of what rustup ships for `-gnu`.
//!
//! The gnullvm standard library is the only one on this machine that carries
//! `profiler_builtins`, but its target expects an llvm-mingw clang as the linker driver.
//! There is none here -- there is no C toolchain at all -- so this hands the link to the
//! link-only `x86_64-w64-mingw32-gcc` that rust-mingw ships for `-gnu`, and makes the line
//! look like the one rustc writes for `-gnu` itself:
//!
//! * the two clang-only flags (`-nolibc`, `--unwindlib=none`) are dropped;
//! * LLVM's libunwind (`-lunwind`, which nothing here provides) becomes libgcc's unwinder,
//!   which exports the same `_Unwind_*` names on SEH;
//! * the start file and the mingw/libgcc libraries come from rust-mingw's `self-contained`
//!   directory, with `-nostartfiles -nodefaultlibs`, exactly as for `-gnu`.
//!
//! Built with the stable `-gnu` rustc by `rust/pgo.sh`. Environment: `IKA100_GCC` is the
//! gcc driver, `IKA100_MINGW_LIB` the `self-contained` library directory beside it.

use std::process::{exit, Command};

fn keep(arg: &str) -> Vec<String> {
    let bare = arg.trim_matches('"');
    match bare {
        "-nolibc" | "--unwindlib=none" => vec![],
        "-lunwind" => vec!["-lgcc_eh".to_string(), "-l:libpthread.a".to_string()],
        _ => vec![arg.to_string()],
    }
}

fn var(name: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| {
        eprintln!("gnullvm_link: {name} is not set");
        exit(2);
    })
}

fn main() {
    let gcc = var("IKA100_GCC");
    let lib = var("IKA100_MINGW_LIB");
    // `-B` beside it so it finds the `ld.exe` rust-mingw ships next to it; no LTO plugin,
    // which rustc says for `-gnu` too.
    let beside = std::path::Path::new(&gcc).parent().expect("gcc directory");
    let mut args: Vec<String> = vec![
        "-fno-use-linker-plugin".to_string(),
        format!("-B{}", beside.display()),
        format!("{lib}/crt2.o"),
    ];
    for arg in std::env::args().skip(1) {
        if let Some(path) = arg.strip_prefix('@') {
            // A response file: one argument per line, filtered the same way.
            let text = std::fs::read_to_string(path).expect("response file");
            let kept: Vec<String> = text.lines().flat_map(keep).collect();
            let copy = format!("{path}.gnullvm");
            std::fs::write(&copy, kept.join("\n")).expect("filtered response file");
            args.push(format!("@{copy}"));
        } else {
            args.extend(keep(&arg));
        }
    }
    for tail in [
        "-L", &lib, "-lmsvcrt", "-lmingwex", "-lmingw32", "-lgcc", "-lmsvcrt", "-lmingwex",
        "-luser32", "-lkernel32", "-nostartfiles", "-nodefaultlibs",
    ] {
        args.push(tail.to_string());
    }
    let status = Command::new(gcc).args(&args).status().expect("run the gcc driver");
    exit(status.code().unwrap_or(1));
}
