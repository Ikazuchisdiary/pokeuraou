//! The other road for a node's body: memory both processes can see.
//!
//! An encoded node is a JSON header line followed by the raw arrays, and until now both
//! went down the same pipe. The arrays are the large part -- measured on 2026-09-20, the
//! parent spent 26.7% of an analysis run inside `_read_exactly` -- and a pipe is the wrong
//! carrier for them: it has a buffer of tens of kilobytes, so a hundred and forty
//! megabytes is thousands of round trips in which each process waits for the other to
//! drain or fill it. `inference.py` reached the same conclusion for the same arrays going
//! the other way, and says so: "generation moves about 900 MB a second of encoded leaves,
//! which is not a thing to put down a socket".
//!
//! So the parent creates a block, names it in the request, and this attaches to it and
//! writes the body there instead. The header still goes down the pipe, and still says
//! which road the body took, because a measurement that cannot tell them apart is not a
//! measurement. Nothing else changes: `write_body` is the same function writing the same
//! bytes in the same order, only into a different sink, which is what makes "the arrays
//! are identical" a property of the code rather than a hope.
//!
//! Attaching, never creating. The parent owns the block -- it knows when the last reader
//! is done with it and it is the process that outlives this one -- and it grows it when a
//! node does not fit. A node that does not fit is answered down the pipe, so a block that
//! is too small costs one slow node and not a wrong one.
//!
//! Windows and Linux have it; anything else keeps the pipe. On Windows the block is a
//! named file mapping backed by the paging file, which is what `multiprocessing.shared_
//! memory` creates there, so four kernel32 calls reach it and no crate is needed. On
//! Linux the same object is a file under `/dev/shm`, which is tmpfs, so an ordinary
//! `write` lands in the pages the parent has mapped.

use std::io::{self, Write};

#[cfg(windows)]
mod platform {
    use std::ffi::c_void;
    use std::io::{self, Write};

    /// `FILE_MAP_WRITE`. On a `PAGE_READWRITE` mapping this grants read *and* write.
    const FILE_MAP_WRITE: u32 = 0x0002;

    #[link(name = "kernel32")]
    extern "system" {
        fn OpenFileMappingW(access: u32, inherit: i32, name: *const u16) -> *mut c_void;
        fn MapViewOfFile(
            mapping: *mut c_void,
            access: u32,
            offset_high: u32,
            offset_low: u32,
            bytes: usize,
        ) -> *mut c_void;
        fn UnmapViewOfFile(address: *const c_void) -> i32;
        fn CloseHandle(object: *mut c_void) -> i32;
    }

    pub struct Shared {
        name: String,
        mapping: *mut c_void,
        base: *mut u8,
        /// What the parent said the block holds. The view is at least this: a mapping is
        /// rounded up to a page, never down, so writing within this cannot leave it.
        capacity: usize,
        at: usize,
    }

    impl Shared {
        pub fn attach(name: &str, capacity: usize) -> Option<Shared> {
            let wide: Vec<u16> = name.encode_utf16().chain(std::iter::once(0)).collect();
            // SAFETY: `wide` is a NUL-terminated UTF-16 string that outlives the call, and
            // the handle is checked before it is mapped.
            let mapping = unsafe { OpenFileMappingW(FILE_MAP_WRITE, 0, wide.as_ptr()) };
            if mapping.is_null() {
                return None;
            }
            // SAFETY: `mapping` is a live handle; 0 bytes means "the whole block".
            let base = unsafe { MapViewOfFile(mapping, FILE_MAP_WRITE, 0, 0, 0) } as *mut u8;
            if base.is_null() {
                // SAFETY: the handle is live and is not used again.
                unsafe { CloseHandle(mapping) };
                return None;
            }
            Some(Shared { name: name.to_string(), mapping, base, capacity, at: 0 })
        }

        pub fn name(&self) -> &str {
            &self.name
        }

        pub fn capacity(&self) -> usize {
            self.capacity
        }

        pub fn rewind(&mut self) {
            self.at = 0;
        }
    }

    impl Write for Shared {
        fn write(&mut self, data: &[u8]) -> io::Result<usize> {
            if data.len() > self.capacity - self.at {
                return Err(io::Error::new(
                    io::ErrorKind::WriteZero,
                    "the shared block is smaller than the body",
                ));
            }
            // SAFETY: the view is at least `capacity` bytes and the write fits in what is
            // left of it; the regions cannot overlap, one being this process's own buffer.
            unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), self.base.add(self.at), data.len()) };
            self.at += data.len();
            Ok(data.len())
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    impl Drop for Shared {
        fn drop(&mut self) {
            // SAFETY: both were handed out by the calls above and are released once.
            unsafe {
                UnmapViewOfFile(self.base as *const c_void);
                CloseHandle(self.mapping);
            }
        }
    }
}

#[cfg(target_os = "linux")]
mod platform {
    use std::fs::{File, OpenOptions};
    use std::io::{self, Seek, SeekFrom, Write};

    /// A `/dev/shm` file, which is the same object `shm_open` hands Python.
    ///
    /// Written through the descriptor rather than mapped: tmpfs has no backing store, so
    /// the page a `write` lands in *is* the page the parent has mapped, and going that way
    /// needs no `mmap` and so no libc.
    pub struct Shared {
        name: String,
        file: File,
        capacity: usize,
        at: usize,
    }

    impl Shared {
        pub fn attach(name: &str, capacity: usize) -> Option<Shared> {
            let file = OpenOptions::new().write(true).open(format!("/dev/shm/{name}")).ok()?;
            Some(Shared { name: name.to_string(), file, capacity, at: 0 })
        }

        pub fn name(&self) -> &str {
            &self.name
        }

        pub fn capacity(&self) -> usize {
            self.capacity
        }

        pub fn rewind(&mut self) {
            self.at = 0;
            let _ = self.file.seek(SeekFrom::Start(0));
        }
    }

    impl Write for Shared {
        fn write(&mut self, data: &[u8]) -> io::Result<usize> {
            if data.len() > self.capacity - self.at {
                return Err(io::Error::new(
                    io::ErrorKind::WriteZero,
                    "the shared block is smaller than the body",
                ));
            }
            self.file.write_all(data)?;
            self.at += data.len();
            Ok(data.len())
        }

        fn flush(&mut self) -> io::Result<()> {
            self.file.flush()
        }
    }
}

#[cfg(not(any(windows, target_os = "linux")))]
mod platform {
    use std::io::{self, Write};

    /// Nowhere to attach to, so the body keeps the pipe. The type exists so the caller is
    /// the same code everywhere and the fallback is one `None` rather than a `cfg` forest.
    pub struct Shared {
        name: String,
        capacity: usize,
    }

    impl Shared {
        pub fn attach(_name: &str, _capacity: usize) -> Option<Shared> {
            None
        }

        pub fn name(&self) -> &str {
            &self.name
        }

        pub fn capacity(&self) -> usize {
            self.capacity
        }

        pub fn rewind(&mut self) {}
    }

    impl Write for Shared {
        fn write(&mut self, _data: &[u8]) -> io::Result<usize> {
            Err(io::Error::new(io::ErrorKind::Unsupported, "no shared memory on this platform"))
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }
}

pub use platform::Shared;

/// Where the parent wants this node's body put, as the request names it.
///
/// A request that carries one of these is a caller willing to hold a block; the name is
/// absent when it does not hold one yet, which is how the first node of a process asks
/// for one of exactly its own size rather than being guessed at.
#[derive(Clone)]
pub struct Target {
    pub name: Option<String>,
    pub capacity: usize,
}

/// The block attached to last time, kept so that the usual case costs nothing.
///
/// The parent reuses one block until a node outgrows it, so the name is the same request
/// after request -- and re-attaching every time would mean re-faulting every page of it,
/// which for a node of a hundred megabytes is tens of thousands of page faults to save an
/// eight-byte comparison.
#[derive(Default)]
pub struct Cache {
    held: Option<Shared>,
}

impl Cache {
    /// The block named by the target, attached if this is the first sight of it.
    ///
    /// `None` means the body goes down the pipe: the platform has no shared memory, the
    /// parent's block could not be opened, or the body does not fit in it.
    pub fn take(&mut self, target: &Target, body_bytes: usize) -> Option<&mut Shared> {
        let wanted = target.name.as_deref()?;
        if body_bytes > target.capacity {
            return None;
        }
        let stale = match &self.held {
            Some(held) => held.name() != wanted || held.capacity() != target.capacity,
            None => true,
        };
        if stale {
            self.held = Shared::attach(wanted, target.capacity);
        }
        let held = self.held.as_mut()?;
        held.rewind();
        Some(held)
    }
}

/// Writes `body` into the block, saying so, or `None` if it has to keep the pipe.
pub fn place<F>(cache: &mut Cache, target: &Target, body_bytes: usize, body: F) -> Option<()>
where
    F: FnOnce(&mut dyn Write) -> io::Result<()>,
{
    let block = cache.take(target, body_bytes)?;
    match body(block) {
        Ok(()) => Some(()),
        Err(error) => {
            // The parent is about to be told the body came down the pipe, so a half-written
            // block is harmless -- but a block that cannot be written to twice is not, and
            // dropping it here makes the next request attach afresh.
            eprintln!("node: the shared block refused the body ({error}); using the pipe");
            cache.held = None;
            None
        }
    }
}
