//! Short inline identifiers.
//!
//! Every name the resolver moves around -- species, ability, item, move, status, type,
//! volatile -- is a Showdown id, and the longest one in this regulation is 23 bytes
//! (`embodyaspecthearthflame`). Holding them inline makes `Pokemon` copyable by memcpy,
//! which is the point of the port: `Pokemon.copy` is 16% of a generation run's time in
//! Python, and an id that allocates would put that cost straight back.
//!
//! Overflow panics rather than truncating. A silently shortened id would compare equal to
//! a different name, which is the one failure mode that could not be found by testing.

use serde::de::{self, Visitor};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::fmt;
use std::hash::{Hash, Hasher};

pub const ID_CAPACITY: usize = 31;

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Id {
    len: u8,
    bytes: [u8; ID_CAPACITY],
}

impl Id {
    pub const EMPTY: Id = Id { len: 0, bytes: [0; ID_CAPACITY] };

    pub fn new(text: &str) -> Id {
        let raw = text.as_bytes();
        assert!(
            raw.len() <= ID_CAPACITY,
            "id {text:?} is {} bytes, over the {ID_CAPACITY} this type holds",
            raw.len()
        );
        let mut bytes = [0u8; ID_CAPACITY];
        bytes[..raw.len()].copy_from_slice(raw);
        Id { len: raw.len() as u8, bytes }
    }

    #[inline]
    pub fn as_str(&self) -> &str {
        // Safe: the only constructor copies from a &str, so the bytes are valid UTF-8.
        unsafe { std::str::from_utf8_unchecked(&self.bytes[..self.len as usize]) }
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn starts_with(&self, prefix: &str) -> bool {
        self.as_str().starts_with(prefix)
    }
}

impl Default for Id {
    fn default() -> Self {
        Id::EMPTY
    }
}

impl PartialEq<&str> for Id {
    #[inline]
    fn eq(&self, other: &&str) -> bool {
        self.as_str() == *other
    }
}

impl fmt::Debug for Id {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self.as_str())
    }
}

impl fmt::Display for Id {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl Hash for Id {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.as_str().hash(state);
    }
}

impl std::borrow::Borrow<str> for Id {
    fn borrow(&self) -> &str {
        self.as_str()
    }
}

impl Serialize for Id {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(self.as_str())
    }
}

struct IdVisitor;

impl<'de> Visitor<'de> for IdVisitor {
    type Value = Id;

    fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("a Showdown id")
    }

    fn visit_str<E: de::Error>(self, value: &str) -> Result<Id, E> {
        if value.len() > ID_CAPACITY {
            return Err(E::custom(format!("id {value:?} is longer than {ID_CAPACITY} bytes")));
        }
        Ok(Id::new(value))
    }
}

impl<'de> Deserialize<'de> for Id {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Id, D::Error> {
        deserializer.deserialize_str(IdVisitor)
    }
}

/// FNV-1a, because the standard hasher is SipHash and these keys are two words long.
#[derive(Default, Clone, Copy)]
pub struct FnvHasher(u64);

impl Hasher for FnvHasher {
    #[inline]
    fn finish(&self) -> u64 {
        self.0
    }

    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        let mut hash = if self.0 == 0 { 0xcbf29ce484222325 } else { self.0 };
        for byte in bytes {
            hash ^= *byte as u64;
            hash = hash.wrapping_mul(0x100000001b3);
        }
        self.0 = hash;
    }
}

pub type FnvBuild = std::hash::BuildHasherDefault<FnvHasher>;
pub type IdMap<V> = std::collections::HashMap<Id, V, FnvBuild>;
