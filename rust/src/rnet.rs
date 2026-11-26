//! IKA-461: reads the `.rnet` file `src/pokeuraou/rustnet.py` writes (a trained value net as
//! raw float32 and a JSON header). Layout: `RNET0001`, the header's length as a
//! little-endian u64, the header, then the tensors at the offsets the header names (from
//! the end of the header).

use serde_json::Value;
use std::collections::HashMap;

pub struct RNet {
    pub header: Value,
    /// name -> (shape, values)
    pub tensors: HashMap<String, (Vec<usize>, Vec<f32>)>,
    /// The `.pt` file the net came from (its name, for `describe`).
    pub source: String,
}

pub fn read(path: &str) -> Result<RNet, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{path}: {e}"))?;
    if bytes.len() < 16 || &bytes[..8] != b"RNET0001" {
        return Err(format!("{path} is not an .rnet file"));
    }
    let len = u64::from_le_bytes(bytes[8..16].try_into().unwrap()) as usize;
    let data_at = 16 + len;
    if bytes.len() < data_at {
        return Err(format!("{path}: the header is cut short"));
    }
    let header: Value = serde_json::from_slice(&bytes[16..data_at]).map_err(|e| format!("{path}: {e}"))?;
    let mut tensors = HashMap::new();
    for t in header["tensors"].as_array().ok_or("the header has no tensors")? {
        let name = t["name"].as_str().ok_or("a tensor without a name")?.to_string();
        let shape: Vec<usize> = t["shape"]
            .as_array()
            .ok_or("a tensor without a shape")?
            .iter()
            .map(|n| n.as_u64().unwrap_or(0) as usize)
            .collect();
        let offset = t["offset"].as_u64().ok_or("a tensor without an offset")? as usize;
        let count: usize = shape.iter().product();
        let from = data_at + offset;
        let raw = bytes.get(from..from + count * 4).ok_or(format!("{path}: {name} is cut short"))?;
        let values = raw.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();
        tensors.insert(name, (shape, values));
    }
    let source = header["source"].as_str().unwrap_or("").to_string();
    Ok(RNet { header, tensors, source })
}
