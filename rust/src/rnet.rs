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

#[cfg(test)]
mod tests {
    use super::*;

    fn file(header: &str, data: &[u8]) -> Vec<u8> {
        let mut bytes = b"RNET0001".to_vec();
        bytes.extend((header.len() as u64).to_le_bytes());
        bytes.extend(header.as_bytes());
        bytes.extend(data);
        bytes
    }

    #[test]
    fn reads_tensors_at_their_offsets() {
        let header = r#"{"source":"x.pt","tensors":[{"name":"a","shape":[2,2],"offset":0},{"name":"b","shape":[1],"offset":64}]}"#;
        let mut data = vec![0u8; 68];
        for (i, v) in [1.0f32, 2.0, 3.0, 4.0].iter().enumerate() {
            data[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
        }
        data[64..68].copy_from_slice(&(-0.5f32).to_le_bytes());
        let path = std::env::temp_dir().join("pokeuraou-rnet-test.rnet");
        std::fs::write(&path, file(header, &data)).unwrap();
        let net = read(path.to_str().unwrap()).unwrap();
        assert_eq!(net.source, "x.pt");
        assert_eq!(net.tensors["a"], (vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]));
        assert_eq!(net.tensors["b"], (vec![1], vec![-0.5]));
        std::fs::write(&path, b"nonsense and then some").unwrap();
        assert!(read(path.to_str().unwrap()).is_err());
        // A tensor that runs past the end of the file is refused, not read short.
        std::fs::write(&path, file(header, &data[..66])).unwrap();
        match read(path.to_str().unwrap()) {
            Err(e) => assert!(e.contains("cut short"), "{e}"),
            Ok(_) => panic!("a tensor past the end of the file was read"),
        }
    }
}
