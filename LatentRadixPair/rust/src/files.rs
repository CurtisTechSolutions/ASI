//! Files: JSON, gzipped when the name ends in .gz, written through a temporary file; gzip is sniffed on
//! read, so a file renamed either way still loads.

use flate2::read::GzDecoder;
use flate2::write::GzEncoder;
use flate2::Compression;
use std::io::{Read, Write};
use std::path::Path;

/// Reads a file, inflating it when it starts with the gzip magic.
pub fn read_file(path: &str) -> std::io::Result<Vec<u8>> {
    let data = std::fs::read(path)?;
    if data.len() >= 2 && data[0] == 0x1f && data[1] == 0x8b {
        let mut out = Vec::new();
        GzDecoder::new(&data[..]).read_to_end(&mut out)?;
        return Ok(out);
    }
    Ok(data)
}

/// Writes data to path atomically, gzipped when the name ends in .gz.
pub fn write_file(path: &str, data: &[u8]) -> std::io::Result<()> {
    let bytes = if path.ends_with(".gz") {
        let mut enc = GzEncoder::new(Vec::new(), Compression::default());
        enc.write_all(data)?;
        enc.finish()?
    } else {
        data.to_vec()
    };
    if let Some(dir) = Path::new(path).parent() {
        if !dir.as_os_str().is_empty() {
            std::fs::create_dir_all(dir)?;
        }
    }
    let tmp = format!("{path}.tmp");
    std::fs::write(&tmp, bytes)?;
    std::fs::rename(&tmp, path)
}
