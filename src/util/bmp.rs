use std::io::Write;

pub fn write_bgra(path: &str, pixels: &[u8], width: u32, height: u32) -> std::io::Result<()> {
    // BMP rows are 4-byte aligned.
    let row_bytes = (width * 3 + 3) & !3;
    let pixel_size = row_bytes * height;
    let file_size = 54 + pixel_size;
    let mut f = std::fs::File::create(path)?;
    f.write_all(b"BM")?;
    f.write_all(&file_size.to_le_bytes())?;
    f.write_all(&0u32.to_le_bytes())?;
    f.write_all(&54u32.to_le_bytes())?;
    f.write_all(&40u32.to_le_bytes())?;
    f.write_all(&(width as i32).to_le_bytes())?;
    // Negative height = top-down rows.
    f.write_all(&(-(height as i32)).to_le_bytes())?;
    f.write_all(&1u16.to_le_bytes())?;
    f.write_all(&24u16.to_le_bytes())?;
    f.write_all(&0u32.to_le_bytes())?;
    f.write_all(&pixel_size.to_le_bytes())?;
    f.write_all(&[0u8; 16])?;
    let mut row = vec![0u8; row_bytes as usize];
    for y in 0..height as usize {
        for x in 0..width as usize {
            let src = y * width as usize * 4 + x * 4;
            let dst = x * 3;
            row[dst] = pixels[src];
            row[dst + 1] = pixels[src + 1];
            row[dst + 2] = pixels[src + 2];
        }
        f.write_all(&row)?;
    }
    Ok(())
}
