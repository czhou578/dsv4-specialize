use crate::inventory::Inventory;

/// Reserves all lines of an order, or none. Err names the first line that is short.
pub fn place(inv: &mut Inventory, lines: &[(&str, u32)]) -> Result<(), String> {
    if let Some((name, _)) = lines.iter().find(|(name, n)| inv.available(name) < *n) {
        return Err(name.to_string());
    }
    for (name, n) in lines {
        inv.reserve(name, *n);
    }
    Ok(())
}
