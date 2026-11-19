use std::collections::HashMap;

#[derive(Clone, Debug, PartialEq)]
pub struct Item { pub qty: u32, pub reserved: u32 }

#[derive(Default)]
pub struct Inventory { items: HashMap<String, Item> }

impl Inventory {
    pub fn add(&mut self, name: &str, qty: u32) {
        self.items.entry(name.to_string()).or_insert(Item { qty: 0, reserved: 0 }).qty += qty;
    }

    pub fn get(&self, name: &str) -> Option<Item> { self.items.get(name).cloned() }

    pub fn available(&self, name: &str) -> u32 { self.get(name).map_or(0, |i| i.qty - i.reserved) }

    /// Reserves n units. Returns false and changes nothing when fewer than n are available.
    pub fn reserve(&mut self, name: &str, n: u32) -> bool {
        match self.items.get_mut(name) {
            Some(item) if item.qty - item.reserved >= n => { item.reserved += n; true }
            _ => false,
        }
    }
}
