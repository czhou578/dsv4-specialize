use stock::inventory::{Inventory, Item};

#[test]
fn reserve_updates_the_stored_item() {
    let mut inv = Inventory::default();
    inv.add("gear", 5);
    assert!(inv.reserve("gear", 2));
    assert_eq!(inv.get("gear"), Some(Item { qty: 5, reserved: 2 }));
    assert!(inv.reserve("gear", 3));
    assert!(!inv.reserve("gear", 1));
    assert!(!inv.reserve("none", 1));
    inv.add("gear", 4);
    assert_eq!(inv.available("gear"), 4);
}
