use stock::{inventory::Inventory, orders::place};

#[test]
fn two_orders() {
    let mut inv = Inventory::default();
    inv.add("bolt", 10);
    inv.add("nut", 3);
    assert_eq!(place(&mut inv, &[("bolt", 6), ("nut", 1)]), Ok(()));
    assert_eq!(place(&mut inv, &[("nut", 2), ("bolt", 6)]), Err("bolt".to_string()));
    assert_eq!(inv.available("nut"), 2);
}
