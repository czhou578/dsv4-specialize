use wordrank::top_words;

#[test]
fn counts() {
    assert_eq!(top_words("a b a c a b", 2), vec![("a".to_string(), 3), ("b".to_string(), 2)]);
}
