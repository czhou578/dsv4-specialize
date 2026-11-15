use wordrank::top_words;

fn w(v: &[(&str, usize)]) -> Vec<(String, usize)> {
    v.iter().map(|(s, n)| (s.to_string(), *n)).collect()
}

#[test]
fn ties_keep_first_appearance() {
    assert_eq!(top_words("b a B a c", 3), w(&[("b", 2), ("a", 2), ("c", 1)]));
    assert_eq!(top_words("z y x", 2), w(&[("z", 1), ("y", 1)]));
}

#[test]
fn case_and_separators() {
    assert_eq!(top_words("The THE the, it's", 9), w(&[("the", 3), ("it", 1), ("s", 1)]));
    assert_eq!(top_words("naïve abc123 ABC123", 9), w(&[("abc123", 2), ("na", 1), ("ve", 1)]));
}

#[test]
fn small_k_and_empty() {
    assert_eq!(top_words("a a b", 0), w(&[]));
    assert_eq!(top_words("", 3), w(&[]));
    assert_eq!(top_words(" -- ", 3), w(&[]));
}
