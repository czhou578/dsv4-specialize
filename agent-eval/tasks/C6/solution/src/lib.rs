pub fn top_words(text: &str, k: usize) -> Vec<(String, usize)> {
    let mut words: Vec<(String, usize)> = Vec::new();
    for t in text.split(|c: char| !c.is_ascii_alphanumeric()).filter(|t| !t.is_empty()) {
        let t = t.to_ascii_lowercase();
        match words.iter_mut().find(|(s, _)| *s == t) {
            Some(e) => e.1 += 1,
            None => words.push((t, 1)),
        }
    }
    words.sort_by(|a, b| b.1.cmp(&a.1));
    words.truncate(k);
    words
}
