use pyo3::prelude::*;
use rustc_hash::FxHashMap;
use std::collections::HashSet;

const BOUNDARY: &str = "▁";
const SPECIALS: [&str; 4] = ["<pad>", "<unk>", "<s>", "</s>"];

/// Morphology-aware BPE encoder/decoder implemented in Rust.
///
/// The morpheme boundary constraint (no merges across ▁) is enforced entirely
/// at training time — pairs containing ▁ are never added to the merge table.
/// This encoder therefore requires no special-casing: ▁ pairs simply have no
/// entry in merge_ranks and are never selected by the greedy algorithm.
#[pyclass]
pub struct CoreBPE {
    vocab: FxHashMap<String, u32>,
    id_to_token: FxHashMap<u32, String>,
    /// Key: "a\tb" (tab-separated pair), Value: merge rank (lower = earlier merge)
    merge_ranks: FxHashMap<String, u32>,
    specials: HashSet<u32>,
    unk_id: u32,
}

#[pymethods]
impl CoreBPE {
    /// Build from the vocab dict and ordered merges list produced by MorphAwareBPE.
    ///
    /// vocab   — token_str → id  (same as vocab.json)
    /// merges  — [(a, b), ...]   in training order (same as merges.txt)
    #[new]
    fn new(
        vocab: std::collections::HashMap<String, u32>,
        merges: Vec<(String, String)>,
    ) -> Self {
        let vocab: FxHashMap<String, u32> = vocab.into_iter().collect();

        let id_to_token: FxHashMap<u32, String> = vocab
            .iter()
            .map(|(k, &v)| (v, k.clone()))
            .collect();

        // Store merge ranks as "a\tb" → rank so we can look up with &str (no allocation).
        let merge_ranks: FxHashMap<String, u32> = merges
            .into_iter()
            .enumerate()
            .map(|(rank, (a, b))| {
                let mut key = a;
                key.push('\t');
                key.push_str(&b);
                (key, rank as u32)
            })
            .collect();

        let specials: HashSet<u32> = SPECIALS
            .iter()
            .filter_map(|s| vocab.get(*s).copied())
            .collect();

        let unk_id = vocab.get("<unk>").copied().unwrap_or(1);

        CoreBPE {
            vocab,
            id_to_token,
            merge_ranks,
            specials,
            unk_id,
        }
    }

    /// Encode a boundary-annotated surface token into BPE token IDs.
    ///
    /// The input is a single pre-segmented token string such as "k▁um▁ain".
    /// Pairs spanning ▁ are absent from merge_ranks and are never merged.
    fn encode(&self, text: &str) -> Vec<u32> {
        if text.is_empty() {
            return vec![];
        }

        // Split into individual Unicode scalar values.
        let mut symbols: Vec<String> = text.chars().map(|c| c.to_string()).collect();

        // Reusable key buffer — allocated once per encode call, cleared each lookup.
        let mut key_buf = String::with_capacity(32);

        // Greedy BPE: find and merge the lowest-rank adjacent pair, repeat.
        loop {
            if symbols.len() < 2 {
                break;
            }

            let mut best_rank = u32::MAX;
            let mut best_i = usize::MAX;

            for i in 0..symbols.len() - 1 {
                key_buf.clear();
                key_buf.push_str(&symbols[i]);
                key_buf.push('\t');
                key_buf.push_str(&symbols[i + 1]);

                if let Some(&rank) = self.merge_ranks.get(key_buf.as_str()) {
                    if rank < best_rank {
                        best_rank = rank;
                        best_i = i;
                    }
                }
            }

            if best_rank == u32::MAX {
                break; // No more mergeable pairs.
            }

            // Merge: absorb symbols[best_i+1] into symbols[best_i].
            let right = symbols.remove(best_i + 1);
            symbols[best_i].push_str(&right);
        }

        // Map symbols → IDs, with per-character fallback for unseen symbols.
        let mut ids = Vec::with_capacity(symbols.len());
        for s in &symbols {
            if let Some(&id) = self.vocab.get(s.as_str()) {
                ids.push(id);
            } else {
                // Character-level fallback: split unknown merged symbol into chars.
                for c in s.chars() {
                    let cs = c.to_string();
                    ids.push(self.vocab.get(cs.as_str()).copied().unwrap_or(self.unk_id));
                }
            }
        }
        ids
    }

    /// Decode a sequence of token IDs back to readable text.
    ///
    /// Special tokens are silently dropped; boundary markers are removed.
    fn decode(&self, ids: Vec<u32>) -> String {
        let mut out = String::new();
        for id in ids {
            if self.specials.contains(&id) {
                continue;
            }
            if let Some(tok) = self.id_to_token.get(&id) {
                out.push_str(tok);
            }
        }
        out.replace(BOUNDARY, "")
    }
}

#[pymodule]
fn _bpe_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<CoreBPE>()?;
    Ok(())
}

// ================================================================== //
//  Rust unit tests — run with: cargo test                             //
// ================================================================== //

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal CoreBPE for testing without Python.
    fn make_bpe(extra_merges: Vec<(&str, &str)>) -> CoreBPE {
        // Base vocab: specials + individual chars used in tests
        let mut vocab = std::collections::HashMap::new();
        for (i, s) in ["<pad>", "<unk>", "<s>", "</s>"].iter().enumerate() {
            vocab.insert(s.to_string(), i as u32);
        }
        let base_chars = ['k', 'a', 'i', 'n', 'u', 'm', 'b', 't', 's', ' ', '▁'];
        for (i, c) in base_chars.iter().enumerate() {
            vocab.insert(c.to_string(), (4 + i) as u32);
        }
        // Add merged tokens produced by extra_merges
        let mut next_id = vocab.len() as u32;
        let mut merges: Vec<(String, String)> = Vec::new();
        for (a, b) in &extra_merges {
            let merged = format!("{}{}", a, b);
            if !vocab.contains_key(&merged) {
                vocab.insert(merged, next_id);
                next_id += 1;
            }
            merges.push((a.to_string(), b.to_string()));
        }
        CoreBPE::new(vocab, merges)
    }

    #[test]
    fn test_encode_single_char() {
        let bpe = make_bpe(vec![]);
        // 'k' is at index 4 in base vocab
        let ids = bpe.encode("k");
        assert_eq!(ids.len(), 1);
    }

    #[test]
    fn test_encode_applies_merge() {
        // Merge ('k', 'a') → "ka"
        let bpe = make_bpe(vec![("k", "a")]);
        let ids = bpe.encode("ka");
        // Should produce one token ("ka") not two ("k", "a")
        assert_eq!(ids.len(), 1);
    }

    #[test]
    fn test_boundary_blocks_merge() {
        // Even if ("a", "▁") existed as a merge it won't be in merge_ranks
        // because training never emits boundary-crossing pairs.
        // Simulate: give a merge ("a", "▁") — it must NOT fire since such
        // pairs are excluded at train time, so we verify manually.
        let bpe = make_bpe(vec![("k", "a")]); // only "ka" merge, no boundary merge
        let ids = bpe.encode("k▁a"); // boundary between k and a
        // "k▁a" → k, ▁, a  (3 tokens; "k▁" and "▁a" are not in merge_ranks)
        assert_eq!(ids.len(), 3);
    }

    #[test]
    fn test_encode_empty() {
        let bpe = make_bpe(vec![]);
        assert_eq!(bpe.encode(""), vec![]);
    }

    #[test]
    fn test_decode_removes_boundary() {
        let bpe = make_bpe(vec![]);
        // Encode "k▁a" → [id_k, id_▁, id_a], decode → "ka" (▁ removed)
        let ids = bpe.encode("k▁a");
        let text = bpe.decode(ids);
        assert_eq!(text, "ka");
    }

    #[test]
    fn test_decode_drops_specials() {
        let bpe = make_bpe(vec![]);
        // IDs 0-3 are <pad> <unk> <s> </s> — all should be silently dropped
        let text = bpe.decode(vec![0, 1, 2, 3]);
        assert_eq!(text, "");
    }

    #[test]
    fn test_roundtrip() {
        let bpe = make_bpe(vec![("k", "a"), ("i", "n")]);
        let original = "kain";
        let ids = bpe.encode(original);
        let decoded = bpe.decode(ids);
        assert_eq!(decoded, original);
    }
}
