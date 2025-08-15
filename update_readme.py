import random
import re
import json
import nltk
from nltk.corpus import wordnet as wn
from nltk import pos_tag, word_tokenize
from pathlib import Path
from datetime import datetime

nltk.download('punkt_tab', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

README_FILE = Path("README.md")
HISTORY_FILE = Path(".history.json")

# Load text as lines
lines = README_FILE.read_text(encoding="utf-8").splitlines()

# Load history
if HISTORY_FILE.exists():
    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        history = json.load(f)
else:
    history = {
        "used_synonyms": {},  # word -> list of used replacements
        "changes": []         # ordered change log
    }

# Tokenize all lines separately to preserve empty lines
tokenized_lines = [word_tokenize(line) for line in lines]
tagged_lines = [pos_tag(tokens) for tokens in tokenized_lines]

# Flatten tokens and keep mapping to line/word index
flat_tokens = []
token_map = []
for line_idx, tokens in enumerate(tokenized_lines):
    for word_idx, token in enumerate(tokens):
        flat_tokens.append(token)
        token_map.append((line_idx, word_idx))

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wn.ADJ
    elif treebank_tag.startswith('V'):
        return wn.VERB
    elif treebank_tag.startswith('N'):
        return wn.NOUN
    elif treebank_tag.startswith('R'):
        return wn.ADV
    else:
        return None

# Skip all proper nouns except "Alice"
eligible_indices = [
    i for i, (word, pos) in enumerate(tagged)
    if (pos not in ('NNP', 'NNPS') or word == "Alice") and word.isalpha()
]

if not eligible_indices:
    exit()

# Shuffle to try in random order
random.shuffle(eligible_indices)
replacement_made = False

for idx in eligible_indices:
    word = flat_tokens[idx]
    pos = pos_tag([word])[0][1]
    wn_pos = get_wordnet_pos(pos)
    if not wn_pos:
        continue

    original_synsets = wn.synsets(word, pos=wn_pos)
    if not original_synsets:
        continue

    # Gather synonyms
    synonyms = set()
    for syn in original_synsets:
        for lemma in syn.lemmas():
            candidate = lemma.name().replace('_', ' ')
            if candidate.lower() != word.lower():
                synonyms.add(candidate)

    if not synonyms:
        continue

    # Don't reuse previous synonyms
    used_syns = set(history["used_synonyms"].get(word.lower(), []))
    available_syns = list(synonyms - used_syns)
    if not available_syns:
        continue

    # Compute similarity scores
    scored_syns = []
    for candidate in available_syns:
        candidate_synsets = wn.synsets(candidate, pos=wn_pos)
        if not candidate_synsets:
            continue
        score = original_synsets[0].wup_similarity(candidate_synsets[0])
        if score is None:
            score = 0.0
        scored_syns.append((candidate, score))

    if not scored_syns:
        continue
        
    # Weighted choice by similarity
    total_score = sum(score for _, score in scored_syns)
    if total_score == 0:
        replacement = random.choice([c for c, _ in scored_syns])
        chosen_score = 0.0
    else:
        weights = [score / total_score for _, score in scored_syns]
        replacement, chosen_score = random.choices(scored_syns, weights=weights, k=1)[0]

    # Replace token in flattened list
    flat_tokens[idx] = replacement
    history["used_synonyms"].setdefault(word.lower(), []).append(replacement)

    # Log change
    history["changes"].append({
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "index": idx,
        "original_word": word,
        "replacement_word": replacement,
        "similarity": round(chosen_score, 4)
    })

    replacement_made = True
    break

if replacement_made:
    # Rebuild text with original line breaks
    updated_text = " ".join(tokens)
    updated_text = re.sub(r'\s+([?.!,;:])', r'\1', updated_text)

    # Preserve original empty lines by combining with the original line structure
    final_text_lines = []
    token_idx = 0
    for line in lines:
        line_tokens = word_tokenize(line)
        reconstructed = []
        for _ in line_tokens:
            if token_idx < len(tokens):
                reconstructed.append(tokens[token_idx])
                token_idx += 1
        final_text_lines.append(" ".join(reconstructed) + ("\n" if line.endswith("\n") else ""))

    # Save updates to README and history log
    README_FILE.write_text("".join(final_text_lines), encoding="utf-8")
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)