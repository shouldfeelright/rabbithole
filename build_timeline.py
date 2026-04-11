"""
Build _data/timeline.json by replaying .history.json from baseline.md.
Must match update-readme.py token handling and reconstruction.
"""
from __future__ import annotations

import html
import json
import re
import sys
from pathlib import Path

import nltk
import pyinflect  # noqa: F401 — registers spacy ._.inflect
import spacy
from nltk import pos_tag, word_tokenize

nltk.download("punkt_tab", quiet=True)
nltk.download("averaged_perceptron_tagger_eng", quiet=True)

nlp = spacy.load("en_core_web_sm")

BASELINE_FILE = Path("baseline.md")
README_FILE = Path("README.md")
HISTORY_FILE = Path(".history.json")
OUTPUT_FILE = Path("_data/timeline.json")


def match_tense(original_word: str, replacement_word: str, pos_tag_str: str) -> str:
    if pos_tag_str.startswith("V"):
        doc = nlp(replacement_word)
        token = doc[0]
        inflected = token._.inflect(pos_tag_str)
        return inflected if inflected else replacement_word
    return replacement_word


def flatten_tokens(lines: list[str]) -> list[str]:
    out: list[str] = []
    for line in lines:
        out.extend(word_tokenize(line))
    return out


def reconstruct_text(lines_template: list[str], flat_tokens: list[str]) -> str:
    token_idx = 0
    final_lines: list[str] = []
    for line in lines_template:
        if line.strip() == "":
            final_lines.append("")
        else:
            line_toks = word_tokenize(line)
            recon: list[str] = []
            for _ in line_toks:
                if token_idx >= len(flat_tokens):
                    raise ValueError("Ran out of flat_tokens during reconstruct")
                recon.append(flat_tokens[token_idx])
                token_idx += 1
            line_text = " ".join(recon)
            line_text = re.sub(r"\s+([?.!,;:])", r"\1", line_text)
            final_lines.append(line_text)
    if token_idx != len(flat_tokens):
        raise ValueError(
            f"Token count mismatch: consumed {token_idx}, flat has {len(flat_tokens)}"
        )
    return "\n".join(final_lines)


def replacement_tokens(change: dict, original_at_index: str) -> list[str]:
    if change.get("display_replacement"):
        return change["display_replacement"].split()
    rep = change["replacement_word"]
    words = rep.split()
    pos = pos_tag([original_at_index])[0][1]
    words[0] = match_tense(original_at_index, words[0], pos)
    return words


def apply_change(flat_tokens: list[str], change: dict) -> tuple[int, int, str]:
    idx = change["index"]
    original = change["original_word"]
    if flat_tokens[idx] != original:
        raise ValueError(
            f"Expected token {original!r} at index {idx}, got {flat_tokens[idx]!r}"
        )
    rep = replacement_tokens(change, original)
    new_flat = flat_tokens[:idx] + rep + flat_tokens[idx + 1 :]
    flat_tokens[:] = new_flat
    meta_to = " ".join(rep)
    return idx, idx + len(rep) - 1, meta_to


def render_step_html(
    lines_template: list[str],
    flat_tokens: list[str],
    hl: tuple[int, int] | None,
    meta_from: str,
    meta_to: str,
) -> str:
    """HTML body: paragraphs; highlight inclusive global token range hl."""
    token_idx = 0
    parts: list[str] = []
    for line in lines_template:
        if line.strip() == "":
            parts.append('<p class="readme-line readme-line--empty"></p>')
            continue
        line_toks = word_tokenize(line)
        chunks: list[str] = []
        for _ in line_toks:
            g = token_idx
            tok = flat_tokens[g]
            token_idx += 1
            esc = html.escape(tok)
            if hl and hl[0] <= g <= hl[1]:
                chunks.append(
                    '<mark class="timeline-highlight" '
                    f'data-from="{html.escape(meta_from)}" data-to="{html.escape(meta_to)}">{esc}</mark>'
                )
            else:
                chunks.append(esc)
        inner = " ".join(chunks)
        parts.append(f'<p class="readme-line">{inner}</p>')
    if token_idx != len(flat_tokens):
        raise ValueError("render_step_html token mismatch")
    return "\n".join(parts)


def main() -> int:
    if not BASELINE_FILE.is_file():
        print("Missing baseline.md", file=sys.stderr)
        return 1
    if not HISTORY_FILE.is_file():
        print("Missing .history.json", file=sys.stderr)
        return 1

    lines = BASELINE_FILE.read_text(encoding="utf-8-sig").splitlines()
    history = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
    changes: list[dict] = history.get("changes") or []

    lines_work = list(lines)
    flat = flatten_tokens(lines_work)

    steps: list[dict] = []

    # Step 0: baseline
    steps.append(
        {
            "html": render_step_html(lines_work, flat, None, "", ""),
            "meta": None,
        }
    )

    for change in changes:
        hl_start, hl_end, meta_to = apply_change(flat, change)
        meta_from = change["original_word"]
        steps.append(
            {
                "html": render_step_html(
                    lines_work,
                    flat,
                    (hl_start, hl_end),
                    meta_from,
                    meta_to,
                ),
                "meta": {
                    "from": meta_from,
                    "to": meta_to,
                    "timestamp": change.get("timestamp", ""),
                },
            }
        )
        text = reconstruct_text(lines_work, flat)
        lines_work = text.splitlines()
        flat = flatten_tokens(lines_work)

    expected = README_FILE.read_text(encoding="utf-8-sig").replace("\r\n", "\n")
    final_plain = reconstruct_text(lines_work, flat)
    if final_plain != expected:
        print(
            "Replay final text does not match README.md (baseline/history drift).\n"
            f"Diff length: readme={len(expected)} replay={len(final_plain)}",
            file=sys.stderr,
        )
        return 1

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    out = {
        "step_count": len(steps),
        "max_index": len(steps) - 1,
        "steps": steps,
    }
    OUTPUT_FILE.write_text(
        json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Wrote {OUTPUT_FILE} with {len(steps)} steps.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
