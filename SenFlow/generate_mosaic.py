"""MOSAIC: hybrid-document corpus generator.

Produces train/val/test splits of documents where a subset of sentences are
replaced by LLM-generated completions (DeepSeek-V3.2 / Kimi K2), filtered
through perplexity-consistency checks.

Usage:
    python generate_mosaic.py --domain pubmed --generator deepseek --target 4000
    python generate_mosaic.py --domain xsum   --generator kimi     --target 4000 --debug
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from threading import Lock
from typing import Dict, List, Optional, Tuple

import nltk
import torch
from datasets import load_dataset
from openai import OpenAI
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SEED = 42
random.seed(SEED)

GAMMA = 0.3
BLOCK_CHOICES = [1, 2, 3]
BLOCK_WEIGHTS = [0.3, 0.5, 0.2]

MAX_SEQ_LEN = 20
MAX_WORD_LEN = 96
MIN_SOURCE_SENTENCES = 15
MIN_AI_WORDS = 4
AI_LEN_MIN_RATIO = 0.25
AI_LEN_MAX_RATIO = 2.5

PPL_MARGIN = 15.0
TEMPERATURE = 0.7
MAX_API_RETRIES = 3
API_RETRY_DELAY = 2.0

NUM_WORKERS = 30
SPLIT_RATIOS = (0.70, 0.15, 0.15)

SUBGROUP_MAP = {
    ("pubmed", "deepseek"): 0,
    ("pubmed", "kimi"): 1,
    ("xsum", "deepseek"): 2,
    ("xsum", "kimi"): 3,
}

# Paths — override via environment variables.
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
MOONSHOT_API_KEY = os.environ.get("MOONSHOT_API_KEY")

ORACLE_MODEL_PATH = os.environ.get(
    "ORACLE_MODEL_PATH",
    "/root/autodl-tmp/my_models/LLM-Research/Meta-Llama-3.1-8B-Instruct",
)
PUBMED_LOCAL_PATH = os.environ.get(
    "PUBMED_LOCAL_PATH", "/root/autodl-tmp/pubmed_raw/train*.parquet"
)
XSUM_LOCAL_PATH = os.environ.get(
    "XSUM_LOCAL_PATH", "/root/autodl-tmp/xsum_raw/train.parquet"
)
OUTPUT_DIR = os.environ.get("MOSAIC_OUTPUT_DIR", "/root/autodl-tmp/datasets_v2")

# ---------------------------------------------------------------------------
# Sentence splitting
# ---------------------------------------------------------------------------
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)


def split_sentences_pubmed(text: str) -> List[str]:
    text = re.sub(r"\s+\.\s+", ". ", text)
    text = re.sub(r"\s+,\s+", ", ", text)
    text = re.sub(r"\s+;\s+", "; ", text)
    text = re.sub(r"\s+:\s+", ": ", text)
    sents = nltk.sent_tokenize(text)
    return [s.strip() for s in sents if s.strip() and len(s.strip()) > 10]


def split_sentences_xsum(text: str) -> List[str]:
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    return [line for line in lines if len(line) > 10]


SENTENCE_SPLITTER = {
    "pubmed": split_sentences_pubmed,
    "xsum": split_sentences_xsum,
}


def compute_sentence_spans(sentences: List[str]) -> Tuple[str, List[List[int]]]:
    spans: List[List[int]] = []
    full = ""
    for i, s in enumerate(sentences):
        if i > 0:
            full += " "
        start = len(full)
        full += s
        spans.append([start, len(full)])
    return full, spans


# ---------------------------------------------------------------------------
# PubMed format normalization
# ---------------------------------------------------------------------------
def normalize_to_pubmed_format(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"([a-z0-9])\s*([.!?])", r"\1 \2", text)
    text = re.sub(r"\s*([,;:])\s*", r" \1 ", text)
    text = re.sub(r"\s*\(\s*", " ( ", text)
    text = re.sub(r"\s*\)\s*", " ) ", text)
    text = re.sub(r"([a-z])-([a-z])", r"\1 - \2", text)
    text = re.sub(r"\s*/\s*", " / ", text)
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------------------
# Output cleaning
# ---------------------------------------------------------------------------
_REASONING_TAG_PATTERNS = [
    re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE),
    re.compile(r"<thinking>.*?</thinking>", re.DOTALL | re.IGNORECASE),
    re.compile(r"<reasoning>.*?</reasoning>", re.DOTALL | re.IGNORECASE),
]

_META_PREFIXES = [
    "new completed document:",
    "here is the completed document:",
    "here's the completed document:",
    "completed document:",
    "here is the document with masks filled:",
    "here are the filled sentences:",
    "filled document:",
    "i'll fill in each [mask]:",
    "let me fill in the [mask] tokens:",
    "sure, here is",
    "sure, here's",
    "sure!",
    "certainly!",
    "okay, here is",
    "okay, here's",
    "here you go:",
]


def clean_generated_text(raw: str) -> str:
    if not raw:
        return ""
    text = raw
    for pat in _REASONING_TAG_PATTERNS:
        text = pat.sub("", text)
    text = text.strip()
    for _ in range(3):
        text_lower = text.lower()
        for prefix in _META_PREFIXES:
            if text_lower.startswith(prefix):
                text = text[len(prefix):].lstrip(":：\n\r \t")
                break
        else:
            break
    text = re.sub(r"\*{2,}", "", text)
    text = re.sub(r"(?<!\w)\*(?!\w)", "", text)
    text = re.sub(r"`{1,3}", "", text)
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*[-•]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\d+\.\s+", "", text, flags=re.MULTILINE)
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("‘", "'").replace("’", "'")
    text = text.replace("—", "-").replace("–", "-")
    text = text.replace(" ", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Block-random masking
# ---------------------------------------------------------------------------
def block_random_mask(
    num_sentences: int, gamma: float = GAMMA
) -> List[int]:
    total_mask = max(1, int(round(num_sentences * gamma)))
    total_mask = min(total_mask, max(1, num_sentences - 2))
    num_blocks = random.choices(BLOCK_CHOICES, weights=BLOCK_WEIGHTS, k=1)[0]
    num_blocks = min(num_blocks, total_mask)

    if num_blocks == 1:
        block_sizes = [total_mask]
    else:
        cuts = sorted(random.sample(range(1, total_mask), num_blocks - 1))
        block_sizes = (
            [cuts[0]]
            + [cuts[i] - cuts[i - 1] for i in range(1, len(cuts))]
            + [total_mask - cuts[-1]]
        )

    masked_indices: List[int] = []
    for block_size in block_sizes:
        valid_starts = []
        for start in range(num_sentences - block_size + 1):
            buffer_range = set(
                range(
                    max(0, start - 1),
                    min(num_sentences, start + block_size + 1),
                )
            )
            if not buffer_range.intersection(masked_indices):
                valid_starts.append(start)
        if not valid_starts:
            continue
        start = random.choice(valid_starts)
        masked_indices.extend(range(start, start + block_size))

    return sorted(set(masked_indices))


def build_masked_document(
    sentences: List[str], mask_indices: List[int]
) -> str:
    out: List[str] = []
    mask_counter = 0
    for i, s in enumerate(sentences):
        if i in mask_indices:
            out.append(f"[MASK_{mask_counter}]")
            mask_counter += 1
        else:
            out.append(s)
    return " ".join(out)


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------
def build_fill_prompt(
    masked_doc: str, num_masks: int, domain: str
) -> str:
    if domain == "pubmed":
        style_note = (
            "IMPORTANT FORMAT RULES for this biomedical document:\n"
            "- Use ALL LOWERCASE letters throughout.\n"
            "- Insert a space BEFORE and AFTER every punctuation mark "
            "(periods, commas, semicolons, colons, parentheses).\n"
            "- Example format: 'the results showed a significant decrease "
            "( p < 0.05 ) in mortality rates .'\n"
            "- Content should match formal biomedical scientific writing "
            "with precise terminology.\n"
        )
    else:
        style_note = (
            "FORMAT RULES for this news document:\n"
            "- Use normal capitalization (sentence case).\n"
            "- Use standard punctuation (no extra spaces around punctuation).\n"
            "- Content should match concise, factual news reporting style "
            "(BBC-style).\n"
        )

    return (
        "You are completing a document where some sentences have been "
        "replaced by [MASK_i] placeholders.\n"
        "Your task: for each [MASK_i], write ONE sentence that fits "
        "naturally in context.\n\n"
        f"{style_note}\n"
        "OUTPUT RULES:\n"
        "1. Return ONLY the filled sentences, one per line, in exactly "
        "this format:\n"
        "   [MASK_0]: <your sentence>\n"
        "   [MASK_1]: <your sentence>\n"
        "   ...\n"
        "2. Do NOT rewrite or modify any non-masked sentences.\n"
        "3. Do NOT add explanations, preamble, or markdown formatting.\n"
        "4. Each filled sentence should be a complete sentence with "
        "proper punctuation.\n"
        "5. Keep each filled sentence similar in length to surrounding "
        "sentences.\n\n"
        f"Document with masks:\n{masked_doc}\n\n"
        f"Now output the {num_masks} filled sentences in the required format:"
    )


# ---------------------------------------------------------------------------
# LLM API callers
# ---------------------------------------------------------------------------
def _extract_content_from_reasoning(reasoning: str) -> str:
    """Try to extract the final answer from a reasoning trace."""
    if "</think>" in reasoning.lower():
        parts = re.split(r"</think>", reasoning, flags=re.IGNORECASE)
        content = parts[-1].strip()
        if content:
            return content
    mask_lines = re.findall(
        r"\[mask_\d+\]\s*[:：]\s*.+",
        reasoning,
        re.IGNORECASE,
    )
    if mask_lines:
        return "\n".join(mask_lines)
    return reasoning


def call_deepseek(prompt: str) -> Tuple[str, str]:
    if not DEEPSEEK_API_KEY:
        raise ValueError("DEEPSEEK_API_KEY not set")
    client = OpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url="https://api.deepseek.com/v1",
        timeout=120.0,
    )
    for attempt in range(MAX_API_RETRIES):
        try:
            r = client.chat.completions.create(
                model="deepseek-reasoner",
                messages=[{"role": "user", "content": prompt}],
                temperature=TEMPERATURE,
                max_tokens=2000,
            )
            msg = r.choices[0].message
            content = (msg.content or "").strip()
            reasoning = getattr(msg, "reasoning_content", "") or ""
            if not content and reasoning:
                content = _extract_content_from_reasoning(reasoning)
            return content, reasoning
        except Exception as exc:
            if attempt < MAX_API_RETRIES - 1:
                time.sleep(API_RETRY_DELAY * (attempt + 1))
                continue
            raise RuntimeError(f"DeepSeek API failed: {exc}") from exc
    raise RuntimeError("DeepSeek API: unreachable")


def call_kimi(prompt: str) -> Tuple[str, str]:
    if not MOONSHOT_API_KEY:
        raise ValueError("MOONSHOT_API_KEY not set")
    client = OpenAI(
        api_key=MOONSHOT_API_KEY,
        base_url="https://api.moonshot.cn/v1",
        timeout=120.0,
    )
    for attempt in range(MAX_API_RETRIES):
        try:
            r = client.chat.completions.create(
                model="kimi-k2-0905-preview",
                messages=[{"role": "user", "content": prompt}],
                temperature=TEMPERATURE,
                max_tokens=2000,
            )
            msg = r.choices[0].message
            content = (msg.content or "").strip()
            reasoning = getattr(msg, "reasoning_content", "") or ""
            if not content and reasoning:
                content = _extract_content_from_reasoning(reasoning)
            return content, reasoning
        except Exception as exc:
            if attempt < MAX_API_RETRIES - 1:
                time.sleep(API_RETRY_DELAY * (attempt + 1))
                continue
            raise RuntimeError(f"Kimi API failed: {exc}") from exc
    raise RuntimeError("Kimi API: unreachable")


GENERATOR_DISPATCH = {"deepseek": call_deepseek, "kimi": call_kimi}


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------
_MASK_FILL_PATTERN = re.compile(
    r"\[MASK_(\d+)\]\s*[:：]\s*(.+?)(?=\n\s*\[MASK_\d+\]|\Z)",
    re.DOTALL | re.IGNORECASE,
)


def parse_fills(
    generated_text: str, num_masks: int, domain: str
) -> Optional[List[str]]:
    cleaned = clean_generated_text(generated_text)
    matches = _MASK_FILL_PATTERN.findall(cleaned)
    if len(matches) < num_masks:
        return None
    fills = [""] * num_masks
    for idx_str, content in matches:
        idx = int(idx_str)
        if 0 <= idx < num_masks:
            first_line = content.strip().split("\n")[0].strip()
            if domain == "pubmed":
                first_line = normalize_to_pubmed_format(first_line)
            fills[idx] = first_line
    if any(not f for f in fills):
        return None
    return fills


# ---------------------------------------------------------------------------
# Perplexity oracle
# ---------------------------------------------------------------------------
def _load_oracle():
    print(f"[INFO] Loading oracle: {ORACLE_MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(ORACLE_MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        ORACLE_MODEL_PATH,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    ).eval()
    return tokenizer, model


oracle_tokenizer, oracle_model = _load_oracle()
ppl_lock = Lock()


@torch.no_grad()
def compute_perplexity(text: str, max_length: int = 2048) -> float:
    inputs = oracle_tokenizer(
        text, return_tensors="pt", truncation=True, max_length=max_length
    )
    inputs = {k: v.to(oracle_model.device) for k, v in inputs.items()}
    if inputs["input_ids"].shape[1] < 2:
        return float("inf")
    outputs = oracle_model(**inputs, labels=inputs["input_ids"])
    return math.exp(min(outputs.loss.item(), 20.0))


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_source_dataset(domain: str):
    if domain == "pubmed":
        path, field = PUBMED_LOCAL_PATH, "article"
    elif domain == "xsum":
        path, field = XSUM_LOCAL_PATH, "document"
    else:
        raise ValueError(f"Unknown domain: {domain}")
    print(f"[INFO] Loading {domain} from {path}")
    ds = load_dataset("parquet", data_files=path, split="train")
    return ds.shuffle(seed=SEED), field


# ---------------------------------------------------------------------------
# Sample dataclass & per-document processing
# ---------------------------------------------------------------------------
@dataclass
class HybridSample:
    doc_id: str
    domain: str
    generator: str
    subgroup_id: int
    original_text: str
    hybrid_text: str
    sentence_spans: List[List[int]]
    ai_indices: List[int]
    num_sentences: int
    mask_ratio: float
    original_ppl: float
    hybrid_ppl: float


def _make_failure_stats() -> Dict[str, int]:
    return {
        "too_short": 0,
        "api_error": 0,
        "parse_fail": 0,
        "length_fail": 0,
        "ppl_fail": 0,
        "success": 0,
    }


def process_one_sample(
    example,
    text_field: str,
    domain: str,
    generator_name: str,
    sample_idx: int,
    stats: Dict[str, int],
    stats_lock: Lock,
    debug: bool = False,
) -> Optional[HybridSample]:
    raw_text = example.get(text_field, "")
    if not raw_text or len(raw_text) < 100:
        with stats_lock:
            stats["too_short"] += 1
        return None

    splitter = SENTENCE_SPLITTER[domain]
    sentences = splitter(raw_text)
    min_sents = 8 if domain == "xsum" else MIN_SOURCE_SENTENCES
    if len(sentences) < min_sents:
        with stats_lock:
            stats["too_short"] += 1
        return None

    sentences = sentences[:MAX_SEQ_LEN]
    sentences = [
        (
            " ".join(s.split()[:MAX_WORD_LEN])
            if len(s.split()) > MAX_WORD_LEN
            else s
        )
        for s in sentences
    ]
    num_sentences = len(sentences)

    mask_indices = block_random_mask(num_sentences, gamma=GAMMA)
    if not mask_indices:
        return None

    masked_doc = build_masked_document(sentences, mask_indices)
    prompt = build_fill_prompt(masked_doc, len(mask_indices), domain)

    try:
        call_fn = GENERATOR_DISPATCH[generator_name]
        raw_response, _ = call_fn(prompt)
    except Exception as exc:
        if debug:
            print(f"[DEBUG] API error sample {sample_idx}: {exc}")
        with stats_lock:
            stats["api_error"] += 1
        return None

    fills = parse_fills(raw_response, len(mask_indices), domain)
    if fills is None:
        if debug and stats["parse_fail"] < 3:
            print(
                f"\n[DEBUG] Parse fail sample {sample_idx}, "
                f"expected {len(mask_indices)} masks"
            )
            print(f"[DEBUG] Response preview: {raw_response[:400]}\n")
        with stats_lock:
            stats["parse_fail"] += 1
        return None

    avg_human_len = (
        sum(
            len(sentences[i].split())
            for i in range(num_sentences)
            if i not in mask_indices
        )
        / max(1, num_sentences - len(mask_indices))
    )

    for fill in fills:
        fill_len = len(fill.split())
        if (
            fill_len < MIN_AI_WORDS
            or fill_len > avg_human_len * AI_LEN_MAX_RATIO
            or fill_len < avg_human_len * AI_LEN_MIN_RATIO
        ):
            with stats_lock:
                stats["length_fail"] += 1
            return None

    hybrid_sentences = list(sentences)
    for fill_idx, sent_idx in enumerate(mask_indices):
        hybrid_sentences[sent_idx] = fills[fill_idx]

    original_full, _ = compute_sentence_spans(sentences)
    hybrid_full, hybrid_spans = compute_sentence_spans(hybrid_sentences)

    ppl_ori = compute_perplexity(original_full)
    ppl_new = compute_perplexity(hybrid_full)

    if abs(ppl_new - ppl_ori) > PPL_MARGIN or ppl_ori > 150 or ppl_new > 150:
        with stats_lock:
            stats["ppl_fail"] += 1
        return None

    with stats_lock:
        stats["success"] += 1
        cur_success = stats["success"]

    if debug and cur_success <= 3:
        print(f"\n[DEBUG] ===== Sample {sample_idx} SUCCESS =====")
        print(f"[DEBUG] AI indices: {mask_indices}")
        print(f"[DEBUG] PPL: ori={ppl_ori:.2f} new={ppl_new:.2f}")
        for mi in mask_indices:
            print(f"[DEBUG]   AI sent [{mi}]: {hybrid_sentences[mi][:120]}")
        first_human_idx = next(
            (i for i in range(num_sentences) if i not in mask_indices), 0
        )
        print(
            f"[DEBUG]   Human [{first_human_idx}]: "
            f"{sentences[first_human_idx][:120]}"
        )
        print("[DEBUG] =====================================\n")

    return HybridSample(
        doc_id=f"{domain}_{generator_name}_{sample_idx:06d}",
        domain=domain,
        generator=generator_name,
        subgroup_id=SUBGROUP_MAP[(domain, generator_name)],
        original_text=original_full,
        hybrid_text=hybrid_full,
        sentence_spans=hybrid_spans,
        ai_indices=mask_indices,
        num_sentences=num_sentences,
        mask_ratio=round(len(mask_indices) / num_sentences, 3),
        original_ppl=round(ppl_ori, 3),
        hybrid_ppl=round(ppl_new, 3),
    )


# ---------------------------------------------------------------------------
# Pipeline: generate one (domain, generator) cell
# ---------------------------------------------------------------------------
def generate_cell(
    domain: str,
    generator: str,
    target: int,
    debug: bool = False,
) -> List[HybridSample]:
    print(f"\n{'=' * 70}")
    print(
        f"[CELL] domain={domain}  generator={generator}  "
        f"target={target}  debug={debug}"
    )
    print(f"{'=' * 70}")

    ds, text_field = load_source_dataset(domain)

    collected: List[HybridSample] = []
    collected_lock = Lock()
    stats = _make_failure_stats()
    stats_lock = Lock()
    pbar = tqdm(total=target, desc=f"{domain}/{generator}")

    def _worker(args) -> None:
        with collected_lock:
            if len(collected) >= target:
                return
        ex, idx = args
        result = process_one_sample(
            ex, text_field, domain, generator, idx,
            stats=stats, stats_lock=stats_lock, debug=debug,
        )
        if result is not None:
            with collected_lock:
                if len(collected) < target:
                    collected.append(result)
                    pbar.update(1)

    multiplier = 3 if debug else 5
    pool_size = min(len(ds), target * multiplier)
    tasks = [(ds[i], i) for i in range(pool_size)]

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = []
        for t in tasks:
            with collected_lock:
                if len(collected) >= target:
                    break
            futures.append(executor.submit(_worker, t))
        for _ in as_completed(futures):
            with collected_lock:
                if len(collected) >= target:
                    for f in futures:
                        f.cancel()
                    break
    pbar.close()

    total = sum(stats.values())
    print(f"\n[STATS] Success:     {stats['success']}")
    print(f"[STATS] Too short:   {stats['too_short']}")
    print(f"[STATS] API error:   {stats['api_error']}")
    print(f"[STATS] Parse fail:  {stats['parse_fail']}")
    print(f"[STATS] Length fail: {stats['length_fail']}")
    print(f"[STATS] PPL fail:    {stats['ppl_fail']}")
    print(f"[STATS] Total:       {total}")

    return collected[:target]


def save_cell(
    samples: List[HybridSample], domain: str, generator: str
) -> str:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    rng = random.Random(SEED)
    shuffled = samples[:]
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * SPLIT_RATIOS[0])
    n_val = int(n * SPLIT_RATIOS[1])
    train_s = shuffled[:n_train]
    val_s = shuffled[n_train : n_train + n_val]
    test_s = shuffled[n_train + n_val :]

    output = {
        "metadata": {
            "domain": domain,
            "generator": generator,
            "subgroup_id": SUBGROUP_MAP[(domain, generator)],
            "oracle_model": "Llama-3.1-8B-Instruct",
            "mask_gamma": GAMMA,
            "mask_strategy": "block_random (1-3 blocks)",
            "max_seq_len": MAX_SEQ_LEN,
            "max_word_len": MAX_WORD_LEN,
            "ppl_margin": PPL_MARGIN,
            "total_samples": n,
            "split": {
                "train": len(train_s),
                "val": len(val_s),
                "test": len(test_s),
            },
            "version": "v2.1",
        },
        "train": [asdict(s) for s in train_s],
        "val": [asdict(s) for s in val_s],
        "test": [asdict(s) for s in test_s],
    }

    save_path = os.path.join(OUTPUT_DIR, f"{domain}_{generator}_{n}.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(
        f"[SAVE] {save_path}\n"
        f"       train={len(train_s)}  val={len(val_s)}  test={len(test_s)}"
    )
    return save_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="MOSAIC hybrid-document corpus generator"
    )
    parser.add_argument(
        "--domain", required=True,
        choices=["pubmed", "xsum"],
        help="Source domain",
    )
    parser.add_argument(
        "--generator", required=True,
        choices=["deepseek", "kimi"],
        help="LLM generator to use for filling masks",
    )
    parser.add_argument(
        "--target", type=int, default=4000,
        help="Number of hybrid documents to generate (default: 4000)",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Enable debug output for the first few samples",
    )
    args = parser.parse_args()

    start = time.time()
    samples = generate_cell(
        args.domain, args.generator, args.target, debug=args.debug
    )

    if samples:
        save_cell(samples, args.domain, args.generator)
    else:
        print("[ERROR] No samples collected!")

    elapsed = (time.time() - start) / 60
    print(f"\n[DONE] Time: {elapsed:.1f} min  |  Samples: {len(samples)}")


if __name__ == "__main__":
    main()
