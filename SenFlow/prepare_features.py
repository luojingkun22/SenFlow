import argparse
import json
import os
import re

import nltk
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def denormalize_pubmed_sentence(text: str) -> str:
    text = re.sub(r"\s+([.!?,;:])", r"\1", text)
    text = re.sub(r"\(\s+", "(", text)
    text = re.sub(r"\s+\)", ")", text)
    text = re.sub(r"\s+/\s+", "/", text)
    text = re.sub(r"\s+-\s+", "-", text)
    text = text.strip()
    if text:
        text = text[0].upper() + text[1:]
    text = re.sub(
        r"([.!?]\s+)([a-z])",
        lambda m: m.group(1) + m.group(2).upper(), text)
    return re.sub(r"\s+", " ", text).strip()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_SEQ_LEN = 20
MAX_WORD_LEN = 96
MAX_DOC_TOKENS = 2048
HIDDEN_DIM = 4096
DIVEYE_DIM = 4

LLAMA_PATH = os.environ.get(
    "LLAMA_PATH", "/root/autodl-tmp/lora_models/llama_aligned_merged")

print(f"[INFO] Loading Llama-3.1-8B from {LLAMA_PATH}")
print(f"[INFO] Device: {device}")

ref_tokenizer = AutoTokenizer.from_pretrained(LLAMA_PATH)
if ref_tokenizer.pad_token is None:
    ref_tokenizer.pad_token = ref_tokenizer.eos_token

ref_model = AutoModelForCausalLM.from_pretrained(
    LLAMA_PATH,
    device_map="auto",
    torch_dtype=torch.bfloat16,
).eval()

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)


def split_sentences_pubmed(text: str):
    text = re.sub(r"\s+\.\s+", ". ", text)
    text = re.sub(r"\s+,\s+", ", ", text)
    text = re.sub(r"\s+;\s+", "; ", text)
    text = re.sub(r"\s+:\s+", ": ", text)
    sents = nltk.sent_tokenize(text)
    return [s.strip() for s in sents if s.strip() and len(s.strip()) > 10]


def split_sentences_xsum(text: str):
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    return [line for line in lines if len(line) > 10]


SPLITTERS = {"pubmed": split_sentences_pubmed, "xsum": split_sentences_xsum}


def load_v2_data(json_path: str):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    metadata = data["metadata"]
    train = data.get("train", [])
    val = data.get("val", [])
    test = data.get("test", [])
    print(f"[INFO] Loaded {json_path}")
    print(f"       Domain: {metadata['domain']} | "
          f"Generator: {metadata['generator']}")
    print(f"       Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")
    return metadata, train, val, test


def extract_sentences_and_labels(sample: dict, domain: str):
    hybrid_text = sample["hybrid_text"]
    ai_indices = set(sample["ai_indices"])
    spans = sample["sentence_spans"]
    sentences = []
    for start, end in spans:
        sent = hybrid_text[start:end].strip()
        if sent:
            if domain == "pubmed":
                sent = denormalize_pubmed_sentence(sent)
            sentences.append(sent)
    sentences = sentences[:MAX_SEQ_LEN]
    num_sents = len(sentences)
    labels = [1 if i in ai_indices else 0 for i in range(num_sents)]
    return sentences, labels


@torch.no_grad()
def extract_document_features(sentences, labels, subgroup_id: int):
    doc_len = len(sentences)
    full_text = " ".join(sentences)
    full_encoded = ref_tokenizer(
        full_text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_DOC_TOKENS,
        return_offsets_mapping=True,
    )
    input_ids = full_encoded["input_ids"].to(device)
    offset_mapping = full_encoded["offset_mapping"][0]

    char_positions = []
    cursor = 0
    for s in sentences:
        start = full_text.find(s, cursor)
        if start == -1:
            start = cursor
        end = start + len(s)
        char_positions.append((start, end))
        cursor = end

    sent_token_spans = []
    for char_start, char_end in char_positions:
        tok_start, tok_end = None, None
        for tok_idx, (os_start, os_end) in enumerate(offset_mapping):
            os_start, os_end = os_start.item(), os_end.item()
            if os_start == 0 and os_end == 0:
                continue
            if tok_start is None and os_end > char_start:
                tok_start = tok_idx
            if os_start < char_end:
                tok_end = tok_idx + 1
        if tok_start is None:
            tok_start = 0
        if tok_end is None:
            tok_end = tok_start + 1
        sent_token_spans.append((tok_start, tok_end))

    outputs = ref_model(input_ids, output_hidden_states=True)
    all_logits = outputs.logits[0].to(torch.float32)
    all_hidden = outputs.hidden_states[-1][0].to(torch.float32)
    all_input_ids = input_ids[0]

    all_probs = torch.softmax(all_logits, dim=-1)
    all_target_probs = torch.gather(
        all_probs, 1, all_input_ids.unsqueeze(1)).squeeze(1)
    all_entropy = -torch.sum(
        all_probs * torch.log(all_probs + 1e-9), dim=-1)

    hidden_states_seq = torch.zeros(
        (MAX_SEQ_LEN, HIDDEN_DIM), dtype=torch.float32)
    target_probs_seq = torch.zeros(
        (MAX_SEQ_LEN, MAX_WORD_LEN), dtype=torch.float32)
    entropies_seq = torch.zeros(
        (MAX_SEQ_LEN, MAX_WORD_LEN), dtype=torch.float32)
    diveye_feats_seq = torch.zeros(
        (MAX_SEQ_LEN, DIVEYE_DIM), dtype=torch.float32)
    pad_mask = torch.zeros(MAX_SEQ_LEN, dtype=torch.uint8)
    pad_mask[:doc_len] = 1

    for j in range(doc_len):
        tok_start, tok_end = sent_token_spans[j]
        sent_len = min(tok_end - tok_start, MAX_WORD_LEN)
        if sent_len <= 0:
            continue

        sent_hidden = all_hidden[tok_start:tok_start + sent_len]
        hidden_states_seq[j] = sent_hidden.mean(dim=0).cpu()

        sent_probs = all_target_probs[tok_start:tok_start + sent_len]
        target_probs_seq[j, :sent_len] = sent_probs.cpu()

        sent_entropy = all_entropy[tok_start:tok_start + sent_len]
        entropies_seq[j, :sent_len] = sent_entropy.cpu()

        surprisal = -torch.log2(sent_probs + 1e-9)
        diveye_feats_seq[j, 0] = surprisal.mean().cpu()
        diveye_feats_seq[j, 1] = (
            surprisal.var().cpu() if sent_len > 1 else 0.0)
        diveye_feats_seq[j, 2] = surprisal.max().cpu()
        diveye_feats_seq[j, 3] = surprisal.median().cpu()

    return {
        "hidden_states": hidden_states_seq,
        "target_probs": target_probs_seq,
        "entropies": entropies_seq,
        "diveye_feats": diveye_feats_seq,
        "pad_mask": pad_mask,
        "labels": labels,
        "subgroup_id": subgroup_id,
    }


def process_split(samples, domain: str, split_name: str):
    cached = []
    failed = 0
    for sample in tqdm(samples, desc=f"  {split_name}"):
        try:
            sentences, labels = extract_sentences_and_labels(sample, domain)
            if len(sentences) < 3:
                failed += 1
                continue
            subgroup_id = sample.get("subgroup_id", 0)
            features = extract_document_features(
                sentences, labels, subgroup_id)
            cached.append(features)
        except Exception:
            failed += 1
    if failed > 0:
        print(f"    [WARN] {split_name}: {failed} samples failed")
    return cached


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    metadata, train_samples, val_samples, test_samples = load_v2_data(
        args.input)
    domain = metadata["domain"]

    print(f"\n[INFO] Extracting features for domain={domain}")
    print(f"[INFO] MAX_SEQ_LEN={MAX_SEQ_LEN}, MAX_WORD_LEN={MAX_WORD_LEN}")

    train_cached = process_split(train_samples, domain, "train")
    val_cached = process_split(val_samples, domain, "val")
    test_cached = process_split(test_samples, domain, "test")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    save_data = {
        "metadata": metadata,
        "train": train_cached,
        "val": val_cached,
        "test": test_cached,
    }
    torch.save(save_data, args.output)

    print(f"\n[SAVE] {args.output}")
    print(f"       train={len(train_cached)}  val={len(val_cached)}  "
          f"test={len(test_cached)}")
    print(f"[DONE]")


if __name__ == "__main__":
    main()
