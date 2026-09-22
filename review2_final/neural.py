"""Word-complete windows; every native token is supervised and scored once."""
import time
from itertools import islice
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForTokenClassification
from tqdm.auto import tqdm
from .common import seed_all, write

MODEL = "microsoft/deberta-v3-small"
MODEL_REVISION = "2e588b26c0e08f967ad40f553cfff473b6bce0d6"

def tokenizer_for(path=MODEL):
    kwargs = {"revision": MODEL_REVISION} if str(path) == MODEL else {}
    tokenizer = AutoTokenizer.from_pretrained(str(path), use_fast=True, add_prefix_space=True, **kwargs)
    if not tokenizer.is_fast:
        raise RuntimeError("A fast tokenizer is required for native-token alignment")
    return tokenizer


def encode(records, tokenizer, label_names, max_length=512):
    """Every native token has exactly one supervised/scored position.

    Independent word-complete windows are reassembled before entity evaluation.
    Context at window boundaries is limited; all windows of a document stay together.
    Oversized single-token representations retain a bounded head and tail, with
    explicit truncation records. Native annotations/text are never changed.
    """
    mapping = {name: i for i, name in enumerate(label_names)}
    windows = []
    special = tokenizer.num_special_tokens_to_add(pair=False)
    probe = tokenizer("alignment probe", add_special_tokens=False)["input_ids"]
    expected_probe = [tokenizer.cls_token_id] + probe + [tokenizer.sep_token_id]
    if special != 2 or tokenizer("alignment probe", add_special_tokens=True)["input_ids"] != expected_probe:
        raise ValueError("Unsupported special-token layout")
    if max_length <= special:
        raise ValueError("max_length too short")
    for doc, row in enumerate(tqdm(records, desc="Tokenizing documents", unit="doc", mininterval=1.0)):
        encoded = tokenizer(row["tokens"], is_split_into_words=True, add_special_tokens=False,
                            truncation=False, return_attention_mask=False, verbose=False)
        pieces = [[] for _ in row["tokens"]]
        for token, word in zip(encoded["input_ids"], encoded.word_ids()):
            if word is not None:
                pieces[word].append(token)
        truncations = {}
        budget = max_length-special
        for i, ids in enumerate(pieces):
            # A whitespace-only native token can disappear in byte-level tokenization.
            # Retain it explicitly with UNK, report it, never silently drop its label.
            if not ids:
                ids.append(tokenizer.unk_token_id)
            if len(ids) > budget:
                original_length = len(ids)
                head = (budget+1)//2
                tail = budget-head
                pieces[i] = ids[:head] + (ids[-tail:] if tail else [])
                truncations[i] = {"record_id": row["record_id"], "token_index": i,
                                  "original_subwords": original_length, "retained_subwords": budget,
                                  "discarded_subwords": original_length-budget,
                                  "strategy": "oversized_native_token_head_tail"}
        start = 0
        while start < len(pieces):
            end, size = start, 0
            while end < len(pieces) and size+len(pieces[end]) <= max_length-special:
                size += len(pieces[end])
                end += 1
            ids, targets, word_indices = [], [], []
            for word in range(start, end):
                ids.extend(pieces[word])
                targets.extend([mapping[row["labels"][word]]] + [-100]*(len(pieces[word])-1))
                word_indices.extend([word] + [-1]*(len(pieces[word])-1))
            full = [tokenizer.cls_token_id] + ids + [tokenizer.sep_token_id]
            windows.append({"input_ids": full, "labels": [-100]+targets+[-100],
                            "words": [-1]+word_indices+[-1], "doc": doc,
                            "token_truncations": [truncations[w] for w in range(start, end) if w in truncations],
                            "unknown_native_tokens": sum(p == [tokenizer.unk_token_id] for p in pieces[start:end])})
            start = end
    return windows


class Collate:
    def __init__(self, pad):
        self.pad = pad

    def __call__(self, rows):
        width = max(len(r["input_ids"]) for r in rows)
        ids, masks, labels = [], [], []
        for row in rows:
            n = len(row["input_ids"])
            ids.append(row["input_ids"] + [self.pad]*(width-n))
            masks.append([1]*n+[0]*(width-n))
            labels.append(row["labels"]+[-100]*(width-n))
        return {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(masks),
                "labels": torch.tensor(labels), "rows": rows}


def loader(windows, tokenizer, batch_size, shuffle=False, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(windows, batch_size=batch_size, shuffle=shuffle, collate_fn=Collate(tokenizer.pad_token_id),
                      generator=generator, num_workers=0)


def device_for(requested="auto"):
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device(requested)


def load_model(labels, checkpoint=None, device="cpu"):
    kwargs = {} if checkpoint else {"revision": MODEL_REVISION, "num_labels": len(labels), "id2label": dict(enumerate(labels)),
                                    "label2id": {v: i for i, v in enumerate(labels)}}
    model = AutoModelForTokenClassification.from_pretrained(str(checkpoint or MODEL), **kwargs)
    if model.config.num_labels != len(labels) or [model.config.id2label[i] for i in range(len(labels))] != labels:
        raise ValueError("Checkpoint label map differs from this dataset")
    return model.to(device)


def inputs(batch, device):
    return {k: batch[k].to(device) for k in ("input_ids", "attention_mask", "labels")}


def fgsm_embeddings(model, x, epsilon):
    if epsilon <= 0:
        raise ValueError("FGSM epsilon must be positive")
    embedding = model.get_input_embeddings()(x["input_ids"])
    output = model(inputs_embeds=embedding, attention_mask=x["attention_mask"], labels=x["labels"])
    grad = torch.autograd.grad(output.loss, embedding)[0]
    # Only content positions, not padding/CLS/SEP. Include continuation subwords.
    mask = x["attention_mask"].clone()
    mask[:, 0] = 0
    mask[torch.arange(len(mask), device=mask.device), x["attention_mask"].sum(1)-1] = 0
    return embedding.detach() + epsilon*grad.detach().sign()*mask.unsqueeze(-1)


def train_epoch(model, batches, optimizer, device, epsilon=0., max_steps=None,
                description="Training", log_every=50, budget_check=None):
    model.train()
    started, total, count = time.perf_counter(), 0., 0
    steps = min(len(batches), max_steps) if max_steps else len(batches)
    print(f"{description}: {steps} batches on {device}, FGSM epsilon={epsilon:g}", flush=True)
    progress = tqdm(islice(batches, steps), total=steps, desc=description, unit="batch", mininterval=1.0)
    for batch in progress:
        if budget_check:
            budget_check()
        x = inputs(batch, device)
        optimizer.zero_grad(set_to_none=True)
        if epsilon:
            # Deterministic attack direction; train clean and adversarial losses with dropout.
            model.eval()
            adv = fgsm_embeddings(model, x, epsilon)
            model.train()
            # Reconnect adversarial embedding to the encoder's embedding weights.
            base = model.get_input_embeddings()(x["input_ids"])
            perturbed = base + (adv-base.detach()).detach()
            clean = model(**x).loss
            attack = model(inputs_embeds=perturbed, attention_mask=x["attention_mask"], labels=x["labels"]).loss
            loss = .5*(clean+attack)
        else:
            loss = model(**x).loss
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        total += float(loss.detach())
        count += 1
        elapsed = time.perf_counter()-started
        progress.set_postfix(loss=f"{float(loss.detach()):.4f}", mean=f"{total/count:.4f}", refresh=False)
        if count == 1 or (log_every and count % log_every == 0) or count == steps:
            eta = elapsed/count*(steps-count)
            tqdm.write(f"{description} | batch {count}/{steps} | loss={float(loss.detach()):.4f} "
                       f"mean_loss={total/count:.4f} | elapsed={elapsed:.1f}s ETA={eta:.1f}s")
    return {"loss": total/max(count, 1), "steps": count, "seconds": time.perf_counter()-started}


def dropout_on(model):
    model.eval()
    count = 0
    for module in model.modules():
        if isinstance(module, torch.nn.modules.dropout._DropoutNd) or module.__class__.__name__ == "StableDropout":
            module.train()
            count += 1
    if not count:
        raise RuntimeError("No dropout modules activated")
    return count


def predict_logits(model, batches, records, nlabels, device, epsilon=0., stochastic=False,
                   description=None, budget_check=None):
    if stochastic:
        dropout_on(model)
    else:
        model.eval()
    output = [np.zeros((len(r["tokens"]), nlabels), dtype=np.float32) for r in records]
    seen = [np.zeros(len(r["tokens"]), dtype=bool) for r in records]
    description = description or ("MC inference" if stochastic else f"FGSM evaluation eps={epsilon:g}" if epsilon else "Clean evaluation")
    for batch in tqdm(batches, desc=description, unit="batch", mininterval=1.0, leave=not stochastic):
        if budget_check:
            budget_check()
        x = inputs(batch, device)
        if epsilon:
            with torch.enable_grad():
                adv = fgsm_embeddings(model, x, epsilon)
            with torch.no_grad():
                logits = model(inputs_embeds=adv, attention_mask=x["attention_mask"]).logits
        else:
            with torch.no_grad():
                logits = model(input_ids=x["input_ids"], attention_mask=x["attention_mask"]).logits
        logits = logits.detach().float().cpu().numpy()
        for b, row in enumerate(batch["rows"]):
            doc = row["doc"]
            for i, word in enumerate(row["words"]):
                if word >= 0:
                    if seen[doc][word]:
                        raise AssertionError("A native token was scored twice")
                    output[doc][word] = logits[b, i]
                    seen[doc][word] = True
    if not all(x.all() for x in seen):
        raise AssertionError("Native token predictions are missing")
    return output


def decode(logits, labels):
    return [[labels[i] for i in row.argmax(-1)] for row in logits]


def save_checkpoint(model, tokenizer, directory):
    print(f"Saving checkpoint: {directory}", flush=True)
    Path(directory).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(directory)
    tokenizer.save_pretrained(directory)
