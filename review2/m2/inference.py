import time
import torch
from .alignment import collate, decode
from .metrics import entity_metrics


def predict(model, examples, tokenizer, labels, batch_size=8, temperature=1.):
    model.eval()
    device = next(model.parameters()).device
    predictions, logits = [], []
    if device.type == 'cuda':
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        for offset in range(0, len(examples), batch_size):
            part = examples[offset:offset + batch_size]
            batch = collate(part, tokenizer.pad_token_id)
            output = model(input_ids=batch['input_ids'].to(device), attention_mask=batch['attention_mask'].to(device)).logits.cpu()
            for example, z in zip(part, output):
                z = z[:len(example['input_ids'])]
                p = (z / temperature).softmax(-1)
                predictions.append(decode(p.argmax(-1), example['offsets'], labels, p))
                logits.append(z)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    return predictions, logits, time.perf_counter() - start


def evaluate(model, examples, tokenizer, labels, batch_size=8, train_counts=None):
    pred, logits, elapsed = predict(model, examples, tokenizer, labels, batch_size)
    metrics = entity_metrics([e['record'] for e in examples], pred, train_counts)
    metrics.update(seconds=elapsed, milliseconds_per_record=elapsed * 1000 / max(1, len(examples)))
    return metrics, pred, logits
