import time
import torch


def coverage(rows, tokenizer, lengths, model=None):
    report = []
    for length in lengths:
        start = time.perf_counter()
        truncated = affected = total_spans = 0
        for row in rows:
            full = tokenizer(row['text'], truncation=False)['input_ids']
            offsets = tokenizer(row['text'], truncation=True, max_length=length, return_offsets_mapping=True)['offset_mapping']
            end = max(b for a, b in offsets)
            truncated += len(full) > length
            affected += sum(s['end'] > end for s in row['canonical_spans'])
            total_spans += len(row['canonical_spans'])
        item = {'length': length, 'truncated_record_fraction': truncated / max(1, len(rows)),
                'affected_gold_span_fraction': affected / max(1, total_spans),
                'tokenization_seconds': time.perf_counter() - start, 'split': 'train'}
        if model is not None:
            device = next(model.parameters()).device
            ids = torch.full((1, length), tokenizer.unk_token_id, dtype=torch.long, device=device)
            model.eval()
            if device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.no_grad():
                model(input_ids=ids, attention_mask=torch.ones_like(ids))
            if device.type == 'cuda':
                torch.cuda.synchronize()
            item.update(forward_seconds=time.perf_counter() - start,
                        peak_cuda_bytes=torch.cuda.max_memory_allocated() if device.type == 'cuda' else None,
                        attention_cells_per_head=length * length,
                        memory_note='CPU allocator peak unavailable; attention-cell count is analytical, not measured memory')
        report.append(item)
    return report
