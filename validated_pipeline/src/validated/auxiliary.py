"""Optional isolated evidence evaluation on provided synthetic structured spans."""
import re
import yaml
from .common import path,metadata,write_json
from .evidence import extract,extractor
from .corpus import normalize


def score_row(row, spec, nlp=None):
    text = row['source_text']
    spans = row['privacy_mask']
    if not isinstance(spans,list):
        raise ValueError('Expected privacy_mask list of provided spans')
    gold, unsupported = set(), 0
    for span in spans:
        start,end=int(span['start']),int(span['end'])
        if text[start:end] != span['value']:
            raise ValueError('Provided PII span does not match source text')
        kind=spec['supported_types'].get(span['label'])
        if kind:
            gold.add((kind,start,end))
        else:
            unsupported += 1
    # Preserve original offsets for this span benchmark (normalization would shift them).
    ev=extract({'query_normalized_text':text},nlp)
    pred=set()
    for kind in set(spec['supported_types'].values()):
        for hit in ev['patterns'][kind]:
            start,end=hit['start'],hit['end']
            # Identifier/credential evidence includes its cue; score the value-only portion.
            if kind in ['credential','identifier']:
                pattern=(r'^(?:password|api[ _-]?key|secret|otp|pin)\s*(?:is|:|=)\s*' if kind=='credential' else
                         r'^(?:ssn|patient id|account number|account no|iban|mrn)\s*[:=#-]?\s*')
                prefix=re.match(pattern,hit['text'],re.I)
                if prefix:
                    start+=prefix.end()
            pred.add((kind,start,end))
    return dict(tp=len(gold&pred),fp=len(pred-gold),fn=len(gold-pred),unsupported_gold_spans=unsupported)


def run_auxiliary(cfg,args):
    from datasets import load_dataset
    spec=yaml.safe_load(path('configs/auxiliary_pii.yaml').read_text())
    directory=path(args.output or 'outputs/m1_auxiliary_pii_benchmark')
    nlp=extractor(cfg['evidence'])
    ds=load_dataset(spec['dataset_id'],revision=spec['revision'],split=spec['split'],streaming=True,cache_dir=str(path('cache/auxiliary')))
    counts=dict(tp=0,fp=0,fn=0,unsupported_gold_spans=0)
    n=0
    for row in ds:
        if row['language'] != spec['language']:
            continue
        score=score_row(row,spec,nlp)
        for k,v in score.items():
            counts[k]+=v
        n+=1
        if n>=int(spec['max_records']):
            break
    if not n:
        raise ValueError('No auxiliary evidence records loaded; no fallback')
    precision=counts['tp']/(counts['tp']+counts['fp']) if counts['tp']+counts['fp'] else None
    recall=counts['tp']/(counts['tp']+counts['fn']) if counts['tp']+counts['fn'] else None
    result=dict(provenance=metadata(cfg,spec,n,'auxiliary_train',None),evaluation_name=spec['evaluation_name'],
                counts=counts,precision=precision,recall=recall,matching='exact typed value spans; configured structured types only',
                excluded_from_natural_M2_M5=True)
    write_json(directory/'results.json',result)
    print(spec['evaluation_name'],n,'records')
