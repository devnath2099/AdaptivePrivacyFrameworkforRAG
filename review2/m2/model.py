"""Standard transformer token classifiers; tiny random variants are testing only."""
from transformers import AutoModelForTokenClassification, AutoTokenizer


def build(spec, labels, tokenizer=None):
    if spec.get('tiny'):
        from transformers import DebertaConfig, DebertaForTokenClassification, DistilBertConfig, DistilBertForTokenClassification
        args = dict(vocab_size=len(tokenizer), num_labels=len(labels), pad_token_id=tokenizer.pad_token_id)
        if spec['family'] == 'deberta':
            config = DebertaConfig(**args, hidden_size=32, num_hidden_layers=1, num_attention_heads=2,
                                   intermediate_size=64, max_position_embeddings=512)
            model = DebertaForTokenClassification(config)
        else:
            config = DistilBertConfig(**args, dim=32, n_layers=1, n_heads=2, hidden_dim=64, max_position_embeddings=512)
            model = DistilBertForTokenClassification(config)
    else:
        model = AutoModelForTokenClassification.from_pretrained(spec['name'], revision=spec['revision'],
                                                                num_labels=len(labels))
    model.config.id2label = dict(enumerate(labels))
    model.config.label2id = {v: k for k, v in enumerate(labels)}
    return model


def get_tokenizer(spec):
    tokenizer = AutoTokenizer.from_pretrained(spec.get('tokenizer', spec['name']),
                                              revision=spec.get('tokenizer_revision', spec['revision']), use_fast=True)
    if not tokenizer.is_fast:
        raise ValueError('Offset-aware fast tokenizer required')
    return tokenizer
