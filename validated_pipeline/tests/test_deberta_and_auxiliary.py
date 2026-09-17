import torch
import yaml
from validated.common import path
from validated.model import PrivacyModel
from validated.adversarial import perturb,adversarial_loss
from validated.calibration import enable_mc_dropout
from validated.auxiliary import score_row


def test_actual_deberta_interface_without_pretrained_weights(cfg,batch):
    from transformers import DebertaConfig,DebertaModel
    enc=DebertaModel(DebertaConfig(hidden_size=24,num_hidden_layers=1,num_attention_heads=4,intermediate_size=48,vocab_size=64))
    model=PrivacyModel(enc,cfg['active_tasks'],cfg['model'])
    adversarial_loss(model,batch,cfg).backward()
    assert all(head.weight.grad is not None for head in model.heads.values())
    with torch.no_grad():
        adv,_=perturb(model,batch,cfg)
    assert adv.shape == (4,8,24)
    enable_mc_dropout(model)
    assert model.dropout.training
    assert any(m.training for m in enc.modules() if m.__class__.__name__ in ['Dropout','StableDropout'])


def test_auxiliary_span_contract():
    spec=yaml.safe_load(path('configs/auxiliary_pii.yaml').read_text())
    text='Email a@example.org'
    row={'source_text':text,'privacy_mask':[dict(start=6,end=len(text),value='a@example.org',label='EMAIL')]}
    assert score_row(row,spec)==dict(tp=1,fp=0,fn=0,unsupported_gold_spans=0)


def test_dynamic_taxonomy_and_explicit_unfreeze(cfg,batch):
    from transformers import DebertaConfig,DebertaModel
    from validated.model import loss
    enc=DebertaModel(DebertaConfig(hidden_size=24,num_hidden_layers=2,num_attention_heads=4,intermediate_size=48,vocab_size=64))
    cfg['model']['unfreeze_last_n_layers']=1
    tasks={'custom_category':{'kind':'categorical','classes':['a','b','c','d']},
           'custom_binary':{'kind':'multilabel','labels':['x','y']}}
    model=PrivacyModel(enc,tasks,cfg['model'])
    assert not any(p.requires_grad for p in enc.encoder.layer[0].parameters())
    assert all(p.requires_grad for p in enc.encoder.layer[1].parameters())
    output=model(input_ids=batch['input_ids'],attention_mask=batch['attention_mask'])
    assert {k:tuple(v.shape) for k,v in output.items()}=={'custom_category':(4,4),'custom_binary':(4,2)}
    targets={'custom_category':torch.full((4,4),.25),'custom_binary':torch.full((4,2),.5)}
    observed={'custom_category':torch.ones(4),'custom_binary':torch.ones(4,2)}
    loss(output,targets,observed,tasks,{'custom_category':1.,'custom_binary':.5}).backward()
    assert model.heads['custom_binary'].weight.grad is not None
