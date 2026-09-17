from argparse import Namespace
import json
import pytest
import yaml
from validated.corpus import record
from validated import runner,corpus


def test_stage_receipts_reject_corruption_and_changed_config(tmp_path,cfg,monkeypatch):
    cfg['source_manifest']='sources.yaml'
    cfg['profiles']['overnight_50k']={'a':40,'b':40}
    (tmp_path/'pipeline.yaml').write_text(yaml.safe_dump(cfg))
    (tmp_path/'sources.yaml').write_text(yaml.safe_dump({'version':1,'seed':42,'allow_synthetic_fallback':False,'sources':{}}))
    monkeypatch.setattr(runner,'path',lambda p:tmp_path/p)
    records=[record(f'{s} query {i}','',s,s,i,i,'fixture') for s in ['a','b'] for i in range(50)]
    calls=[]
    def source_loader(*args):
        calls.append(True)
        return records,{}
    monkeypatch.setattr(corpus,'load_sources',source_loader)
    args=Namespace(config='pipeline.yaml',profile='overnight_50k',output='run',until='data',mc_diagnostic=False)
    runner.run(args)
    runner.run(args)
    assert len(calls)==1
    train=tmp_path/'run'/'corpus'/'train.jsonl'
    original=train.read_bytes()
    train.write_bytes(original+b'\n')
    with pytest.raises(ValueError,match='Corrupt stage artifact'):
        runner.run(args)
    train.write_bytes(original)
    cfg['seed']=99
    (tmp_path/'pipeline.yaml').write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError,match='Incompatible'):
        runner.run(args)
