import copy
import torch
from torch.utils.data import Dataset
from validated.model import checkpoint_contract
from validated.trainer import train, load_checkpoint
from validated.uncertainty import run_m5
from validated.common import read_json


class FixtureDataset(Dataset):
    def __init__(self, batch):
        self.batch = batch
    def __len__(self):
        return len(self.batch['input_ids'])
    def __getitem__(self, i):
        return {k: ({t: x[i] for t,x in v.items()} if isinstance(v,dict) else v[i]) for k,v in self.batch.items()}


def test_m3_m4_m5_offline_smoke_and_resume(tmp_path, cfg, model, batch):
    data = FixtureDataset(batch)
    c3 = checkpoint_contract(cfg,'fixture_split','fixture_source',None,'m3')
    parent = train(model,data,data,cfg,torch.device('cpu'),tmp_path/'m3',c3,{'fixture':True})
    assert len(parent) == 64
    before = copy.deepcopy(model.state_dict())
    assert train(model,data,data,cfg,torch.device('cpu'),tmp_path/'m3',c3,{'fixture':True}) == parent
    assert all(torch.equal(before[k],v) for k,v in model.state_dict().items())
    c4 = checkpoint_contract(cfg,'fixture_split','fixture_source',parent,'m4')
    h4 = train(model,data,data,cfg,torch.device('cpu'),tmp_path/'m4',c4,{'fixture':True})
    assert c4['parent_checkpoint'] == parent
    assert load_checkpoint(tmp_path/'m4'/'best.pt',c4)['contract']['stage'] == 'm4'
    files = run_m5(model,{'calibration':data,'validation':data},
                   {'calibration':[{'record_id':str(i)} for i in range(4)], 'validation':[{'record_id':str(i)} for i in range(4)]},
                   cfg,torch.device('cpu'),tmp_path/'m5',{'fixture':True,'parent':h4})
    assert all(p.exists() for p in files)
    report = read_json(tmp_path/'m5'/'calibration_metrics.json')
    assert set(report['temperatures']) == set(cfg['active_tasks'])
    assert all(x['temperature'] > 0 for x in report['temperatures'].values())


def test_mid_epoch_resume_exactly_replays(tmp_path, cfg, model, batch, monkeypatch):
    import validated.trainer as trainer
    cfg['training'].update(batch_size=2,m3_epochs=2,checkpoint_every_batches=1)
    data = FixtureDataset(batch)
    initial = copy.deepcopy(model.state_dict())
    contract = checkpoint_contract(cfg,'fixture','fixture',None,'m3')
    torch.manual_seed(901)
    train(model,data,data,cfg,torch.device('cpu'),tmp_path/'full',contract,{'fixture':True})
    expected = copy.deepcopy(model.state_dict())
    model.load_state_dict(initial)
    torch.manual_seed(901)
    save = trainer.atomic_save
    triggered = []
    def interrupt(value,p):
        save(value,p)
        if value['epoch'] == 0 and value['batch_offset'] == 1 and not triggered:
            triggered.append(True)
            raise KeyboardInterrupt('simulated interruption after durable checkpoint')
    monkeypatch.setattr(trainer,'atomic_save',interrupt)
    import pytest
    with pytest.raises(KeyboardInterrupt):
        train(model,data,data,cfg,torch.device('cpu'),tmp_path/'resumed',contract,{'fixture':True})
    monkeypatch.setattr(trainer,'atomic_save',save)
    train(model,data,data,cfg,torch.device('cpu'),tmp_path/'resumed',contract,{'fixture':True})
    assert all(torch.equal(expected[k],v) for k,v in model.state_dict().items())
