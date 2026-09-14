import sys; sys.path.insert(0,'src')
from transformers import DebertaModel
import torch
m = DebertaModel.from_pretrained('microsoft/deberta-base')
x = torch.randint(0,1000,(1,128))
a = torch.ones(1,128)
o = m(x, attention_mask=a)
print('Output type:', type(o))
print('Attributes:', [x for x in dir(o) if not x.startswith('_')])
print()
# Try last_hidden_state
if hasattr(o, 'last_hidden_state'):
    print('last_hidden_state shape:', o.last_hidden_state.shape)
# Try pooler_output
if hasattr(o, 'pooler_output'):
    print('pooler_output shape:', o.pooler_output.shape)
else:
    print('No pooler_output - using last_hidden_state[:, 0]')
    print('cls token shape:', o.last_hidden_state[:, 0].shape)
