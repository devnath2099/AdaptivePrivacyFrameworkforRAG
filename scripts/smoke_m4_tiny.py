"""Quick smoke test for M4 adversarial training with tiny dataset.

Verifies:
- model.requires_grad_(True) works after loading checkpoint
- compute_clean_loss_and_grad computes gradients successfully
- trainer.train() works end-to-end with the torch.no_grad() fix
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from m3_privacy_prediction.model import DeBERTaMultiTaskModel
from m3_privacy_prediction.losses import compute_total_loss
from m4_adversarial_training.fgsm import adversarial_forward
from m4_adversarial_training.trainer import M4Trainer

print("=" * 80)
print("M4 TINY SMOKE TEST")
print("=" * 80)

# 1. Load model from checkpoint
print("\n[1] Loading M3 checkpoint...")
checkpoint = torch.load("outputs/m3/smoke/best_model.pt", map_location="cpu")
model = DeBERTaMultiTaskModel(model_name="microsoft/deberta-base", dropout_rate=0.1)
model.load_state_dict(checkpoint)
model.requires_grad_(True)
print(f"  Model parameters require grad: {next(model.parameters()).requires_grad}")

# 2. Create tiny random dataset
print("\n[2] Creating tiny random dataset...")
class TinyDataset(torch.utils.data.Dataset):
    def __init__(self, n=50, seq_len=128):
        self.n = n
        self.seq_len = seq_len

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        labels = {
            "sensitivity": torch.randint(0, 3, ()),
            "intent": torch.randint(0, 5, ()),
            "disclosure_scope": torch.randint(0, 2, ()),
            "entity_tags": torch.randint(0, 4, ()),
            "threat_content": torch.randint(0, 3, ()),
        }
        soft_targets = {}
        for dim_name, num_labels in {
            "sensitivity": 3, "intent": 5, "disclosure_scope": 2,
            "entity_tags": 4, "threat_content": 3
        }.items():
            one_hot = torch.zeros(num_labels)
            one_hot[labels[dim_name]] = 1.0
            soft_targets[dim_name] = one_hot
        return {
            "input_ids": torch.randint(0, 32000, (self.seq_len,)),
            "attention_mask": torch.ones(self.seq_len, dtype=torch.long),
            "soft_targets": soft_targets,
        }

ds_train = TinyDataset(n=50)
ds_val = TinyDataset(n=10)
print(f"  Train: {len(ds_train)}, Val: {len(ds_val)}")

# 3. Create trainer
print("\n[3] Setting up trainer...")
device = torch.device("cpu")
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)
trainer = M4Trainer(model=model, train_dataset=ds_train, val_dataset=ds_val,
                    optimizer=optimizer, device=device)

# 4. Test compute_clean_loss_and_grad directly
print("\n[4] Testing compute_clean_loss_and_grad...")
from m4_adversarial_training.fgsm import compute_clean_loss_and_grad
batch = ds_train[0]
input_ids = batch["input_ids"].unsqueeze(0)
attention_mask = batch["attention_mask"].unsqueeze(0)
targets = {k: v.unsqueeze(0) for k, v in batch["soft_targets"].items()}

embeddings, clean_loss, grad_wrt_embeddings, task_losses = compute_clean_loss_and_grad(
    model, input_ids, attention_mask, targets, compute_total_loss
)
assert embeddings.requires_grad, "Embeddings should require grad!"
assert grad_wrt_embeddings is not None, "Gradient should not be None!"
print(f"  clean_loss={clean_loss.item():.4f}, grad shape={grad_wrt_embeddings.shape}")
print("  PASSED: gradients computed successfully")

# 5. Test adversarial_forward
print("\n[5] Testing adversarial_forward...")
total_loss, clean_loss_val, adv_loss_val, task_losses, delta, max_delta = adversarial_forward(
    model, input_ids, attention_mask, targets, compute_total_loss, epsilon=1e-3
)
assert total_loss.requires_grad or total_loss.grad_fn is not None
print(f"  total_loss={total_loss.item():.4f}, delta={delta.shape}")
print("  PASSED: adversarial forward works")

# 6. Test trainer.train with tiny dataset, 1 epoch
print("\n[6] Testing trainer.train (1 epoch, tiny dataset)...")
try:
    result = trainer.train(epochs=1, batch_size=8, learning_rate=1e-5, epsilon=1e-3, lambda_adv=1.0)
    print("  PASSED: trainer.train completed successfully!")
    print(f"  Result keys: {list(result.keys())}")
except Exception as e:
    print(f"  FAILED: {e}")
    raise

print("\n" + "=" * 80)
print("ALL M4 TINY SMOKE TESTS PASSED")
print("=" * 80)
