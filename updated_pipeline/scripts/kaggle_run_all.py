# Kaggle One-Cell Pipeline Runner
# Paste this entire cell into a Kaggle notebook and run

import os, sys, subprocess, json, time
from pathlib import Path

# ===== CONFIG =====
REPO_URL = "https://github.com/YOUR_USERNAME/review1_privacy_pipeline.git"  # UPDATE THIS
REPO_DIR = "review1_privacy_pipeline"
TARGET_DIR = "review1/updated_pipeline"  # or "review2" if that's your new folder

# ===== SETUP =====
print("=" * 60)
print("KAGGLE M1→M6 PIPELINE RUNNER")
print("=" * 60)

# Clone if needed
if not Path(REPO_DIR).exists():
    print("📥 Cloning repository...")
    subprocess.run(["git", "clone", REPO_URL], check=True)

os.chdir(REPO_DIR / TARGET_DIR)
PROJECT_ROOT = Path.cwd()
os.environ["PROJECT_ROOT"] = str(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Create outputs dir
(PROJECT_ROOT / "outputs").mkdir(parents=True, exist_ok=True)
print(f"📁 Project root: {PROJECT_ROOT}")

# ===== STAGE RUNNER =====
STAGES = [
    ("M1", "scripts/run_m1_partition.py", "outputs/m1_unified_dataset.jsonl", 600),
    ("M2", "scripts/run_m2_only.py", "outputs/m2_weak_labels", 1800),
    ("M3", "scripts/run_m3.py", "outputs/m3/best_model.pt", 3600),
    ("M4", "scripts/run_m4.py", "outputs/m4/best_model.pt", 10800),
    ("M5", "scripts/run_m5.py", "outputs/m5/full_m4/uncertainty_predictions.jsonl", 3600),
    ("M6", "scripts/run_m6.py", "outputs/m6/decision_results.jsonl", 300),
]

def checkpoint_exists(path):
    p = Path(path)
    if p.is_dir(): return any(p.iterdir())
    return p.exists() and p.stat().st_size > (1_000_000 if p.suffix == ".pt" else 0)

# Check what's already done
completed = []
for name, script, ckpt, _ in STAGES:
    if checkpoint_exists(ckpt):
        completed.append(name)
        print(f"✅ {name} already done")
    else:
        print(f"⏳ {name} pending")

# Run remaining stages
for name, script, ckpt, timeout in STAGES:
    if name in completed:
        continue
    
    print(f"\n{'='*60}")
    print(f"🚀 Running {name}...")
    print(f"{'='*60}")
    
    start = time.time()
    env = {**os.environ, "PROJECT_ROOT": str(PROJECT_ROOT)}
    
    try:
        result = subprocess.run(
            [sys.executable, script],
            env=env,
            cwd=str(PROJECT_ROOT),
            timeout=timeout,
            capture_output=True,
            text=True,
        )
        
        elapsed = time.time() - start
        if result.returncode == 0 and checkpoint_exists(ckpt):
            print(f"✅ {name} done in {elapsed/60:.1f} min")
            completed.append(name)
        else:
            print(f"❌ {name} FAILED (exit={result.returncode})")
            print(f"STDOUT:\n{result.stdout[-3000:]}")
            print(f"STDERR:\n{result.stderr[-3000:]}")
            break
            
    except subprocess.TimeoutExpired:
        print(f"⏱️ {name} TIMED OUT after {timeout/60:.0f} min")
        break
    except Exception as e:
        print(f"💥 {name} CRASHED: {e}")
        break

# ===== FINAL SUMMARY =====
print(f"\n{'='*60}")
print("PIPELINE SUMMARY")
print(f"{'='*60}")
for name, _, ckpt, _ in STAGES:
    status = "✅" if name in completed else "❌"
    print(f"  {status} {name}")

# Show key outputs
print(f"\n📊 KEY OUTPUTS:")
for name, _, ckpt, _ in STAGES:
    if name in completed:
        p = Path(ckpt)
        if p.exists():
            size = p.stat().st_size / (1024**2) if p.is_file() else "dir"
            print(f"  {name}: {ckpt} ({size} MB)" if isinstance(size, float) else f"  {name}: {ckpt}")

print(f"\n🎯 Done! Check outputs/ for results")