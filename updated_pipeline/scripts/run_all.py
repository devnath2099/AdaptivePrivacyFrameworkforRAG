#!/usr/bin/env python3
"""
Unified M1→M6 Pipeline Runner with Checkpoint/Resume
Runs: M1 → M2 → M3 → M4 → M5 → M6
Can be interrupted and resumed from last completed stage.
"""

from __future__ import annotations

import json
import sys
import os
import time
import subprocess
import signal
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", str(Path(__file__).resolve().parent.parent)))
SRC_DIR = PROJECT_ROOT / "src"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Stage definitions with their checkpoint files
STAGES = [
    {
        "name": "M1",
        "script": "run_m1_partition.py",
        "checkpoint": "outputs/m1_unified_dataset.jsonl",
        "description": "Data integration & normalization",
        "estimated_time_min": 10,
    },
    {
        "name": "M2",
        "script": "run_m2_only.py",
        "checkpoint": "outputs/m2_weak_labels",
        "description": "Weak label generation (Snorkel)",
        "estimated_time_min": 30,
    },
    {
        "name": "M3",
        "script": "run_m3.py",
        "checkpoint": "outputs/m3/best_model.pt",
        "description": "Baseline DeBERTa multi-task training",
        "estimated_time_min": 60,
    },
    {
        "name": "M4",
        "script": "run_m4.py",
        "checkpoint": "outputs/m4/best_model.pt",
        "description": "Adversarial FGSM fine-tuning",
        "estimated_time_min": 180,
    },
    {
        "name": "M5",
        "script": "run_m5.py",
        "checkpoint": "outputs/m5/full_m4/uncertainty_predictions.jsonl",
        "description": "MC-Dropout + temperature scaling",
        "estimated_time_min": 60,
    },
    {
        "name": "M6",
        "script": "run_m6.py",
        "checkpoint": "outputs/m6/decision_results.jsonl",
        "description": "Risk-to-policy decision engine",
        "estimated_time_min": 5,
    },
]

STATE_FILE = OUTPUTS_DIR / "pipeline_state.json"
LOG_FILE = OUTPUTS_DIR / "pipeline.log"

class PipelineRunner:
    def __init__(self):
        self.start_time = time.time()
        self.state = self._load_state()
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        self._setup_signal_handlers()
    
    def _setup_signal_handlers(self):
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        print(f"\n⚠️  Received signal {signum}, saving state...")
        self._save_state()
        sys.exit(130)
    
    def _load_state(self):
        if STATE_FILE.exists():
            with open(STATE_FILE) as f:
                return json.load(f)
        return {"completed": [], "current": None, "start_time": None}
    
    def _save_state(self):
        self.state["last_update"] = datetime.now().isoformat()
        self.state["elapsed_seconds"] = time.time() - self.start_time
        with open(STATE_FILE, "w") as f:
            json.dump(self.state, f, indent=2)
    
    def _log(self, msg):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{timestamp}] {msg}"
        print(line)
        with open(LOG_FILE, "a") as f:
            f.write(line + "\n")
    
    def _check_checkpoint(self, stage):
        """Verify checkpoint exists and is valid"""
        ckpt = PROJECT_ROOT / stage["checkpoint"]
        if ckpt.is_dir():
            # Directory checkpoint - check for files
            return any(ckpt.iterdir())
        elif ckpt.suffix == ".pt":
            # Model checkpoint - check size > 1MB
            return ckpt.exists() and ckpt.stat().st_size > 1_000_000
        else:
            return ckpt.exists() and ckpt.stat().st_size > 0
    
    def run_stage(self, stage):
        """Run a single pipeline stage"""
        name = stage["name"]
        script = stage["script"]
        
        self._log(f"🚀 Starting {name}: {stage['description']}")
        self.state["current"] = name
        self._save_state()
        
        script_path = SCRIPTS_DIR / script
        if not script_path.exists():
            self._log(f"❌ Script not found: {script_path}")
            return False
        
        env = {**os.environ, "PROJECT_ROOT": str(PROJECT_ROOT)}
        
        try:
            result = subprocess.run(
                [sys.executable, str(script_path)],
                env=env,
                cwd=str(PROJECT_ROOT),
                timeout=stage["estimated_time_min"] * 60 * 2,  # 2x buffer
                capture_output=True,
                text=True,
            )
            
            if result.returncode == 0:
                if self._check_checkpoint(stage):
                    self._log(f"✅ {name} completed successfully")
                    if name not in self.state["completed"]:
                        self.state["completed"].append(name)
                    return True
                else:
                    self._log(f"⚠️ {name} ran but checkpoint missing!")
                    return False
            else:
                self._log(f"❌ {name} failed with code {result.returncode}")
                self._log(f"STDOUT: {result.stdout[-2000:]}")
                self._log(f"STDERR: {result.stderr[-2000:]}")
                return False
                
        except subprocess.TimeoutExpired:
            self._log(f"⏱️ {name} timed out after {stage['estimated_time_min']*2} min")
            return False
        except Exception as e:
            self._log(f"💥 {name} crashed: {e}")
            return False
    
    def run(self, start_from=None):
        """Run full pipeline from start or resume point"""
        self._log("=" * 60)
        self._log("M1→M6 UNIFIED PIPELINE STARTED")
        self._log("=" * 60)
        
        if self.state.get("start_time") is None:
            self.state["start_time"] = self.start_time
        
        # Determine start index
        start_idx = 0
        if start_from:
            start_idx = next((i for i, s in enumerate(STAGES) if s["name"] == start_from), 0)
        elif self.state["completed"]:
            # Resume from next uncompleted
            last_completed = self.state["completed"][-1]
            start_idx = next((i for i, s in enumerate(STAGES) if s["name"] == last_completed), 0) + 1
            self._log(f"📋 Resuming from {STAGES[start_idx]['name']} (last completed: {last_completed})")
        
        # Run stages
        for i in range(start_idx, len(STAGES)):
            stage = STAGES[i]
            
            if stage["name"] in self.state["completed"]:
                self._log(f"⏭️  Skipping {stage['name']} (already completed)")
                continue
            
            success = self.run_stage(stage)
            if not success:
                self._log(f"⛔ Pipeline stopped at {stage['name']}")
                self._save_state()
                return False
        
        self._log("=" * 60)
        self._log("🎉 ALL STAGES COMPLETED SUCCESSFULLY!")
        self._log(f"Total time: {(time.time() - self.start_time)/60:.1f} minutes")
        self._log("=" * 60)
        self.state["completed"] = [s["name"] for s in STAGES]
        self._save_state()
        return True


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Unified M1→M6 Pipeline Runner")
    parser.add_argument("--start-from", choices=[s["name"] for s in STAGES],
                        help="Start from specific stage (default: auto-resume)")
    parser.add_argument("--only", choices=[s["name"] for s in STAGES],
                        help="Run only this stage")
    parser.add_argument("--list", action="store_true", help="List stages and exit")
    args = parser.parse_args()
    
    if args.list:
        print("Pipeline stages:")
        for s in STAGES:
            ckpt_status = "✅" if (PROJECT_ROOT / s["checkpoint"]).exists() else "⬜"
            print(f"  {s['name']}: {s['description']} ({s['estimated_time_min']} min) {ckpt_status}")
        return
    
    runner = PipelineRunner()
    
    if args.only:
        # Find and run single stage
        stage = next(s for s in STAGES if s["name"] == args.only)
        success = runner.run_stage(stage)
        sys.exit(0 if success else 1)
    else:
        success = runner.run(start_from=args.start_from)
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()