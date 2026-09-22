"""Separate development and sealed-test workflows. No imports from the old project."""
import gc
import platform
import time
from pathlib import Path
import numpy as np
import torch
import transformers

from .common import read, write, digest, fingerprint, seed_all, fresh_dir, development_guard
from .m1 import load_split, statistics, allocate
from .metrics import entity_metrics
from .neural import (MODEL, MODEL_REVISION, tokenizer_for, load_model, encode, loader, device_for,
                     train_epoch, predict_logits, decode, save_checkpoint)
from .m4 import softmax, fit_temperature, gold_array, sample_predictive, report


def environment():
    return {"python": platform.python_version(), "torch": torch.__version__, "transformers": transformers.__version__,
            "numpy": np.__version__, "torch_threads": torch.get_num_threads(),
            "base_model_revision": MODEL_REVISION,
            "source_hashes": {p.name: digest(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
            "cuda": torch.cuda.is_available(), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}


def smoke_rows(rows):
    # Actual native documents only; shortest positive and negative for an execution check.
    selected = []
    for positive in (True, False):
        pool = [r for r in rows if bool(r["entities"]) == positive]
        if pool:
            selected.append(min(pool, key=lambda r: (len(r["tokens"]), r["record_id"])))
    return selected


def data_context(data, config, smoke=False):
    labels = read(Path(data)/"labels.json")
    train, validation = load_split(data, "train"), load_split(data, "validation")
    if smoke:
        train, validation = smoke_rows(train), smoke_rows(validation)
    return labels, train, validation


def make_batches(rows, tokenizer, labels, cfg, shuffle=False):
    print(f"Preparing {len(rows):,} documents (max_length={cfg['max_length']}, batch_size={cfg['batch_size']})...", flush=True)
    windows = encode(rows, tokenizer, labels, cfg["max_length"])
    batches = loader(windows, tokenizer, cfg["batch_size"], shuffle, cfg["seed"])
    truncations = [t for w in windows for t in w["token_truncations"]]
    print(f"Prepared {len(windows):,} windows / {len(batches):,} batches; oversized tokens={len(truncations)}", flush=True)
    return batches, {"documents": len(rows), "windows": len(windows),
                     "supervised_tokens": sum(len(r["tokens"]) for r in rows),
                     "oversized_native_tokens": len(truncations),
                     "discarded_subwords": sum(t["discarded_subwords"] for t in truncations),
                     "token_truncations": truncations,
                     "unknown_native_tokens": sum(w["unknown_native_tokens"] for w in windows)}


def evaluate(model, batches, rows, labels, device, epsilon=0.):
    start = time.perf_counter()
    logits = predict_logits(model, batches, rows, len(labels), device, epsilon)
    result = entity_metrics(rows, decode(logits, labels), sorted({t[2:] for t in labels if t != "O"}))
    result["seconds"] = time.perf_counter()-start
    return result


def checkpoint_signature(path):
    return {p.name: digest(p) for p in sorted(Path(path).glob("*")) if p.is_file()}


def validate_parent(data, path):
    parent = read(Path(path)/"run.json")
    if digest(Path(data)/"manifest.json") != parent["data_manifest_sha256"]:
        raise ValueError("Parent checkpoint was trained on a different split manifest")
    return parent


def baseline(data, output, cfg, smoke=False):
    print(f"M2 starting: loading fixed splits from {data}", flush=True)
    development_guard(data)
    if cfg["model"] != MODEL:
        raise ValueError("The baseline is fixed to microsoft/deberta-v3-small")
    root = fresh_dir(output)
    labels, train, val = data_context(data, cfg, smoke)
    tokenizer = tokenizer_for()
    device = device_for(cfg["device"])
    val_batches, val_coverage = make_batches(val, tokenizer, labels, cfg)
    subsets = read(Path(data)/"subsets.json")
    fractions = [100] if smoke else cfg["fractions"]
    results = []
    write(root/"run.json", {"stage": "M2", "smoke": smoke, "config": cfg, "environment": environment(),
                            "data_manifest_sha256": digest(Path(data)/"manifest.json"),
                            "subsets_sha256": digest(Path(data)/"subsets.json"),
                            "validation_record_ids": [r["record_id"] for r in val],
                            "purpose": "Execution verification only" if smoke else "Learning curve baseline"})
    for fraction in fractions:
        print(f"\nM2: starting training fraction {fraction}%", flush=True)
        seed_all(cfg["seed"])
        ids = set(subsets[str(fraction)]["record_ids"])
        rows = train if smoke else [r for r in train if r["record_id"] in ids]
        batches, coverage = make_batches(rows, tokenizer, labels, cfg, True)
        model = load_model(labels, device=device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
        directory = root/f"fraction_{fraction}"
        directory.mkdir()
        history, best, train_seconds = [], -1., 0.
        for epoch in range(1, (1 if smoke else cfg["epochs"])+1):
            loss = train_epoch(model, batches, optimizer, device, max_steps=1 if smoke else None,
                               description=f"M2 {fraction}% epoch {epoch}/{1 if smoke else cfg['epochs']}")
            print("M2: evaluating validation entities...", flush=True)
            measured = evaluate(model, val_batches, val, labels, device)
            history.append({"epoch": epoch, "training": loss, "validation": measured})
            train_seconds += loss["seconds"]
            if measured["micro"]["f1"] > best:
                best = measured["micro"]["f1"]
                save_checkpoint(model, tokenizer, directory/"best")
                selected_metrics = measured
            write(directory/"history.json", history)
            print(f"M2 fraction={fraction} epoch={epoch} validation F1={measured['micro']['f1']:.5f}", flush=True)
        results.append({"fraction": fraction, "checkpoint": str((directory/"best").resolve()),
                        "checkpoint_signature": checkpoint_signature(directory/"best"),
                        "training_statistics": statistics(rows), "training_record_ids": [r["record_id"] for r in rows],
                        "training_seconds": train_seconds, "coverage": coverage,
                        "validation_coverage": val_coverage, "validation": selected_metrics})
        del optimizer, model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        write(root/"learning_curve.json", results)
    best = max(results, key=lambda r: (r["validation"]["micro"]["f1"], -r["fraction"]))
    write(root/"selected.json", {**best, "selection_rule": "Highest validation strict micro F1; ties prefer less training data",
                                 "checkpoint_signature": checkpoint_signature(best["checkpoint"]), "smoke": smoke})
    return best


def robustness(data, baseline_dir, output, cfg, smoke=False):
    print(f"M3 starting: baseline={baseline_dir}", flush=True)
    development_guard(data)
    parent = validate_parent(data, baseline_dir)
    if parent["smoke"] != smoke:
        raise ValueError("Cannot mix smoke and research runs")
    selected = read(Path(baseline_dir)/"selected.json")
    if checkpoint_signature(selected["checkpoint"]) != selected["checkpoint_signature"]:
        raise ValueError("Baseline checkpoint changed")
    root = fresh_dir(output)
    labels, train, val = data_context(data, cfg, smoke)
    ids = set(selected["training_record_ids"])
    train = [r for r in train if r["record_id"] in ids]
    fraction = cfg["fgsm_train_fraction"]/100
    if not 0 < fraction <= 1:
        raise ValueError("Invalid FGSM training fraction")
    if fraction < 1:
        train = allocate(train, {"train": fraction, "unused": 1-fraction}, cfg["seed"])["train"]
    tokenizer = tokenizer_for(selected["checkpoint"])
    batches, coverage = make_batches(train, tokenizer, labels, cfg, True)
    vb, _ = make_batches(val, tokenizer, labels, cfg)
    device = device_for(cfg["device"])
    epsilons = cfg["fgsm_epsilons"][:1] if smoke else cfg["fgsm_epsilons"]
    write(root/"run.json", {"stage": "M3", "smoke": smoke, "config": cfg, "environment": environment(),
                            "data_manifest_sha256": digest(Path(data)/"manifest.json"),
                            "baseline_dir": str(Path(baseline_dir).resolve()),
                            "training_statistics": statistics(train), "coverage": coverage,
                            "training_record_ids": [r["record_id"] for r in train],
                            "attack": "White-box untargeted FGSM on input word embeddings; gold labels used only for benchmark attack. Not a realizable text edit."})
    results = []
    # Clean continuation controls additional optimization budget, separate from original M2.
    variants = [("clean_detector", None)] + [("clean_continuation", 0.)] + [(f"fgsm_{e}", e) for e in epsilons]
    for name, epsilon in variants:
        print(f"\nM3: starting variant {name}", flush=True)
        seed_all(cfg["seed"])
        model = load_model(labels, selected["checkpoint"], device)
        training = []
        if epsilon is not None:
            optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
            # Reset dataloader generator so variants see the same ordering.
            batches, _ = make_batches(train, tokenizer, labels, cfg, True)
            for epoch in range(1 if smoke else cfg["fgsm_epochs"]):
                training.append(train_epoch(model, batches, optimizer, device, epsilon, 1 if smoke else None,
                                            description=f"M3 {name} epoch {epoch+1}/{1 if smoke else cfg['fgsm_epochs']}"))
            checkpoint = str((root/name).resolve())
            save_checkpoint(model, tokenizer, checkpoint)
            del optimizer
        else:
            checkpoint = selected["checkpoint"]
        clean = evaluate(model, vb, val, labels, device)
        attacks = {str(e): evaluate(model, vb, val, labels, device, e) for e in epsilons}
        results.append({"name": name, "training_epsilon": epsilon, "training": training, "checkpoint": checkpoint,
                        "checkpoint_signature": checkpoint_signature(checkpoint),
                        "clean_validation": clean, "adversarial_validation": attacks,
                        "selection_score": float(np.mean([clean["micro"]["f1"]]+[m["micro"]["f1"] for m in attacks.values()]))})
        write(root/"comparison.json", results)
        print(f"M3 {name}: clean validation F1={clean['micro']['f1']:.5f}", flush=True)
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    best = max(results, key=lambda r: r["selection_score"])
    write(root/"selected.json", {**best, "checkpoint_signature": checkpoint_signature(best["checkpoint"]),
                                 "selection_rule": "Mean validation strict micro F1 across clean and predeclared epsilon grid; ties prefer original clean detector",
                                 "smoke": smoke})
    return best


def uncertainty(data, robustness_dir, output, cfg, smoke=False):
    print(f"M4 starting: detector selection={robustness_dir}", flush=True)
    development_guard(data)
    parent = validate_parent(data, robustness_dir)
    if parent["smoke"] != smoke:
        raise ValueError("Cannot mix smoke and research runs")
    selected = read(Path(robustness_dir)/"selected.json")
    if checkpoint_signature(selected["checkpoint"]) != selected["checkpoint_signature"]:
        raise ValueError("Selected detector checkpoint changed")
    root = fresh_dir(output)
    labels = read(Path(data)/"labels.json")
    cal, val = load_split(data, "calibration"), load_split(data, "validation")
    if smoke:
        cal, val = smoke_rows(cal), smoke_rows(val)
    tokenizer = tokenizer_for(selected["checkpoint"])
    cb, cc = make_batches(cal, tokenizer, labels, cfg)
    vb, vc = make_batches(val, tokenizer, labels, cfg)
    device = device_for(cfg["device"])
    model = load_model(labels, selected["checkpoint"], device)
    write(root/"run.json", {"stage": "M4", "smoke": smoke, "config": cfg, "environment": environment(),
                            "data_manifest_sha256": digest(Path(data)/"manifest.json"),
                            "robustness_dir": str(Path(robustness_dir).resolve()), "checkpoint": selected["checkpoint"],
                            "checkpoint_signature": selected["checkpoint_signature"],
                            "calibration_coverage": cc, "validation_coverage": vc,
                            "calibration_record_ids": [r["record_id"] for r in cal],
                            "validation_record_ids": [r["record_id"] for r in val],
                            "calibration_method": "Scalar temperature on log predictive probabilities; independently fit for deterministic and each MC pass budget on calibration only."})
    yg = gold_array(cal, labels)
    pc = softmax(np.concatenate(predict_logits(model, cb, cal, len(labels), device)))
    temperature = fit_temperature(pc, yg)
    start = time.perf_counter()
    pv = softmax(np.concatenate(predict_logits(model, vb, val, len(labels), device)))
    elapsed = time.perf_counter()-start
    deterministic = report(val, pv, labels, temperature, cfg["ece_bins"])
    deterministic["inference_seconds"] = elapsed
    write(root/"deterministic.json", deterministic)
    passes = [5] if smoke else cfg["mc_passes"]
    temperatures = {}
    for count, p, _ in sample_predictive(model, cb, cal, labels, device, passes, cfg["seed"]):
        temperatures[count] = fit_temperature(p, yg)
    results = []
    write(root/"validation_index.json", [{"record_id": r["record_id"], "tokens": len(r["tokens"])} for r in val])
    for count, p, u in sample_predictive(model, vb, val, labels, device, passes, cfg["seed"]+10000):
        measured = report(val, p, labels, temperatures[count], cfg["ece_bins"], u["predictive_entropy"])
        measured.update({"passes": count, "inference_seconds": u["seconds"],
                         "seconds_per_document": u["seconds"]/len(val),
                         "mean_predictive_entropy_raw": float(u["predictive_entropy"].mean()),
                         "mean_expected_entropy_raw": float(u["expected_entropy"].mean()),
                         "mean_mutual_information_raw": float(u["mutual_information"].mean())})
        results.append(measured)
        np.savez_compressed(root/f"validation_mc_{count}.npz", raw_mean_probabilities=p,
                            predictive_entropy=u["predictive_entropy"], expected_entropy=u["expected_entropy"],
                            mutual_information=u["mutual_information"], temperature=temperatures[count])
        write(root/"mc_comparison.json", results)
        print(f"M4 passes={count} validation NLL={measured['calibrated_token_metrics']['nll']:.5f}", flush=True)
    best_nll = min(r["calibrated_token_metrics"]["nll"] for r in results)
    best = min((r for r in results if r["calibrated_token_metrics"]["nll"] <= best_nll+cfg["mc_nll_tolerance"]), key=lambda r: r["passes"])
    write(root/"selected.json", {"passes": best["passes"], "temperature": best["temperature"],
                                 "deterministic_temperature": temperature, "smoke": smoke,
                                 "selection_rule": "Fewest passes within configured absolute validation token NLL tolerance of best MC candidate. ECE/AURC/per-class metrics reported separately.",
                                 "nll_tolerance": cfg["mc_nll_tolerance"]})
    return best


def final_test(data, baseline_dir, robustness_dir, uncertainty_dir, output, resume=False):
    """Freeze first, then open the fixed test once for all predeclared comparisons."""
    if not resume:
        development_guard(data)
    m2run = validate_parent(data, baseline_dir)
    m3run = validate_parent(data, robustness_dir)
    m4run = validate_parent(data, uncertainty_dir)
    if any(r["smoke"] for r in (m2run, m3run, m4run)):
        raise RuntimeError("Smoke checkpoints cannot open the research test set")
    if Path(m3run["baseline_dir"]).resolve() != Path(baseline_dir).resolve() or Path(m4run["robustness_dir"]).resolve() != Path(robustness_dir).resolve():
        raise ValueError("M2/M3/M4 lineage mismatch")
    root = Path(output) if resume else fresh_dir(output)
    frozen = {"baseline": str(Path(baseline_dir).resolve()), "robustness": str(Path(robustness_dir).resolve()),
              "uncertainty": str(Path(uncertainty_dir).resolve()), "output": str(root.resolve()),
              "selection_hashes": {name: digest(Path(path)/"selected.json") for name, path in
                                   (("m2", baseline_dir), ("m3", robustness_dir), ("m4", uncertainty_dir))},
              "run_hashes": {name: digest(Path(path)/"run.json") for name, path in
                             (("m2", baseline_dir), ("m3", robustness_dir), ("m4", uncertainty_dir))},
              "comparison_hashes": [digest(Path(baseline_dir)/"learning_curve.json"), digest(Path(robustness_dir)/"comparison.json")]}
    for directory, filename in ((baseline_dir, "learning_curve.json"), (robustness_dir, "comparison.json")):
        for candidate in read(Path(directory)/filename):
            if checkpoint_signature(candidate["checkpoint"]) != candidate["checkpoint_signature"]:
                raise ValueError("A frozen candidate checkpoint changed")
    if resume:
        if read(Path(data)/"TEST_OPENED.json") != frozen:
            raise ValueError("Resume requires identical frozen inputs and output path")
    else:
        write(Path(data)/"TEST_OPENED.json", frozen)
    labels = read(Path(data)/"labels.json")
    rows = load_split(data, "test")
    results = {"frozen": frozen, "learning_curve": [], "robustness": []}
    if resume and (root/"results.json").exists():
        results = read(root/"results.json")
        if results["frozen"] != frozen:
            raise ValueError("Result lineage changed")
    for phase, directory, filename, cfg in (("learning_curve", baseline_dir, "learning_curve.json", m2run["config"]),
                                           ("robustness", robustness_dir, "comparison.json", m3run["config"])):
        for candidate in read(Path(directory)/filename):
            identity = candidate.get("name", candidate.get("fraction"))
            if any(item["candidate"] == identity for item in results[phase]):
                continue
            checkpoint = candidate["checkpoint"]
            tokenizer = tokenizer_for(checkpoint)
            batches, _ = make_batches(rows, tokenizer, labels, cfg)
            device = device_for(cfg["device"])
            model = load_model(labels, checkpoint, device)
            measured = {"candidate": identity,
                        "clean_test": evaluate(model, batches, rows, labels, device)}
            if phase == "robustness":
                measured["adversarial_test"] = {str(e): evaluate(model, batches, rows, labels, device, e) for e in cfg["fgsm_epsilons"]}
            results[phase].append(measured)
            write(root/"results.json", results)
            del model
            gc.collect()
    cfg = m4run["config"]
    chosen = read(Path(uncertainty_dir)/"selected.json")
    checkpoint = m4run["checkpoint"]
    if checkpoint_signature(checkpoint) != m4run["checkpoint_signature"]:
        raise ValueError("Frozen M4 checkpoint changed")
    tokenizer = tokenizer_for(checkpoint)
    batches, _ = make_batches(rows, tokenizer, labels, cfg)
    device = device_for(cfg["device"])
    model = load_model(labels, checkpoint, device)
    pd = softmax(np.concatenate(predict_logits(model, batches, rows, len(labels), device)))
    results["deterministic_calibration"] = report(rows, pd, labels, chosen["deterministic_temperature"], cfg["ece_bins"])
    for count, p, u in sample_predictive(model, batches, rows, labels, device, [chosen["passes"]], cfg["seed"]+20000):
        results["mc_calibration"] = report(rows, p, labels, chosen["temperature"], cfg["ece_bins"], u["predictive_entropy"])
        results["mc_calibration"]["inference_seconds"] = u["seconds"]
    write(root/"results.json", results)
    return results
