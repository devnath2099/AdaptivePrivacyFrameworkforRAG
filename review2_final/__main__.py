import argparse
from pathlib import Path
from .common import read, write


def main():
    parser = argparse.ArgumentParser(description="Fresh PIILO M1--M4 research pipeline")
    sub = parser.add_subparsers(dest="stage", required=True)
    base = Path(__file__).resolve().parent
    audit = sub.add_parser("audit")
    audit.add_argument("--raw", default=str(base/"data/raw/train.json"))
    audit.add_argument("--metadata", default=str(base/"kaggle_release.json"))
    audit.add_argument("--output", default=str(base/"reports/dataset_audit.json"))
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--raw", default=str(base/"data/raw/train.json"))
    prepare.add_argument("--metadata", default=str(base/"kaggle_release.json"))
    prepare.add_argument("--output", default=str(base/"data/prepared"))
    prepare.add_argument("--seed", type=int, default=20260922)
    prepare.add_argument("--proportions", type=float, nargs=4, required=True, metavar=("TRAIN", "CALIBRATION", "VALIDATION", "TEST"))
    for name in ("baseline", "robustness", "uncertainty", "smoke"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--data", default=str(base/"data/prepared"))
        cmd.add_argument("--config", default=str(base/"config.json"))
        cmd.add_argument("--output", required=True)
        if name == "robustness":
            cmd.add_argument("--baseline", required=True)
        if name == "uncertainty":
            cmd.add_argument("--robustness", required=True)
    final = sub.add_parser("final-test")
    final.add_argument("--data", default=str(base/"data/prepared"))
    final.add_argument("--baseline", required=True)
    final.add_argument("--robustness", required=True)
    final.add_argument("--uncertainty", required=True)
    final.add_argument("--output", required=True)
    final.add_argument("--resume", action="store_true", help="Resume identical frozen test inputs after interruption")
    args = parser.parse_args()
    if args.stage == "audit":
        from .m1 import audit
        _, report = audit(args.raw, read(args.metadata))
        write(args.output, report)
        print(report["statistics"])
    elif args.stage == "prepare":
        from .m1 import build
        result = build(args.raw, args.metadata, args.output,
                       dict(zip(("train", "calibration", "validation", "test"), args.proportions)), args.seed)
        print(result["splits"])
    else:
        from .experiments import baseline, robustness, uncertainty, final_test
        import torch
        torch.set_num_threads(4)
        if args.stage == "final-test":
            final_test(args.data, args.baseline, args.robustness, args.uncertainty, args.output, args.resume)
            return
        config = read(args.config)
        if args.stage == "baseline":
            baseline(args.data, args.output, config)
        elif args.stage == "robustness":
            robustness(args.data, args.baseline, args.output, config)
        elif args.stage == "uncertainty":
            uncertainty(args.data, args.robustness, args.output, config)
        else:
            # Real pretrained backbone + native records, one step, no test access.
            config = {**config, "max_length": 128, "batch_size": 1, "epochs": 1, "fgsm_epochs": 1}
            root = Path(args.output)
            baseline(args.data, root/"m2", config, smoke=True)
            robustness(args.data, root/"m2", root/"m3", config, smoke=True)
            uncertainty(args.data, root/"m3", root/"m4", config, smoke=True)
            write(root/"STATUS.json", {"status": "passed", "scope": "Real-data M2/M3/M4 execution smoke; not thesis metrics; research test unopened"})


if __name__ == "__main__":
    main()
