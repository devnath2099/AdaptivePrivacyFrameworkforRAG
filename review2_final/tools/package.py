"""Build a portable code-only GPU bundle and verify local artifact integrity."""
import ast
import json
import shutil
import zipfile
from pathlib import Path

from review2_final.common import read, write, digest


def main():
    root = Path(__file__).resolve().parents[1]
    manifest = read(root/"data/prepared/manifest.json")
    checks = {}
    for name, expected in manifest["files"].items():
        checks[name] = digest(root/f"data/prepared/{name}.json") == expected
    if not all(checks.values()):
        raise ValueError("A prepared split changed")
    subsets = read(root/"data/prepared/subsets.json")
    previous = set()
    for fraction in (25,50,75,100):
        ids = set(subsets[str(fraction)]["record_ids"])
        if not previous <= ids:
            raise ValueError("Learning-curve subsets are not nested")
        previous = ids
    for path in root.rglob("*.py"):
        ast.parse(path.read_text(encoding="utf-8"))
    notebook = read(root/"run_gpu.ipynb")
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))
    reports = root/"reports"
    reports.mkdir(exist_ok=True)
    shutil.copy2(root/"data/prepared/audit.json", reports/"dataset_audit.json")
    shutil.copy2(root/"data/prepared/manifest.json", reports/"split_manifest.json")
    write(reports/"integrity.json", {"split_hashes_match": checks, "subsets_nested": True,
                                   "test_evaluation_opened": (root/"data/prepared/TEST_OPENED.json").exists(),
                                   "notebook_code_cells_parse": True})
    paths = [p for p in root.iterdir() if p.is_file() and p.suffix in (".py", ".json", ".md", ".txt", ".ipynb")]
    paths += list((root/"tests").glob("*.py")) + list((root/"tools").glob("*.py"))
    paths += list(reports.glob("*.json")) + list(reports.glob("*.xml"))
    archive_path = root/"review2_final_source.zip"
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(paths):
            archive.write(path, Path("review2_final")/path.relative_to(root))
    print(json.dumps({"bundle": str(archive_path), "bytes": archive_path.stat().st_size,
                      "sha256": digest(archive_path), "integrity": read(reports/"integrity.json")}, indent=2))


if __name__ == "__main__":
    main()
