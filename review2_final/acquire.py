"""Download the publisher-linked, version-pinned public PIILO release."""
from pathlib import Path
from urllib.request import urlopen
import shutil
import zipfile
from .common import write, read, digest


def main():
    root = Path(__file__).resolve().parent
    destination = root/"data/raw/train.json"
    if destination.exists():
        expected = "8276cd44f3b2eb357dfb405b3c5d8e9f821388e984cbf66e92e7df03f1b13117"
        if digest(destination) != expected:
            raise ValueError("Existing raw file does not match audited PIILO v2")
        print("Verified existing PIILO v2")
        return
    archive = root/"piilo.zip"
    url = "https://www.kaggle.com/api/v1/datasets/download/lburleigh/piilo-dataset?datasetVersionNumber=2"
    with urlopen(url, timeout=120) as source, archive.open("wb") as target:
        shutil.copyfileobj(source, target)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zipped:
        # Extract only the verified file, never arbitrary archive paths.
        with zipped.open("train.json") as source, destination.open("wb") as target:
            shutil.copyfileobj(source, target)
    if digest(destination) != "8276cd44f3b2eb357dfb405b3c5d8e9f821388e984cbf66e92e7df03f1b13117":
        raise ValueError("Downloaded file differs from audited v2; stop and re-audit")
    write(root/"reports/acquisition.json", {"url": url, "sha256": digest(destination),
        "publisher": "https://the-learning-agency.com/guides-resources/datasets/",
        "license": "CC BY 4.0", "release": 2})
    print("Downloaded and verified PIILO v2")


if __name__ == "__main__":
    main()
