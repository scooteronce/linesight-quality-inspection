"""Bundle Linux ARM64 wheels for the AgentCore Python 3.13 CodeZip runtime."""

import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "runtime"
BUNDLE = RUNTIME / "bundle"


def main() -> None:
    if sys.version_info < (3, 13):
        raise SystemExit("Run with the repository Python 3.13 environment: npm run package:runtime")

    if BUNDLE.exists():
        shutil.rmtree(BUNDLE)
    vendor = BUNDLE / "vendor"
    vendor.mkdir(parents=True)

    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--only-binary=:all:",
            "--no-compile",
            "--target",
            str(vendor),
            "--platform",
            "manylinux2014_aarch64",
            "--implementation",
            "cp",
            "--python-version",
            "3.13",
            "--abi",
            "cp313",
            "-r",
            str(RUNTIME / "requirements.txt"),
        ],
        check=True,
    )
    shutil.copy2(RUNTIME / "main.py", BUNDLE / "main.py")
    embeddings = RUNTIME / "reference_embeddings.json"
    if not embeddings.is_file():
        raise SystemExit(
            "runtime/reference_embeddings.json is missing. "
            "Run `python scripts/embed_references.py` first."
        )
    shutil.copy2(embeddings, BUNDLE / "reference_embeddings.json")
    print(f"Packaged AgentCore Python 3.13 dependencies in {BUNDLE}")


if __name__ == "__main__":
    main()
