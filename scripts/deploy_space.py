"""Publish the app as a Hugging Face Space (Docker SDK).

One time:
    hf auth login          # paste a token with WRITE access from
                           # https://huggingface.co/settings/tokens

Then:
    python scripts/deploy_space.py                  # <you>/syllabus-tree-tutor
    python scripts/deploy_space.py --space me/other --private

Hugging Face builds the Docker image itself; the first build takes about ten
minutes. The Space needs its own README (with a YAML header telling Spaces how
to run it), so the files are staged in a temporary folder with that README in
place of the GitHub one.
"""

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# The Windows console is cp1252; the Space README contains an emoji.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Everything the Dockerfile needs, and nothing else.
FILES = ["Dockerfile", ".dockerignore", "requirements.txt"]
FOLDERS = ["backend", "scripts", "eval", "frontend", "deploy/demo"]
SKIP = ["node_modules", "dist", "__pycache__", "*.pyc", ".vite"]

# Binary files must go through Git LFS on the Hub.
GITATTRIBUTES = "*.pdf filter=lfs diff=lfs merge=lfs -text\n" \
                "*.png filter=lfs diff=lfs merge=lfs -text\n"


def stage(target: Path) -> None:
    for name in FILES:
        shutil.copy2(ROOT / name, target / name)
    for name in FOLDERS:
        destination = target / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(ROOT / name, destination, ignore=shutil.ignore_patterns(*SKIP))
    shutil.copy2(ROOT / "deploy" / "space" / "README.md", target / "README.md")
    (target / ".gitattributes").write_text(GITATTRIBUTES, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--space", default="", help="owner/name; defaults to <you>/syllabus-tree-tutor")
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    from huggingface_hub import HfApi

    api = HfApi()
    try:
        me = api.whoami()
    except Exception:
        print("Not logged in to Hugging Face. Run:  hf auth login")
        print("and paste a token with WRITE access from https://huggingface.co/settings/tokens")
        sys.exit(1)

    space = args.space if args.space != "" else me["name"] + "/syllabus-tree-tutor"
    print("publishing to", space, "as", me["name"])

    api.create_repo(space, repo_type="space", space_sdk="docker",
                    private=args.private, exist_ok=True)

    with tempfile.TemporaryDirectory() as temporary:
        target = Path(temporary) / "space"
        target.mkdir()
        stage(target)

        count = 0
        for path in target.rglob("*"):
            if path.is_file():
                count = count + 1
        print("uploading", count, "files")

        api.upload_folder(
            folder_path=str(target),
            repo_id=space,
            repo_type="space",
            commit_message="Deploy from GitHub: Syllabus Tree Tutor",
        )

    url = "https://huggingface.co/spaces/" + space
    print()
    print("done:", url)
    print("the image now builds on Hugging Face; watch the logs at", url + "?logs=build")


if __name__ == "__main__":
    main()
