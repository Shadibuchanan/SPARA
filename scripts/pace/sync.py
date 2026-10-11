"""Run locally: deploy only source files to the user's configured SSH target."""
import argparse
import io
import os
from pathlib import Path
import shlex
import subprocess
import tarfile

parser = argparse.ArgumentParser()
parser.add_argument("--target", default=os.environ.get("SPARA_SSH_TARGET", "pace-ice"))
parser.add_argument("--root", default=os.environ.get("SPARA_REMOTE_ROOT", "~/scratch/SPARA"))
args = parser.parse_args()
project = Path(__file__).resolve().parents[2]
payload = io.BytesIO()
with tarfile.open(fileobj=payload, mode="w:gz") as archive:
    for name in ("pyproject.toml", "README.md", "src", "tests", "configs", "scripts", "requirements"):
        path = project / name
        if not path.exists(): continue
        paths = [path] if path.is_file() else sorted(path.rglob("*"))
        for file in paths:
            if file.is_file() and "__pycache__" not in file.parts:
                archive.add(file, arcname=str(file.relative_to(project)).replace("\\", "/"))
remote = '''import sys, pathlib, tarfile, io
root=pathlib.Path(sys.argv[1]).expanduser().resolve()/"code"
if root.exists() and not (root/".spara-managed").exists():
    raise RuntimeError("Refusing to overwrite an unmanaged code directory")
root.mkdir(parents=True,exist_ok=True)
with tarfile.open(fileobj=io.BytesIO(sys.stdin.buffer.read()),mode="r:gz") as archive:
    for member in archive.getmembers():
        if not (root/member.name).resolve().is_relative_to(root) or not member.isfile():
            raise ValueError("Unsafe source member")
    archive.extractall(root,filter="data")
(root/".spara-managed").write_text("Managed source deployment; no data or results are removed.\\n")
print(root)
'''
subprocess.run(["ssh", "-o", "BatchMode=yes", args.target,
                "python3 -c " + shlex.quote(remote) + " " + shlex.quote(args.root)],
               input=payload.getvalue(), check=True)
