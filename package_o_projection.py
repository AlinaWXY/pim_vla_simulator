"""Snapshot experiment sources, bundled dependencies and runner with hashes."""
import hashlib
import json
from pathlib import Path
import tarfile

root=Path(__file__).resolve().parent
files=sorted(p for p in (root/"experiments").rglob("*") if p.is_file()
    and "__pycache__" not in p.parts and "results" not in p.relative_to(root).parts
    and not any(part.startswith("build") for part in p.relative_to(root).parts[:-1])
    and p.suffix not in (".so",".pyd",".pyc"))
files += [root/"requirements.txt",root/"package_o_projection.py"]
files += [root/"reference"/case/name for case in ("chain","v","pv","gate_up_gpu","down_gpu_input","combined_timing_chain","serial_layer_control")
          for name in ("summary.json","native-audit.json")]
# Keep the historical CA source proof portable without including raw run data.
files += [root/"reference"/"dataflow-comparison"/"legacy-ca-source"/name for name in (
    "historical-lpddr5.py", "historical-lpddr5pim.py", "historical-spec.py",
    "localwr-act2-run.sbatch", "source-byte-verification.json", "SOURCE-SNAPSHOT.json",
    "direct-provenance.json",
)]
manifest={p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
(root/"O-PROJECTION-SOURCE.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
with tarfile.open(root/"o-projection-source.tar.gz","w:gz") as tar:
    for p in files:
        tar.add(p,arcname=p.relative_to(root).as_posix())
    tar.add(root/"O-PROJECTION-SOURCE.json",arcname="O-PROJECTION-SOURCE.json")
print(json.dumps(dict(files=len(files),bytes=(root/"o-projection-source.tar.gz").stat().st_size)))
