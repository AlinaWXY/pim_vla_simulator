"""Copy existing results into a publication worktree; never run simulation."""
import hashlib
import json
import shutil
import subprocess
import tarfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

BASE = Path('/fact_data/xinyaowang')
SOURCE = BASE / 'pim_vla_gio_bcast_e2e_20261109-112224'
CAMPAIGN = SOURCE / 'results/e2e-gio-bcast-calibrated'
WORKTREE = BASE / 'pim_vla_V2_publish_20261109'
DEST = WORKTREE / 'reference/e2e-gio-bcast-20261109'
AUDIT = BASE / 'gio-e2e-audit-20261109-112224'
DATA = DEST / 'data'

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

assert subprocess.check_output(['git', '-C', str(WORKTREE), 'rev-parse', 'HEAD'], text=True).strip() == '6bcd57b073e86674f5abec24eee56ecde8252f30'
assert subprocess.check_output(['git', '-C', str(WORKTREE), 'branch', '--show-current'], text=True).strip() == 'xinyao_1109_V2'
assert not (DATA / 'campaign').exists(), 'Publication copy already exists; refusing overwrite'
DATA.mkdir(parents=True, exist_ok=True)
shutil.copytree(CAMPAIGN, DATA / 'campaign', ignore=lambda path, names: ['frozen'] if Path(path) == CAMPAIGN else [])

inventory = []
for path in sorted(CAMPAIGN.rglob('*')):
    if path.is_file():
        assert not path.is_symlink(), str(path)
        rel = path.relative_to(CAMPAIGN).as_posix()
        entry = dict(path=rel, bytes=path.stat().st_size, sha256=sha(path))
        if not rel.startswith('frozen/'):
            assert sha(DATA / 'campaign' / rel) == entry['sha256']
        inventory.append(entry)
print('Copied and checked all non-frozen campaign files', flush=True)

archive = DATA / 'frozen-runtime.tar.gz'
with tarfile.open(archive, 'w:gz', compresslevel=6) as out:
    out.add(CAMPAIGN / 'frozen', arcname='frozen')
assert archive.stat().st_size < 100 * 1024 * 1024, 'Split archive required before Git publication'
expected = {r['path']: r for r in inventory if r['path'].startswith('frozen/')}
seen = set()
with tarfile.open(archive, 'r:gz') as packed:
    for member in packed:
        if member.isfile():
            assert member.name in expected, member.name
            h = hashlib.sha256()
            with packed.extractfile(member) as stream:
                for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                    h.update(block)
            assert h.hexdigest() == expected[member.name]['sha256'], member.name
            seen.add(member.name)
assert seen == set(expected)
(DATA / 'campaign-inventory.json').write_text(json.dumps(inventory, indent=2) + '\n')
print('Frozen archive verified:', archive.stat().st_size, 'bytes;', len(seen), 'files', flush=True)

for path in sorted((SOURCE / 'results').iterdir()):
    if path.name != CAMPAIGN.name:
        assert path.is_dir(), str(path)
        shutil.copytree(path, DATA / 'validation-results' / path.name)
shutil.copytree(AUDIT, DEST / 'validation' / 'native-audit')
job_evidence = DEST / 'validation' / 'job-evidence'
job_evidence.mkdir(parents=True)
for path in sorted(BASE.glob('gio-e2e-*')):
    if path.is_file() and path.suffix in {'.log', '.sbatch', '.sh', '.py'}:
        shutil.copy2(path, job_evidence / path.name)
shutil.copy2(CAMPAIGN / 'validation/validation-context.json', DEST / 'validation/validation-context.json')
status = subprocess.check_output(['timeout', '20', 'sacct', '-X', '-j', '1981710,1981711,1981712', '-o', 'JobID,State,Elapsed,ExitCode', '-n', '-P'], text=True)
assert '|RUNNING|' not in status and '|PENDING|' not in status
(DATA / 'cancellation-status.txt').write_text(status)

metadata = {
    'schema_version': 1,
    'prepared_at': datetime.now(timezone(timedelta(hours=8))).isoformat(),
    'status': 'cancelled at user request after seven completed cases',
    'simulation_restarted_for_publication': False,
    'requested_branch': 'xinyao_1109_V2',
    'pulled_source_commit': '98a66712755010d3e9e45d0c62663e0702aa13ee',
    'run_source_commit': '6bcd57b073e86674f5abec24eee56ecde8252f30',
    'completed_cases': ['b', 'c', 'd', 'e', 'e-frfcfs', 'e-lookahead', 'e-gap'],
    'repetitions_each': 2,
    'incomplete_cases': ['f', 'g'],
    'calibration_passed': True,
    'full_nine_case_campaign_audit_completed': False,
    'all_original_campaign_files_preserved': True,
    'original_campaign_file_count': len(inventory),
    'original_campaign_bytes': sum(r['bytes'] for r in inventory),
    'frozen_archive': {'path': 'frozen-runtime.tar.gz', 'bytes': archive.stat().st_size, 'sha256': sha(archive)},
    'restore_frozen': 'tar -xzf data/frozen-runtime.tar.gz -C data/campaign',
    'notes': [
        'configs/provenance retain original absolute server paths; they are evidence, not portable launch commands',
        'cases-sequential-attempt is a cancelled initial attempt, never mixed into completed results',
        'original simulator backup remains on server; backup checksums and location are included separately',
        'validation-results includes single-layer/concurrency smoke outputs, not full fused e2e results',
    ],
}
(DATA / 'publication.json').write_text(json.dumps(metadata, indent=2) + '\n')
print(json.dumps(metadata, indent=2), flush=True)
