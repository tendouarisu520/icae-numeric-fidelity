"""V20 collector: CPU only. Runs the frozen analysis and writes one evidence archive without model tensors.

The archive keeps the host's relative paths, so analysis.py can be rerun on the unpacked folder. It holds the
V20 scripts, plan and lock, every V20 output except adapter tensors, the V20 logs, and the small earlier files the
analysis reads (the four cohort data files and the four original run summaries).
"""
from common import *
import zipfile
import analysis


def main():
    environment()
    lock = verify_lock()
    out = O / 'analysis'
    out.mkdir(exist_ok=False)
    archive = O / 'completed_evidence.zip'
    assert not archive.exists()
    report = analysis.run(R)
    write(out / 'report.json', report)
    adapters = [dict(path=p.relative_to(R).as_posix(), sha256=sha(p), bytes=p.stat().st_size)
                for p in sorted(RUNS.rglob('*.safetensors'))] if RUNS.exists() else []
    write(out / 'adapters.json', adapters)
    skip = ('.safetensors', '.pt', '.zip', '.pending', '.b64')
    files = set()
    for base in (R / 'temp/v20', O, RUNS):
        if base.exists():
            files |= {p for p in base.rglob('*') if p.is_file() and p.suffix not in skip and '__pycache__' not in p.parts}
    files |= {p for p in (R / 'logs').glob('v20-*') if 'v20-collect-' not in p.name}
    files.add(R / lock['plan'])
    files |= {R / path for path in COHORTS.values()}
    for pattern in ORIGINAL_RUNS.values():
        for seed in OLD_SEEDS:
            files.add(R / pattern.format(seed=seed) / 'summary.json')
    # The queue rewrites its status file while this job runs; it is fetched separately after the queue ends.
    files -= {O / 'evidence_receipt.json', O / 'queue_status.json'}
    with zipfile.ZipFile(archive, 'x', zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for p in sorted(files):
            z.write(p, p.relative_to(R).as_posix())
    receipt = dict(sha256=sha(archive), bytes=archive.stat().st_size, entries=len(files), time=time.time(),
                   lock_sha256=sha(LOCK), report_sha256=sha(out / 'report.json'), adapters=len(adapters))
    write(O / 'evidence_receipt.json', receipt)
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
