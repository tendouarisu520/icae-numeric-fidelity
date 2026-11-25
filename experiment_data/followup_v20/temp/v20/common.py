"""Shared paths and helpers for the V20 follow-up: seed extension and crossed-view choice scoring."""
from pathlib import Path
import hashlib
import json
import os
import time

R = Path('E:/Research/ICAE-ConditionalMemory-20261021')
O = R / 'results/v20-seeds-crossed-choice-20261113'
RUNS = R / 'runs/v20-seed-extension-20261113'
LOCK = R / 'temp/v20/lock.json'
OLD_SEEDS = (73, 97)
NEW_SEEDS = (101, 131, 151)
REPLAY_STEPS = 32
SFT_LABELS = ('main-compressed-sft-seed73-step512', 'main-compressed-sft-seed97-step512',
              'main-full-sft-seed73-step512', 'main-full-sft-seed97-step512')
COHORTS = {
    'development_crossed': 'data/v16-role-balance-20261106/development.jsonl',
    'new_scene_crossed': 'data/v19-prospective-20261108/crossed.jsonl',
    'new_scene_original': 'data/v19-prospective-20261108/original.jsonl',
    'development_original': 'data/v13-answerable-relations-20261026/development.jsonl',
}
ORIGINAL_RUNS = {
    'sft': 'runs/v13-answerable-reading-20261026/main-compressed-sft-seed{seed}',
    'bottleneck': 'runs/v14-bottleneck-reading-20261026/main-compressed-bottleneck-seed{seed}',
}


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(4194304), b''):
            h.update(block)
    return h.hexdigest()


def read(p):
    return json.loads(Path(p).read_text(encoding='utf8'))


def rows(p):
    return [json.loads(s) for s in Path(p).read_text(encoding='utf8').splitlines() if s.strip()]


def write(p, value):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + '.pending')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf8')
    for attempt in range(200):  # a reader may hold the target open for a moment on Windows
        try:
            os.replace(tmp, p)
            break
        except PermissionError:
            if attempt == 199:
                raise
            time.sleep(.05)


def environment():
    import sys
    assert R.drive.lower() == 'e:'
    os.chdir(R)
    for extra in (str(R / 'src'), str(R / 'temp/v20')):
        if extra not in sys.path:
            sys.path.insert(0, extra)
    os.environ['PYTHONPATH'] = str(R / 'src')
    for k, v in {'TEMP': 'temp', 'TMP': 'temp', 'HF_HOME': 'cache/huggingface', 'TORCH_HOME': 'cache/torch',
                 'CUDA_CACHE_PATH': 'cache/cuda', 'TRITON_CACHE_DIR': 'cache/triton',
                 'TORCHINDUCTOR_CACHE_DIR': 'cache/torchinductor', 'HF_DATASETS_CACHE': 'cache/datasets',
                 'HUGGINGFACE_HUB_CACHE': 'cache/huggingface/hub', 'XDG_CACHE_HOME': 'cache',
                 'NUMBA_CACHE_DIR': 'cache/numba', 'MPLCONFIGDIR': 'cache/matplotlib'}.items():
        p = R / v
        p.mkdir(parents=True, exist_ok=True)
        os.environ[k] = str(p)
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1')


def verify_lock():
    lock = read(LOCK)
    for name, h in lock['files'].items():
        assert sha(R / name) == h, name
    return lock


def emit(path, **values):
    v = dict(time=time.time(), pid=os.getpid(), **values)
    write(path, v)
    print(json.dumps(v), flush=True)


def schedule_v20(limit=None):
    """The schedule of binding.v11_training with the seed registry extended to the seeds fixed in the V20 plan.

    The algorithm is copied unchanged: one random.Random(seed), whole-epoch shuffles, truncation to `steps`.
    `limit` keeps only the first updates and is used for the 32-update replay of an original run.
    """
    import random
    registered = OLD_SEEDS + NEW_SEEDS

    def schedule(ids, seed, steps=1024):
        if seed not in registered or not 1 <= steps <= 1024 or len(ids) != len(set(ids)) or not ids:
            raise ValueError('Unique units, registered seed and <=1024 updates required')
        rng = random.Random(seed)
        result = []
        while len(result) < steps:
            epoch = list(ids)
            rng.shuffle(epoch)
            result.extend(epoch)
        result = result[:steps]
        return result if limit is None else result[:limit]
    return schedule
