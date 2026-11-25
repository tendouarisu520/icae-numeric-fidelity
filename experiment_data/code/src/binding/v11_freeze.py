"""Freeze the reviewed new natural QA cohort; no old data interfaces."""
import json
from pathlib import Path
from .v11_pipeline import read_rows,balanced_select,sha,write_json

ROOT=Path('E:/Research/ICAE-ConditionalMemory-20261021')


def main():
    source=ROOT/'data/v11-natural-admitted-20261025-r3';output=ROOT/'data/v11-natural-frozen-20261025'
    receipt_path=ROOT/'docs/v11-natural-data-review-20261025.json'
    receipt=json.loads(receipt_path.read_text(encoding='utf8'))
    preview=json.loads((ROOT/'results/v11-natural-r3-preview-20261025.json').read_text(encoding='utf8'))
    if receipt['status']!='pass' or receipt['preview']!=preview:raise ValueError('Exact reviewed cohort receipt required')
    output.mkdir(exist_ok=False)
    result=dict(recipe='v11-natural-relations-3',source_manifest_sha256=sha(source/'manifest.json'),
                review_receipt_sha256=sha(receipt_path),overfit_ids=preview['overfit'],partitions={},
                original_official_dev_used=False,old_confirmation_accessed=False,reserved_test_accessed=False)
    for split,count,cap in [('train',512,8),('development',32,2),('confirmation',64,8)]:
        selected=balanced_select(read_rows(source/(split+'.jsonl')),count,cap)
        if split=='development' and [u['id'] for u in selected]!=preview['development']:raise ValueError('Selection changed')
        path=output/(split+'.jsonl')
        path.write_text(''.join(json.dumps(u,ensure_ascii=False,separators=(',',':'))+'\n' for u in selected),encoding='utf8')
        result['partitions'][split]=dict(units=count,articles=len({u['article'] for u in selected}),sha256=sha(path),article_cap=cap)
    write_json(output/'manifest.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
