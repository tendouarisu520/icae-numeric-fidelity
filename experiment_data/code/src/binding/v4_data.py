"""V4: unchanged training bytes and fresh validation only; never read old test."""
import argparse
import collections
import json
from pathlib import Path
import random
import re
import shutil
import string

from .data import CONDITIONS, VALUES, statement, question, read_jsonl, write_jsonl, validate_rows
from .paired_data import LEVELS, _PATTERNS, _label_pairs, canonical_hash, sha256_file, audit_suite as audit_v2

SPLITS = ('train', 'val_id', 'val_ood')
SOURCE_TRAIN_SHA256 = '04e1c41d22f3b3d2a59ab8512912d151f0cc8448d1e8a2f770369810bc316fb1'
SEED = 20261022
SOURCE_NAMES = ('data.py', 'paired_data.py', 'v4_data.py')


def source_hashes():
    return {name: sha256_file(Path(__file__).with_name(name)) for name in SOURCE_NAMES}


def make_suite(train, seed=SEED):
    if seed != SEED:
        raise ValueError('V4 validation seed is frozen at 20261022')
    audit_v2({'train': train}, splits=('train',))
    rng = random.Random(seed)
    used = {name for row in train for name in row['entities']}
    def name():
        while True:
            token = 'Team ' + ''.join(rng.choices(string.ascii_lowercase, k=12))
            if token not in used:
                used.add(token)
                return token
    suite = {'train': train}
    for split in SPLITS[1:]:
        rows = []
        specs=[(family,level,absent,template) for family in CONDITIONS for level in LEVELS
            for absent in range(3) for template in ((0,1) if split=='val_id' else (2,))]
        pairs=_label_pairs(specs,random.Random(canonical_hash([seed,split,'labels'])),split=='val_id')
        for i,(family,level,absent,template) in enumerate(specs):
            fi,li=list(CONDITIONS).index(family),LEVELS.index(level)
            conditions = CONDITIONS[family]
            present = [j for j in range(3) if j != absent]
            labels = pairs[i]
            entity = name()
            foreign = [name() for _ in range(level//2)]
            distractors = [statement(foreign[j//2] if j % 2 else entity,
                rng.choice(conditions) if j % 2 else f'special assignment {j+10}',
                rng.choice(VALUES), template) for j in range(level)]
            order = list(range(level+2)); rng.shuffle(order)
            evidence, contexts = [], []
            for variant in range(2):
                assigned = labels if variant == 0 else labels[::-1]
                support = [statement(entity,conditions[k],assigned[j],template) for j,k in enumerate(present)]
                evidence.append(support)
                parts = support+distractors
                contexts.append('\n'.join(parts[j] for j in order))
            scenario = f'v4-s{seed}-{split}-{i:04d}'
            shared = dict(scenario_id=scenario,split=split,family=family,difficulty=level,
                entity=entity,entities=[entity]+foreign,template_id=template,absent_index=absent)
            for target,index in enumerate(present):
                for variant in range(2):
                    assigned = labels if variant == 0 else labels[::-1]
                    rows.append(dict(shared,id=f'{scenario}-q{index}-v{variant}',
                        pair_id=f'{scenario}-q{index}',kind='swap',queried_index=index,
                        variant=variant,condition=conditions[index],context=contexts[variant],
                        question=question(entity,conditions[index]),answer=assigned[target],foil=assigned[1-target],
                        support=evidence[variant][target],hard_negative=evidence[variant][1-target]))
            variant = (fi+li+absent+template)%2
            rows.append(dict(shared,id=f'{scenario}-unknown',pair_id=None,kind='unknown',
                queried_index=absent,variant=variant,condition=conditions[absent],context=contexts[variant],
                question=question(entity,conditions[absent]),answer='UNKNOWN',foil=labels[variant],
                support=None,hard_negative=None))
        suite[split]=rows
    audit_suite(suite)
    return suite


def audit_suite(suite):
    if set(suite) != set(SPLITS):
        raise ValueError('Unexpected V4 split; no test split is permitted')
    train_audit = audit_v2({'train': suite['train']}, splits=('train',))
    names = {n for row in suite['train'] for n in row['entities']}
    ids = {r['id'] for r in suite['train']}
    scenario_ids = {r['scenario_id'] for r in suite['train']}
    result = dict(train=train_audit['splits']['train'])
    for split in SPLITS[1:]:
        rows=suite[split]
        scenarios=96 if split=='val_id' else 48
        if len(rows)!=scenarios*5:
            raise ValueError('Wrong validation row count')
        validate_rows(rows)
        groups=collections.defaultdict(list)
        cells=collections.Counter()
        for row in rows:
            if row['family'] not in CONDITIONS or row['difficulty'] not in LEVELS or row['split']!=split:
                raise ValueError('Invalid validation stratum')
            fi,li=list(CONDITIONS).index(row['family']),LEVELS.index(row['difficulty'])
            absent=row['absent_index']
            if type(absent)!=int or not 0<=absent<3:
                raise ValueError('Invalid absent condition')
            template=row['template_id']
            if template not in ((0,1) if split=='val_id' else (2,)) or row['queried_index'] not in range(3):
                raise ValueError('Wrong template/query stratum')
            if not row['scenario_id'].startswith(f'v4-s{SEED}-{split}-'):
                raise ValueError('Foreign scenario namespace')
            if row['id'] in ids:
                raise ValueError('Duplicate row namespace')
            ids.add(row['id'])
            entities=row['entities']
            if (len(entities)!=1+row['difficulty']//2 or len(set(entities))!=len(entities)
                or any(not re.fullmatch(r'Team [a-z]{12}',n) for n in entities) or entities[0]!=row['entity']):
                raise ValueError('Invalid opaque entity namespace')
            condition=CONDITIONS[row['family']][row['queried_index']]
            if row['condition']!=condition or row['question']!=question(row['entity'],condition):
                raise ValueError('Question metadata disagrees with actual input')
            facts=[]
            for line in row['context'].splitlines():
                match=re.fullmatch(_PATTERNS[template],line)
                if not match:
                    raise ValueError('Unparseable semantic record')
                fact=match.groupdict()
                if fact['entity'] not in entities or fact['value'] not in VALUES:
                    raise ValueError('Invalid semantic entity/value')
                facts.append((line,fact))
            if len(facts)!=row['difficulty']+2:
                raise ValueError('Wrong actual record count')
            absent_condition=CONDITIONS[row['family']][absent]
            if any(f['entity']==row['entity'] and f['condition']==absent_condition for _,f in facts):
                raise ValueError('Absent condition present in input')
            support=[(line,f) for line,f in facts if f['entity']==row['entity'] and f['condition']==condition]
            if row['kind']=='unknown':
                if (support or row['answer']!='UNKNOWN' or row['queried_index']!=absent or row['pair_id'] is not None
                    or row['support'] is not None or row['hard_negative'] is not None):
                    raise ValueError('Invalid actual UNKNOWN semantics')
            elif row['kind']=='swap':
                negative=[f for line,f in facts if line==row['hard_negative']]
                if (len(support)!=1 or support[0][0]!=row['support'] or support[0][1]['value']!=row['answer']
                    or row['foil'] not in VALUES or row['foil']==row['answer'] or row['pair_id'] is None
                    or len(negative)!=1 or negative[0]['entity']!=row['entity']
                    or negative[0]['condition']==condition or negative[0]['value']!=row['foil']):
                    raise ValueError('Invalid actual support/hard negative semantics')
            else:
                raise ValueError('Invalid row kind')
            groups[row['scenario_id']].append(row)
            cells[(row['family'],row['difficulty'],absent,template)]+=1
        expected={(f,d,a,t) for f in CONDITIONS for d in LEVELS for a in range(3)
            for t in ((0,1) if split=='val_id' else (2,))}
        if set(cells)!=expected or set(cells.values())!={5} or len(groups)!=scenarios:
            raise ValueError('Validation cells incomplete')
        variants=collections.Counter(r['variant'] for r in rows if r['kind']=='unknown')
        if variants!={0:scenarios//2,1:scenarios//2}:
            raise ValueError('UNKNOWN context variants not counterbalanced')
        conditional=collections.defaultdict(collections.Counter)
        labels=collections.Counter()
        for row in rows:
            conditional[(row['family'],row['difficulty'],row['template_id'],row['condition'])][row['kind']]+=1
            if row['kind']=='swap': labels[row['answer']]+=1
        if any(c['swap']!=4*c['unknown'] or not c['unknown'] for c in conditional.values()):
            raise ValueError('Conditional known:UNKNOWN balance violated')
        if labels!=collections.Counter(dict.fromkeys(VALUES,scenarios//2)):
            raise ValueError('Validation global label balance violated')
        for sid,group in groups.items():
            if sid in scenario_ids:
                raise ValueError('Scenario namespace leakage')
            scenario_ids.add(sid)
            if len(group)!=5 or sum(r['kind']=='unknown' for r in group)!=1:
                raise ValueError('Incomplete scenario')
            for key in ('entity','entities','family','difficulty','template_id','absent_index'):
                if any(r[key]!=group[0][key] for r in group):
                    raise ValueError('Shared scenario metadata mismatch')
            contexts={}
            pairs=collections.defaultdict(list)
            for row in group:
                if row['variant'] not in (0,1):
                    raise ValueError('Invalid context variant')
                if row['variant'] in contexts and contexts[row['variant']]!=row['context']:
                    raise ValueError('Changed context within variant')
                contexts[row['variant']]=row['context']
                if row['kind']=='swap': pairs[row['pair_id']].append(row)
            if set(contexts)!={0,1} or len(pairs)!=2:
                raise ValueError('Two complete swapped pairs required')
            for pair in pairs.values():
                if (len(pair)!=2 or {r['variant'] for r in pair}!={0,1}
                    or pair[0]['question']!=pair[1]['question'] or pair[0]['answer']!=pair[1]['foil']
                    or pair[1]['answer']!=pair[0]['foil']):
                    raise ValueError('Invalid swapped pair')
            known=[r for r in group if r['kind']=='swap']
            if {r['queried_index'] for r in known}!=set(range(3))-{group[0]['absent_index']}:
                raise ValueError('Missing present query')
            # Exact record positions and distractors must be preserved under swapping.
            first=contexts[0].splitlines(); second=contexts[1].splitlines()
            differences=0
            for a,b in zip(first,second):
                fa=re.fullmatch(_PATTERNS[group[0]['template_id']],a).groupdict()
                fb=re.fullmatch(_PATTERNS[group[0]['template_id']],b).groupdict()
                if (fa['entity'],fa['condition'])!=(fb['entity'],fb['condition']):
                    raise ValueError('Record positions changed across swap')
                if a!=b:
                    differences+=1
                    if fa['entity']!=group[0]['entity'] or fa['condition'] not in CONDITIONS[group[0]['family']]:
                        raise ValueError('Distractor changed across swap')
            if differences!=2:
                raise ValueError('Swap must change exactly two support records')
            current=set(group[0]['entities'])
            if current & names:
                raise ValueError('Cross-scenario/split entity leakage')
            names.update(current)
        result[split]=dict(rows=len(rows),scenarios=len(groups),pairs=scenarios*2,known=scenarios*4,unknown=scenarios,
            unknown_context_variant_counts=dict(variants),known_label_counts=dict(labels),
            conditional_counts=[dict(family=k[0],difficulty=k[1],template=k[2],condition=k[3],**dict(v)) for k,v in conditional.items()])
    return dict(passed=True,splits=result,reserved_test_policy='not inspected; fresh random namespaces only; test disjointness not empirically checked')


def overfit_rows(train):
    rows=[r for r in train if r['difficulty']==12]
    if len(rows)!=120 or len({r['scenario_id'] for r in rows})!=24:
        raise ValueError('Overfit subset must be all 24 d12 training scenarios')
    return rows


def freeze_suite(source_train, output, seed=SEED):
    source_train=Path(source_train)
    if sha256_file(source_train)!=SOURCE_TRAIN_SHA256:
        raise ValueError('Frozen original train SHA256 mismatch')
    suite=make_suite(read_jsonl(source_train),seed)
    output=Path(output); output.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(source_train,output/'train.jsonl')
    for split in SPLITS[1:]: write_jsonl(output/f'{split}.jsonl',suite[split])
    audit=audit_suite(suite)
    (output/'semantic_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    config=dict(seed=seed,protocol_version=4,source_train_sha256=SOURCE_TRAIN_SHA256,
        train_copy='unchanged bytes',validation_templates='val_id both0/1 per cell; val_ood2',new_test=False)
    manifest=dict(schema_version=4,config=config,config_sha256=canonical_hash(config),source_hashes=source_hashes(),
        splits={s:dict(rows=len(suite[s]),sha256=sha256_file(output/f'{s}.jsonl')) for s in SPLITS},
        semantic_audit_sha256=sha256_file(output/'semantic_audit.json'),reserved_test_policy=audit['reserved_test_policy'])
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    return manifest


def load_suite(output, splits=SPLITS):
    if tuple(splits)!=SPLITS:
        raise ValueError('Only exact V4 train/val_id/val_ood suite is allowed')
    output=Path(output)
    manifest=json.loads((output/'manifest.json').read_text(encoding='utf-8'))
    if (manifest['schema_version']!=4 or set(manifest['splits'])!=set(SPLITS)
        or canonical_hash(manifest['config'])!=manifest['config_sha256'] or manifest['source_hashes']!=source_hashes()
        or manifest['config']['source_train_sha256']!=SOURCE_TRAIN_SHA256):
        raise ValueError('V4 suite/source/config mismatch')
    if sha256_file(output/'semantic_audit.json')!=manifest['semantic_audit_sha256']:
        raise ValueError('Semantic audit hash mismatch')
    suite={}
    for split in SPLITS:
        path=output/f'{split}.jsonl'
        if sha256_file(path)!=manifest['splits'][split]['sha256']:
            raise ValueError('Split hash mismatch: '+split)
        suite[split]=read_jsonl(path)
    if sha256_file(output/'train.jsonl')!=SOURCE_TRAIN_SHA256:
        raise ValueError('Training bytes changed')
    audit_suite(suite)
    return suite


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-train',required=True); parser.add_argument('--output',required=True)
    parser.add_argument('--seed',type=int,default=SEED)
    args=parser.parse_args()
    print(json.dumps(freeze_suite(args.source_train,args.output,args.seed),indent=2))


if __name__=='__main__': main()
