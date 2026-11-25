"""Cross field identity, serialized field position and first-two-team role."""
from collections import OrderedDict
from copy import deepcopy
from .v13_data import FIELDS

def render(facts,background,field_order):
    groups=OrderedDict()
    for fact in facts:
        team,mode,field=fact['key'];groups.setdefault((team,mode),{})[field]=fact['value']
    lines=['Record entries apply only to their named team and operating mode.']
    if background:lines.append('Background: '+background)
    for (team,mode),fields in groups.items():
        lines.append('Team '+team+', '+mode+' mode: '+', '.join(f+' '+fields[f] for f in field_order)+'.')
    return '\n'.join(lines)

def counterbalanced_variants(unit):
    if unit['family']!='probe' or unit['level']!=4:raise ValueError('Fixed high-interference diagnostic')
    background=[s[len('Background: '):] for s in unit['contexts'][0].splitlines() if s.startswith('Background: ')]
    background=background[0] if background else ''
    teams=list(unit['entities']);modes=list(unit['modes']);output=[]
    for semantic in range(3):
        renamed=FIELDS[semantic:]+FIELDS[:semantic];field_map=dict(zip(FIELDS,renamed))
        for layout in range(3):
            field_order=FIELDS[layout:]+FIELDS[:layout]
            for exchange in range(2):
                team_map={t:t for t in teams}
                if exchange:team_map[teams[0]],team_map[teams[1]]=teams[1],teams[0]
                u=deepcopy(unit);u['base_unit_id']=unit['id']
                u['id']=unit['id']+f'-role-s{semantic}-p{layout}-e{exchange}'
                u.update(semantic_rotation=semantic,layout_rotation=layout,target_team_position=exchange,
                    target_field_position=field_order.index(field_map[FIELDS[0]]),rendered_field_order=list(field_order))
                u['query_keys']=[[team_map[t],m,field_map[f]] for t,m,f in unit['query_keys']]
                u['queries']=[f'For team {t} in {m} mode, what is the {f} code?' for t,m,f in u['query_keys']]
                variants=[]
                for i,facts in enumerate(unit['facts']):
                    values=[dict(key=[team_map[t],m,field_map[f]],value=v['value']) for v in facts for t,m,f in [v['key']]]
                    values.sort(key=lambda v:(teams.index(v['key'][0]),modes.index(v['key'][1]),field_order.index(v['key'][2])),reverse=i==3)
                    variants.append(values)
                u['facts']=variants;u['contexts']=[render(v,background,field_order) for v in variants]
                assert all([dict((tuple(f['key']),f['value']) for f in facts)[tuple(k)] for k in u['query_keys']]==answers for facts,answers in zip(u['facts'],u['answers']))
                output.append(u)
    return output
