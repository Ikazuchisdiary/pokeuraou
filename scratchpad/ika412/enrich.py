import json,sys,collections
sys.path.insert(0,'C:/tmp/ika412/wt/src'); sys.path.insert(0,'C:/tmp/ika412/wt/tools')
from pathlib import Path
from slot_swap_check import sample_positions, describe
from pokeuraou.position import Position
S=sample_positions(Path('C:/Users/Ikazuchi/repos/pokeuraou/data/selfplay-mc4'),3000,0)
bad=collections.defaultdict(dict)
for l in open('C:/tmp/ika412/check3000b.jsonl',encoding='utf-8'):
    d=json.loads(l); bad[d['variant']][(d['file'],d['game'],d['turn'])]=d['worst']
res=[]
for s in S:
    pos=Position.from_json(s['position'])
    if pos.ended or pos.request_state!='move': continue
    k=(s['file'],s['gameIndex'],s['turn'])
    f=describe(pos); keys={f"{a}:{x}" for a,xs in f.items() for x in xs}
    # extra: any pokemon with stall/lastmove/choicelock on the field, status
    res.append((keys,{v:bad[v].get(k,0.0) for v in ('own','foe','both')}))
N=len(res); print('positions',N)
def rate(sel,v,th):
    xs=[r for r in res if sel(r[0])]; 
    return len(xs), sum(1 for r in xs if r[1][v]>th)
for feat in ['volatile:stall','item:focussash','item:lifeorb','status:fnt','item:choicescarf','volatile:choicelock','volatile:twoturnmove','ability:friendguard','ability:fairyaura','ability:intimidate']:
    has=lambda k,f=feat:f in k; no=lambda k,f=feat:f not in k
    row=[]
    for v in ('own','both'):
        for th in (1e-6,2e-3):
            a=rate(has,v,th); b=rate(no,v,th)
            row.append(f"{v}>{th:g}: {a[1]}/{a[0]}={a[1]/max(a[0],1):.3f} vs {b[1]}/{b[0]}={b[1]/max(b[0],1):.3f}")
    print(feat,' | '.join(row))
any_status=lambda k:any(x.startswith('status:') and x!='status:' and x!='status:fnt' for x in k)
a=rate(any_status,'own',1e-6); b=rate(lambda k:not any_status(k),'own',1e-6); print('any status',a,b)
