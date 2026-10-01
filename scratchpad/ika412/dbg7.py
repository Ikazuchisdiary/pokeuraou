import json,sys,random,re,collections
from pokeuraou import slotswap
from pokeuraou.position import Position
from pokeuraou.regulation import load_regulation
from pokeuraou.damage import register_mega_stones
from pokeuraou.narrow import narrow
sys.path.insert(0,'C:/tmp/ika412/wt')
from tests._port import resolve_turn, Budget
rows=[json.loads(l) for l in open(sys.argv[2],encoding='utf-8')]
rows=[r for r in rows if r['variant']=='own']
random.Random(2).shuffle(rows)
which=(True,False)
def state(p):
    return {(si,m.species):(m.hp,m.fainted,m.status,tuple(sorted(m.boosts.items())),m.item,tuple(sorted(v.id for v in m.volatiles)),m.is_mega) for si,s in enumerate(p.sides) for m in s.pokemon}, tuple((s.active and [s.pokemon[i].species if i is not None else None for i in s.active]) is not None for s in p.sides)
def key(br): 
    st,_=state(br.position); return (round(br.probability,5),tuple(sorted(st.items())))
kinds=collections.Counter(); ex={}
for d in rows[:int(sys.argv[1])]:
    pos=Position.from_json(d['position'])
    reg=load_regulation(pos.format); register_mega_stones(reg)
    c=max(d['cells'],key=lambda z:abs(z[2]-z[3]))
    ours=narrow(reg,pos,0,limit=6).actions; theirs=narrow(reg,pos,1,limit=6).actions
    a=[x for x in ours if x.to_choice()==c[4]][0]; b=[x for x in theirs if x.to_choice()==c[5]][0]
    m=slotswap.swap_positions(pos,which)
    a2=slotswap.swap_action(a,True,False); b2=slotswap.swap_action(b,False,True)
    r1=resolve_turn(reg,pos,[a,b],budget=Budget.matrix())
    r2=resolve_turn(reg,m,[a2,b2],budget=Budget.matrix())
    A=sorted((key(x) for x in r1.branches),key=repr); B=sorted((key(x) for x in r2.branches),key=repr)
    if A==B:
        kind='same final states (suspended %d/%d)'%(len(r1.suspended),len(r2.suspended))
    else:
        sa={k:v for k,v in A}; 
        # which fields differ in the most likely branch
        a0=max(r1.branches,key=lambda z:z.probability); b0=max(r2.branches,key=lambda z:z.probability)
        sa0,_=state(a0.position); sb0,_=state(b0.position)
        diff=set()
        for k in sa0:
            for i,name in enumerate(('hp','fainted','status','boosts','item','volatiles','mega')):
                if sa0[k][i]!=sb0[k][i]: diff.add(name)
        kind='differ: %s; nbranch %s'%(sorted(diff), 'same' if len(A)==len(B) else '%d vs %d'%(len(A),len(B)))
    kinds[kind]+=1; ex.setdefault(kind,(c[4],c[5],d['turn'],[pos.sides[s].pokemon[i].species for s in (0,1) for i in pos.sides[s].active]))
for k,v in kinds.most_common(): print(v,k,ex[k])
