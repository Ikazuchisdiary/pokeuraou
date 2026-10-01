import json,sys
from pokeuraou import slotswap
from pokeuraou.position import Position
from pokeuraou.regulation import load_regulation
from pokeuraou.damage import register_mega_stones
from pokeuraou.narrow import narrow
sys.path.insert(0,'C:/tmp/ika412/wt')
from tests._port import resolve_turn, turn_expectation, Budget
from pokeuraou.payoff import HP_SHARE
rows=[json.loads(l) for l in open('C:/tmp/ika412/'+(sys.argv[3] if len(sys.argv)>3 else 'check3000.jsonl')+'',encoding='utf-8')]
rows.sort(key=lambda r:-r['worst'])
which_of={'own':(True,False),'foe':(False,True),'both':(True,True)}
n=int(sys.argv[1]); skip=int(sys.argv[2]) if len(sys.argv)>2 else 0
for d in rows[skip:skip+n]:
    pos=Position.from_json(d['position'])
    reg=load_regulation(pos.format); register_mega_stones(reg)
    which=which_of[d['variant']]
    c=max(d['cells'],key=lambda z:abs(z[2]-z[3]))
    ours=narrow(reg,pos,0,limit=6).actions; theirs=narrow(reg,pos,1,limit=6).actions
    a=[x for x in ours if x.to_choice()==c[4]][0]; b=[x for x in theirs if x.to_choice()==c[5]][0]
    m=slotswap.swap_positions(pos,which)
    a2=slotswap.swap_action(a,which[0],which[1]); b2=slotswap.swap_action(b,which[1],which[0])
    print('=====',d['variant'],round(d['worst'],4),'turn',d['turn'],c[4],'|',c[5],' base/swapped',round(c[2],4),round(c[3],4))
    print(' actives', [[(pos.sides[s].pokemon[i].species,pos.sides[s].pokemon[i].item, pos.sides[s].pokemon[i].status) for i in pos.sides[s].active] for s in (0,1)])
    for p,x,y in ((pos,a,b),(m,a2,b2)):
        r=resolve_turn(reg,p,[x,y],budget=Budget.matrix(),events=True)
        print('  value',round(turn_expectation(reg,r,HP_SHARE)[0],4),'branches',len(r.branches),'unmod',r.unmodelled)
        for br in sorted(r.branches,key=lambda z:-z.probability)[:2]:
            print('   ',round(br.probability,3),br.events[:14])
