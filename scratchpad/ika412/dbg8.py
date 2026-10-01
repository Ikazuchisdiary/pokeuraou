import json,sys
exec(open('C:/tmp/ika412/dbg7.py').read().split('kinds=collections.Counter()')[0].replace("sys.argv[2]","'C:/tmp/ika412/check3000b.jsonl'"))
want=sys.argv[1]
for d in rows:
    pos=Position.from_json(d['position'])
    names=[pos.sides[s].pokemon[i].species for s in (0,1) for i in pos.sides[s].active]
    if names==sys.argv[1].split(',') and d['turn']==int(sys.argv[2]) and d['cells'][0][4]==sys.argv[3]: break
reg=load_regulation(pos.format); register_mega_stones(reg)
c=max(d['cells'],key=lambda z:abs(z[2]-z[3]))
ours=narrow(reg,pos,0,limit=6).actions; theirs=narrow(reg,pos,1,limit=6).actions
a=[x for x in ours if x.to_choice()==c[4]][0]; b=[x for x in theirs if x.to_choice()==c[5]][0]
m=slotswap.swap_positions(pos,which)
r1=resolve_turn(reg,pos,[a,b],budget=Budget.matrix(),events=True)
r2=resolve_turn(reg,m,[slotswap.swap_action(a,True,False),slotswap.swap_action(b,False,True)],budget=Budget.matrix(),events=True)
print(c[4],c[5],len(r1.branches),len(r2.branches))
A={}; 
for x in r1.branches: A.setdefault(key(x)[1],[]).append(x.probability)
B={}
for x in r2.branches: B.setdefault(key(x)[1],[]).append(x.probability)
for k in set(A)|set(B):
    pa=sum(A.get(k,[0])); pb=sum(B.get(k,[0]))
    if abs(pa-pb)>1e-9:
        print('prob',round(pa,6),round(pb,6),[ (kk,v[0],v[1]) for kk,v in k if kk in [(0,'gholdengo'),(1,'indeedee'),(1,'sneasler'),(0,'garchomp')]])
print('---- events of branches, orig')
for x in sorted(r1.branches,key=lambda z:-z.probability)[:16]: print(round(x.probability,5),x.events)
print('---- swapped')
for x in sorted(r2.branches,key=lambda z:-z.probability)[:16]: print(round(x.probability,5),x.events)
