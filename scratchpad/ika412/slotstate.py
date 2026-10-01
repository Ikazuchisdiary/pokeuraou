"""IKA-412: per-mon state that Showdown keeps per Pokemon -- is it per Pokemon in the port too?
One mon on the left carries the state; the right carries none. Swap -> the answer must not move."""
import json,sys,random,collections
sys.path.insert(0,'C:/tmp/ika412/wt/src'); sys.path.insert(0,'C:/tmp/ika412/wt/tools')
from pathlib import Path
import numpy as np
from slot_swap_check import sample_positions
from pokeuraou import slotswap
from pokeuraou.position import Position, Effect
from pokeuraou.regulation import load_regulation
from pokeuraou.damage import register_mega_stones
from pokeuraou.narrow import narrow
from pokeuraou.budget import Budget
from pokeuraou.payoff import HP_SHARE
from pokeuraou.port import batched_payoff
S=sample_positions(Path('C:/Users/Ikazuchi/repos/pokeuraou/data/selfplay-mc4'),400,7)
def first_move(m): return m.moves[0].id
def m_stall(m): m.volatiles=[v for v in m.volatiles if v.id!='stall']+[Effect(id='stall',duration=1,counter=3)]
def m_choice(m): m.item='choicescarf'; m.volatiles.append(Effect(id='choicelock',move=first_move(m))); m.locked_move=first_move(m)
def m_last(m): m.last_move=m.moves[0].id; m.move_last_turn_failed=True
def m_hits(m): m.times_attacked=4; m.active_move_actions=2
def m_new(m): m.newly_switched=True; m.active_move_actions=0
def m_sash(m): m.item='focussash'; m.hp=max(1,m.hp//2)
def m_status(m): m.status='brn'
def m_boost(m): m.boosts={'atk':2,'spe':-1}
def m_taunt(m): m.volatiles.append(Effect(id='taunt',duration=2))
def m_protectmid(m): m.volatiles=[v for v in m.volatiles if v.id!='stall']+[Effect(id='stall',duration=1,counter=9)]
MUT={'stall':m_stall,'choice-lock':m_choice,'last move+failed':m_last,'times attacked/active move actions':m_hits,'newly switched':m_new,'sash+half hp':m_sash,'burn':m_status,'boosts':m_boost,'taunt':m_taunt}
reg=None
out={k:collections.Counter() for k in MUT}
budget=Budget.matrix()
def mat(pos,ours,theirs): return batched_payoff(reg,pos,ours,theirs,HP_SHARE.batch,budget=budget)[0]
n=0
for s in S:
    pos=Position.from_json(s['position'])
    if pos.ended or pos.request_state!='move': continue
    if reg is None:
        reg=load_regulation(pos.format); register_mega_stones(reg)
    # only turns where both of our actives are alive and present
    a_act=[i for i in pos.sides[0].active if i is not None]
    if len(a_act)<2: continue
    if any(pos.sides[0].pokemon[i].fainted for i in a_act): continue
    n+=1
    for name,f in MUT.items():
        p=pos.copy(); f(p.sides[0].pokemon[p.sides[0].active[0]])   # left mon of side 0 only
        ours=narrow(reg,p,0,limit=6).actions; theirs=narrow(reg,p,1,limit=6).actions
        if not ours or not theirs: continue
        base=mat(p,ours,theirs)
        pure=pos.copy(); ours0=narrow(reg,pure,0,limit=6).actions; th0=narrow(reg,pure,1,limit=6).actions
        # effect size: same actions where legal (just compare value spread between mutated and unmutated on the shared menu)
        shared=[a.to_choice() for a in ours]; 
        for var,which in (('own',(True,False)),('foe',(False,True))):
            mv=slotswap.swap_positions(p,which)
            mo=[slotswap.swap_action(a,*which) for a in ours]; mt=[slotswap.swap_action(a,which[1],which[0]) for a in theirs]
            d=np.abs(mat(mv,mo,mt)-base)
            out[name]['pos']+=1; out[name][var+'>1e-6']+=int(d.max()>1e-6); out[name][var+'>2e-3']+=int(d.max()>2e-3)
        # effect-size control: mutated vs unmutated matrix on the identical menus, if the menus coincide
        if [a.to_choice() for a in ours0]==[a.to_choice() for a in ours] and [a.to_choice() for a in th0]==[a.to_choice() for a in theirs]:
            e=np.abs(mat(pure,ours0,th0)-base); out[name]['effect_n']+=1; out[name]['effect>2e-3']+=int(e.max()>2e-3)
print('positions',n)
for k,c in out.items(): print(f"{k:36s} {dict(c)}")
