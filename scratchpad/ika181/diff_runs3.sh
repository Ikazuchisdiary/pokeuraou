#!/bin/bash
# IKA-181: divergences split by ally-target turns; the same leads with no ally target (off).
W=C:/Users/Ikazuchi/repos/pokeuraou/.claude/worktrees/agent-af06b00bddc87aa09
PY=C:/Users/Ikazuchi/repos/pokeuraou/.venv/Scripts/python.exe
POOL=C:/Users/Ikazuchi/repos/pokeuraou/data/pool/regmc-matchupweb.json
cd $W
export PYTHONPATH=$W/src POKEURAOU_RUST_NODE_BIN=C:/tmp/ika181/bin-fixed/pokeuraou-damage.exe
H="$PY C:/tmp/pokeuraou-machine/heavy.py --agent ika-181 --cores 2 --est-min 2"
$H --why "IKA-181 diff split any" -- $PY C:/tmp/ika181/diff_ally.py $POOL 300 182 all 0.5 --teams all C:/tmp/ika181/fx_any.json > C:/tmp/ika181/fixed_any2.out 2>&1
$H --why "IKA-181 diff off any" -- $PY C:/tmp/ika181/diff_ally.py $POOL 300 182 off 0.0 --teams all C:/tmp/ika181/fx_off_any.json > C:/tmp/ika181/fixed_off.out 2>&1
$H --why "IKA-181 diff split benefit" -- $PY C:/tmp/ika181/diff_ally.py $POOL 300 181 all 0.5 C:/tmp/ika181/fx_benefit.json > C:/tmp/ika181/fixed_benefit2.out 2>&1
$H --why "IKA-181 diff off benefit" -- $PY C:/tmp/ika181/diff_ally.py $POOL 300 181 off 0.0 C:/tmp/ika181/fx_off_benefit.json > C:/tmp/ika181/fixed_off_benefit.out 2>&1
for f in fixed_any2 fixed_off fixed_benefit2 fixed_off_benefit; do echo "== $f"; grep -E "^port|SPLIT|COUNT|refused a choice" C:/tmp/ika181/$f.out; done
