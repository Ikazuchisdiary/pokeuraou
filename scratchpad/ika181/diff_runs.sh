#!/bin/bash
# IKA-181: port vs Showdown on ally-target turns (benefit leads, any leads, the control).
W=C:/Users/Ikazuchi/repos/pokeuraou/.claude/worktrees/agent-af06b00bddc87aa09
PY=C:/Users/Ikazuchi/repos/pokeuraou/.venv/Scripts/python.exe
POOL=C:/Users/Ikazuchi/repos/pokeuraou/data/pool/regmc-matchupweb.json
cd $W
export PYTHONPATH=$W/src POKEURAOU_RUST_NODE_BIN=C:/tmp/ika181/bin/pokeuraou-damage.exe
H="$PY C:/tmp/pokeuraou-machine/heavy.py --agent ika-181 --cores 2 --est-min 15"
$H --why "IKA-181 diff ally benefit" -- $PY C:/tmp/ika181/diff_ally.py $POOL 300 181 all 0.5 > C:/tmp/ika181/diff_benefit.out 2>&1
$H --why "IKA-181 diff ally any lead" -- $PY C:/tmp/ika181/diff_ally.py $POOL 300 182 all 0.5 --teams all > C:/tmp/ika181/diff_any.out 2>&1
$H --why "IKA-181 diff ally control" -- $PY C:/tmp/ika181/diff_ally.py $POOL 60 181 all 0.5 --control > C:/tmp/ika181/diff_control.out 2>&1
echo done > C:/tmp/ika181/diff_runs.done
