#!/bin/bash
# IKA-411: the impact count over gen-3 and gen-4 (with all-decision totals).
PY=C:/Users/Ikazuchi/repos/pokeuraou/.venv/Scripts/python.exe
D=C:/Users/Ikazuchi/repos/pokeuraou/data
export PYTHONPATH=C:/tmp/ika411/wt/src
$PY C:/tmp/ika411/impact.py $D/selfplay-mc3 C:/tmp/ika411/impact_mc3.json 8 > C:/tmp/ika411/impact_mc3.out 2> C:/tmp/ika411/impact_mc3.err
$PY C:/tmp/ika411/impact.py $D/selfplay-mc4 C:/tmp/ika411/impact_mc4.json 8 > C:/tmp/ika411/impact_mc4.out 2> C:/tmp/ika411/impact_mc4.err
