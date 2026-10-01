#!/bin/bash
# IKA-181: related tests on the fixed port.
W=C:/Users/Ikazuchi/repos/pokeuraou/.claude/worktrees/agent-af06b00bddc87aa09
PY=C:/Users/Ikazuchi/repos/pokeuraou/.venv/Scripts/python.exe
cd $W
export PYTHONPATH=$W/src POKEURAOU_RUST_NODE_BIN=C:/tmp/ika181/bin-fixed/pokeuraou-damage.exe
T="tests/test_after_move_oracle.py tests/test_ally_targets.py tests/test_charge_target.py tests/test_choice_lock.py tests/test_curse_target.py tests/test_dead_field_moves_oracle.py tests/test_decided_leaf.py tests/test_disguise_afterhit.py tests/test_disguise_order.py tests/test_endgame_exact.py tests/test_feint_order.py tests/test_outrage_lock.py tests/test_portserved.py tests/test_random_target.py tests/test_resolve.py tests/test_salt_cure_champions.py tests/test_speed.py tests/test_substitute.py tests/test_trap_immunities.py tests/test_trap_sources.py tests/test_actions.py tests/test_port_menus.py tests/test_poolplay.py tests/test_selfplay.py tests/test_binary_scope.py tests/test_menu_ownership.py tests/test_qhead.py tests/test_provenance.py tests/test_line_endings.py tests/test_no_machine_specific_paths.py"
$PY C:/tmp/pokeuraou-machine/heavy.py --agent ika-181 --cores 6 --est-min 6 --why "IKA-181 related tests" -- $PY -m pytest $T -n 6 -p no:cacheprovider -q -rfEs > C:/tmp/ika181/tests1.log 2>&1
tail -30 C:/tmp/ika181/tests1.log
