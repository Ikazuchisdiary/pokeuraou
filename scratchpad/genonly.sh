set -uo pipefail
cd /c/Users/Ikazuchi/repos/pokeuraou
# Generation is pulled to the front because it is the ONE thing the three menu defects do
# not touch: self-play gives both sides the same evaluator object, so `leaves[1] is
# leaves[0]`, the rebuild branch never fires and the replacement node's `leaves[0]` is
# also `leaves[1]`. Three hours of machine while the matches are repaired.
echo "=== GEN: 24,000 hidden-bench games, leaf gen11L, seed 2001, T=0.5 ==="
date
uv run --group learn python -u tools/generate_queue.py \
  --out data/selfplay-hidden2 --games 24000 --seed 2001 --served --limit 24 \
  --value data/models/value-gen11L.pt --hide-bench \
  -- --rank-leaf --explore-temperature 0.5 --explore-epsilon 0.25 2>&1 | tail -10
echo "24000 hidden games, leaf+book value-gen11L, seed 2001, --rank-leaf, e=0.25 T=0.5" \
  > data/selfplay-hidden2/DONE
echo "=== GEN done ==="
date
