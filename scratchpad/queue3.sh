set -uo pipefail
cd /c/Users/Ikazuchi/repos/pokeuraou
until grep -q "=== ladder done ===" "$Q2OUT"; do sleep 30; done
echo "=== ordering panel starting ==="
date
bash "$ORDER"
echo "=== ordering panel finished ==="
date
