"""IKA-390 stage A: the ladder's workers ask the inference servers in turn."""
from pokeuraou import ladder


def test_workers_take_the_servers_in_turn() -> None:
    args = ("served", "127.0.0.1:1,127.0.0.1:2", "value", False, "q.pt", "q")
    got = [ladder._worker_args(args, i)[1] for i in range(5)]
    assert got == ["127.0.0.1:1", "127.0.0.1:2", "127.0.0.1:1", "127.0.0.1:2", "127.0.0.1:1"]
    # nothing but the address changes
    assert ladder._worker_args(args, 1)[2:] == args[2:]
    assert ladder._worker_args(args, 1)[0] == "served"


def test_one_server_or_another_kind_of_leaf_is_unchanged() -> None:
    one = ("served", "127.0.0.1:1", "value", False, "q.pt", "q")
    assert ladder._worker_args(one, 3) == one
    local = ("local", ["a,b"], "cpu", False, 0.8, "q.pt")
    assert ladder._worker_args(local, 3) == local
    assert ladder._worker_args(("hp-share",), 0) == ("hp-share",)
