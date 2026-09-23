"""The server asks CUDA to sleep while it waits (IKA-106), and says what it got.

A fake driver stands in for nvcuda: the real one would build a context in the test
process, and the thing under test is which calls are made in which order, with what.
Whether the flag then takes is read back by the server on its own startup lines
(`cuda waits: blocking sync`), which is where a generation run's log shows it.
"""

from __future__ import annotations

import pytest

from pokeuraou.inference import BLOCKING_SYNC, scheduling, wait_by_sleeping


class FakeDriver:
    def __init__(self, devices: int = 2, fail: str | None = None, flags: int = 0,
                 active: int = 1) -> None:
        self.devices = devices
        self.fail = fail
        self.flags = flags
        self.active = active
        self.calls: list[tuple] = []

    def _rc(self, name: str) -> int:
        return 999 if self.fail == name else 0

    def cuInit(self, flags):  # noqa: N802
        self.calls.append(("init", flags))
        return self._rc("init")

    def cuDeviceGetCount(self, out):  # noqa: N802
        out._obj.value = self.devices
        return self._rc("count")

    def cuDeviceGet(self, out, index):  # noqa: N802
        out._obj.value = 100 + index
        return self._rc("get")

    def cuDevicePrimaryCtxSetFlags_v2(self, device, flags):  # noqa: N802
        self.calls.append(("set", device.value, flags))
        return self._rc("set")

    def cuDevicePrimaryCtxGetState(self, device, flags, active):  # noqa: N802
        flags._obj.value = self.flags
        active._obj.value = self.active
        return self._rc("state")


def test_every_device_is_set_to_blocking_sync_after_init() -> None:
    driver = FakeDriver(devices=2)
    assert wait_by_sleeping(driver) == 2
    assert driver.calls == [("init", 0), ("set", 100, BLOCKING_SYNC), ("set", 101, BLOCKING_SYNC)]
    assert BLOCKING_SYNC == 0x04  # CU_CTX_SCHED_BLOCKING_SYNC in cuda.h


@pytest.mark.parametrize("step", ["init", "count", "get", "set"])
def test_a_failed_call_is_an_error_not_a_quiet_spin(step: str) -> None:
    with pytest.raises(RuntimeError):
        wait_by_sleeping(FakeDriver(fail=step))


@pytest.mark.parametrize(
    ("flags", "active", "reads"),
    [
        (0x04, 1, "blocking sync"),
        (0x00, 1, "auto"),
        (0x01, 1, "spin"),
        (0x02, 1, "yield"),
        (0x0C, 1, "blocking sync"),  # other bits (map host) do not hide the mode
        (0x04, 0, "blocking sync (context not yet created)"),
    ],
)
def test_the_mode_is_read_back_from_the_low_bits(flags: int, active: int, reads: str) -> None:
    assert scheduling(0, FakeDriver(flags=flags, active=active)) == reads


def test_an_unreadable_state_says_so() -> None:
    assert scheduling(0, FakeDriver(fail="state")) == "unknown"
