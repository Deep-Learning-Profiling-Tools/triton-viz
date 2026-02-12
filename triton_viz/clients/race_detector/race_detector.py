from ...core.client import Client
from ...core.callbacks import OpCallbacks, ForLoopCallbacks
from ...core.data import Op, Load, Store, AtomicRMW, AtomicCas
from .data import AccessType, MemoryAccess, RaceRecord, RaceType
from typing import Callable, Any, Optional
from collections import defaultdict

import numpy as np
from ...utils.traceback_utils import extract_user_frames


class RaceDetector(Client):
    NAME = "race_detector"

    def __init__(self):
        super().__init__()
        # All memory accesses collected across blocks: list of MemoryAccess
        self._accesses: list[MemoryAccess] = []

    def pre_run_callback(self, fn: Callable) -> bool:
        return True

    def post_run_callback(self, fn: Callable) -> bool:
        return True

    def pre_warmup_callback(self, jit_fn: Callable, *args, **kwargs) -> bool:
        return False

    def post_warmup_callback(self, jit_fn: Callable, ret: Any) -> None:
        pass

    def arg_callback(self, name, arg, arg_cvt):
        pass

    def grid_callback(self, grid: tuple[int, ...]):
        pass

    def grid_idx_callback(self, grid_idx: tuple[int, ...]):
        self.grid_idx = grid_idx

    def register_op_callback(self, op_type: type[Op]) -> OpCallbacks:
        @self.lock_fn
        def before_load(ptr, mask, keys):
            grid_idx = self.grid_idx
            if grid_idx is None:
                return
            offsets = ptr.data.flatten().astype(np.int64)
            masks = mask.data.flatten().astype(bool)
            self._accesses.append(
                MemoryAccess(
                    access_type=AccessType.LOAD,
                    ptr=0,
                    offsets=offsets,
                    masks=masks,
                    grid_idx=grid_idx,
                    call_path=extract_user_frames(),
                )
            )

        @self.lock_fn
        def before_store(ptr, mask, keys):
            grid_idx = self.grid_idx
            if grid_idx is None:
                return
            offsets = ptr.data.flatten().astype(np.int64)
            masks = mask.data.flatten().astype(bool)
            self._accesses.append(
                MemoryAccess(
                    access_type=AccessType.STORE,
                    ptr=0,
                    offsets=offsets,
                    masks=masks,
                    grid_idx=grid_idx,
                    call_path=extract_user_frames(),
                )
            )

        @self.lock_fn
        def before_atomic_rmw(rmwOp, ptr, val, mask, sem, scope):
            grid_idx = self.grid_idx
            if grid_idx is None:
                return
            offsets = ptr.data.flatten().astype(np.int64)
            masks = mask.data.flatten().astype(bool)
            self._accesses.append(
                MemoryAccess(
                    access_type=AccessType.ATOMIC,
                    ptr=0,
                    offsets=offsets,
                    masks=masks,
                    grid_idx=grid_idx,
                    call_path=extract_user_frames(),
                )
            )

        @self.lock_fn
        def before_atomic_cas(ptr, cmp, val, sem, scope):
            grid_idx = self.grid_idx
            if grid_idx is None:
                return
            offsets = ptr.data.flatten().astype(np.int64)
            # AtomicCas has no mask arg, treat all elements as active
            masks = np.ones(offsets.shape, dtype=bool)
            self._accesses.append(
                MemoryAccess(
                    access_type=AccessType.ATOMIC,
                    ptr=0,
                    offsets=offsets,
                    masks=masks,
                    grid_idx=grid_idx,
                    call_path=extract_user_frames(),
                )
            )

        if op_type is Load:
            return OpCallbacks(before_callback=before_load)
        elif op_type is Store:
            return OpCallbacks(before_callback=before_store)
        elif op_type is AtomicRMW:
            return OpCallbacks(before_callback=before_atomic_rmw)
        elif op_type is AtomicCas:
            return OpCallbacks(before_callback=before_atomic_cas)

        return OpCallbacks()

    def register_for_loop_callback(self) -> ForLoopCallbacks:
        return ForLoopCallbacks()

    def finalize(self) -> list:
        races = detect_races(self._accesses)
        self._accesses.clear()
        return races


def detect_races(accesses: list[MemoryAccess]) -> list[RaceRecord]:
    """Detect data races using an inverted index on byte addresses.

    A race occurs when:
    1. Two different blocks access the same byte address
    2. At least one access is a write (Store)
    3. The accesses are not both atomic
    """
    # Build inverted index: byte_address -> list of (access, element_index)
    addr_to_accesses: dict[int, list[MemoryAccess]] = defaultdict(list)

    for access in accesses:
        active_offsets = access.offsets[access.masks]
        unique_offsets = np.unique(active_offsets)
        for off in unique_offsets:
            addr_to_accesses[int(off)].append(access)

    races: list[RaceRecord] = []
    seen_pairs: set[tuple[tuple[int, ...], tuple[int, ...], RaceType, int]] = set()

    for addr, access_list in addr_to_accesses.items():
        if len(access_list) < 2:
            continue

        # Group by block
        block_accesses: dict[tuple[int, ...], list[MemoryAccess]] = defaultdict(list)
        for acc in access_list:
            block_accesses[acc.grid_idx].append(acc)

        blocks = list(block_accesses.keys())
        if len(blocks) < 2:
            continue

        for i in range(len(blocks)):
            for j in range(i + 1, len(blocks)):
                block_a, block_b = blocks[i], blocks[j]

                # Deduplicate block pairs
                pair_key = (min(block_a, block_b), max(block_a, block_b))

                accesses_a = block_accesses[block_a]
                accesses_b = block_accesses[block_b]

                for acc_a in accesses_a:
                    for acc_b in accesses_b:
                        # Both atomic -> no race
                        if (
                            acc_a.access_type == AccessType.ATOMIC
                            and acc_b.access_type == AccessType.ATOMIC
                        ):
                            continue

                        # Both loads -> no race
                        if (
                            acc_a.access_type == AccessType.LOAD
                            and acc_b.access_type == AccessType.LOAD
                        ):
                            continue

                        # Determine race type
                        race_type = _classify_race(acc_a, acc_b)
                        if race_type is None:
                            continue

                        # Use a dedup key including block pair + race type + address
                        dedup_key = (pair_key[0], pair_key[1], race_type, addr)
                        if dedup_key in seen_pairs:
                            continue
                        seen_pairs.add(dedup_key)

                        races.append(
                            RaceRecord(
                                race_type=race_type,
                                address_offset=addr,
                                access_a=acc_a,
                                access_b=acc_b,
                            )
                        )

    return races


def _classify_race(a: MemoryAccess, b: MemoryAccess) -> Optional[RaceType]:
    """Classify the race type between two accesses from different blocks."""
    ta = a.access_type
    tb = b.access_type

    # Both loads -> no race
    if ta == AccessType.LOAD and tb == AccessType.LOAD:
        return None

    # Both atomic -> no race
    if ta == AccessType.ATOMIC and tb == AccessType.ATOMIC:
        return None

    is_write_a = ta in (AccessType.STORE, AccessType.ATOMIC)
    is_write_b = tb in (AccessType.STORE, AccessType.ATOMIC)

    if is_write_a and is_write_b:
        return RaceType.WAW
    elif is_write_a and tb == AccessType.LOAD:
        return RaceType.RAW
    elif ta == AccessType.LOAD and is_write_b:
        return RaceType.RAW
    return None
