"""MLIR callsite locations name the callee operation, not its caller."""

from triton_viz.clients.common.ttir_reader import SourceLoc, _LocTable


def _table(*extra):
    table = _LocTable()
    for line in (
        '#loc1 = loc("callee.py":17:9)',
        '#loc2 = loc("caller.py":41:5)',
        '#loc3 = loc("helper"(#loc1))',
        *extra,
    ):
        assert table.add(line)
    return table


def test_direct_callsite_resolves_callee_named_location():
    table = _table("#loc4 = loc(callsite(#loc3 at #loc2))")
    assert table.resolve("#loc4") == SourceLoc("callee.py", 17, 9)


def test_nested_callsites_keep_innermost_access_location():
    table = _table(
        "#loc4 = loc(callsite(#loc3 at #loc2))",
        "#loc5 = loc(callsite(#loc4 at #loc2))",
    )
    assert table.resolve("#loc5") == SourceLoc("callee.py", 17, 9)


def test_unknown_callee_does_not_fall_back_to_caller():
    table = _table("#loc4 = loc(callsite(#loc99 at #loc2))")
    assert table.resolve("#loc4") is None


def test_cyclic_callsites_fail_closed():
    table = _table(
        "#loc4 = loc(callsite(#loc5 at #loc2))",
        "#loc5 = loc(callsite(#loc4 at #loc2))",
    )
    assert table.resolve("#loc4") is None


def test_multi_source_fused_location_is_not_guessed():
    table = _table("#loc4 = loc(fused[#loc1, #loc2])")
    assert table.resolve("#loc4") is None
