"""Happens-before reasoning for race_detector.

HB logic intentionally deferred until after Step 1. Step 0 keeps this file
as a visible placeholder so the directory layout is stable, but nothing in
Step 0/Step 1 should import from here. The real ``HBSolver`` lands with
the Step-2 PR.
"""
