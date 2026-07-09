"""S5 evaluation harness (plan Part III S5; protocol notes in TODO.md).

Driverless: TTIR is host-compiled, the compiled race detector is driven
synthetically, and the C2 replay / dynamic-mode comparison run on the CPU
interpreter. One subprocess per kernel; every row lands in a JSONL file.
"""
