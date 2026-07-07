"""Mechanism shared between client packages.

Modules here are mechanism-only (parsers, encoders, plumbing) and carry no
client policy: what to do with a parse result, a flagged construct, or an
unsupported kernel is decided by the client that uses it (sanitizer, race
detector, ...).
"""
