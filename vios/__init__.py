"""
vios — the v2 system, built as puzzle pieces.

Each subpackage is a plane that can run alone, on any machine, with no
knowledge of the others beyond a shared ledger:

    vios.capture   permalinks -> permanent Telegram storage (this phase)
    vios.process   models -> claims, rotated through a VRAM budget
    vios.publish   claims -> an immutable artifact bundle the website reads

The contract between them is the ledger and the bundle, never a live call.
Nothing here imports the v1 modules at the top level; where it needs one
(config, tg_transport, logger) it imports it explicitly so a plane stays
runnable from a bare Kaggle cell.
"""

__all__ = ["capture"]
