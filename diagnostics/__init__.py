"""Read-only feasibility diagnostics for the distributed controller.

Nothing in this package changes the applied control: the auxiliary LPs below are
solved on copies of the QP data and their results are only logged.
"""
