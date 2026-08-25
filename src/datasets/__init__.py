"""Dataset builders: convert raw QA distributions into FATHOM's labeled jsonl.

One ``build_<dataset>.py`` module per source dataset (each runnable as
``python -m src.datasets.build_<dataset>``), sharing the bucket -> balance ->
sample -> write core in ``common``.
"""

__all__: list = []
