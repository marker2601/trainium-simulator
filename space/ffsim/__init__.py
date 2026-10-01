"""ffsim: FrontierForge Trainium run simulator (see ffsim/CONTRACT.md).

official_bpb = quality(recipe, steps, seed) + offset(eval shard)
steps        = charged_seconds / step_time(code, hardware, runtime)

Pure numpy + stdlib. No torch, no pandas.
"""
__version__ = "0.1.0"
