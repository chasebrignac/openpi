"""Fixed-shape deployment helpers for the AWS pi0.5 reproduction.

Submodules are deliberately not imported here.  Manifest, calibration, and
numerical-audit tooling can therefore run on a CPU utility host without first
loading PyTorch and Transformers.
"""
