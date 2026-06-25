"""
Pytest session configuration.

Cap the BLAS/OpenMP thread pools to a single thread **before** NumPy (and hence
OpenBLAS/MKL) is imported.  PyCBA's analyses are built from many very small
(2x2 / 4x4) dense linear-algebra operations in tight loops (moving-load
traverses, the incremental nonlinear engine).  With the default multi-threaded
BLAS, each tiny operation spawns a thread pool whose synchronisation overhead
dwarfs the work, so on a many-core machine the suite can thrash dozens of cores
and appear to hang.  Forcing single-threaded BLAS makes the whole suite run in
seconds.  ``setdefault`` is used so an explicit environment override is
respected (e.g. to profile multi-threaded behaviour).
"""

import os

for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_var, "1")
