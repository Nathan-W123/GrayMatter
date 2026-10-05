import os

# Parallelism comes from worker processes (see --workers); one BLAS thread per
# process avoids oversubscribing the CPU. Set these variables yourself to override.
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import sys  # noqa: E402

from .cli import main  # noqa: E402

sys.exit(main())
