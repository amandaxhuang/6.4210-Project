"""Tests for the catch pipeline.  Run from the repo root:

    python -W default -m unittest discover -s tests -t .

The -W keeps unittest from replacing pydrake's own warning filter, which hides a
harmless "invalid value" warning from building symbolic expressions in numpy.
"""

import os

os.environ.setdefault("HEADLESS", "1")
