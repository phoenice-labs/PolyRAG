"""
Phase 13 conftest: pre-import heavy scientific stack before pytest collection.

scipy 1.17.x + numpy 2.x has a known recursion issue in array_api_compat's
clone_module when `from numpy import *` is executed inside a partially
initialised pytest process. Pre-importing here (at conftest load time, which
happens before module-level fixture wiring) forces full numpy initialisation
before scipy's array_api_compat ever runs, eliminating the recursion.
"""
import numpy  # noqa: F401 — must come first to fully initialise numpy
import scipy  # noqa: F401 — loads array_api_compat while numpy is ready
from sentence_transformers import SentenceTransformer  # noqa: F401
