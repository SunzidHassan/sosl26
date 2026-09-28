import os
import sys

# The sOSL_* modules (shared with the AI2-THOR code base) import each other
# with flat imports, e.g. `from sOSL_utils import world_to_grid`. Put this
# package directory on sys.path so those imports resolve unchanged when the
# package is installed with colcon.
_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
if _PKG_DIR not in sys.path:
    sys.path.insert(0, _PKG_DIR)
