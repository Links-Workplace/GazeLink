"""Compatibility name: this module lives in ``gazelink_core.gaze.head_features`` (ADR-0002).

The old ``gf_*`` name is bound to the SAME module object, so imports,
attribute patches and ``is`` comparisons behave exactly as before.
"""

import sys

from gazelink_core.gaze import head_features as _module

sys.modules[__name__] = _module
