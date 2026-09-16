"""Compatibility name: this module lives in ``gazelink_core.platform.display`` (ADR-0002).

The old ``gf_*`` name is bound to the SAME module object, so imports,
attribute patches and ``is`` comparisons behave exactly as before.
"""

import sys

from gazelink_core.platform import display as _module

if __name__ == "__main__":
    print(_module.describe_monitors())
    raise SystemExit(0)
sys.modules[__name__] = _module
