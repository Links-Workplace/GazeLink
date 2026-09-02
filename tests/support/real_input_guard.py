"""Guards that keep real operating-system input out of automated tests."""

from __future__ import annotations

import sys
from collections.abc import Iterable
from types import ModuleType

_REAL_INPUT_NAME_TOKENS = ("real", "windows", "win32", "sendinput", "osinput")
_INPUT_NAME_TOKENS = ("input", "cursor", "mouse")


def is_real_input_module_name(module_name: str) -> bool:
    """Return whether a GAZELINK module name denotes a real OS-input adapter."""

    if not module_name.startswith("gazelink."):
        return False
    normalized = module_name.replace("_", "").replace("-", "").lower()
    return any(token in normalized for token in _REAL_INPUT_NAME_TOKENS) and any(
        token in normalized for token in _INPUT_NAME_TOKENS
    )


def loaded_real_input_modules(
    modules: Iterable[str] | None = None,
) -> tuple[str, ...]:
    """List loaded real-input adapter modules in deterministic order."""

    candidates = sys.modules if modules is None else modules
    return tuple(sorted(name for name in candidates if is_real_input_module_name(name)))


def assert_no_real_input_modules_loaded(
    modules: Iterable[str] | None = None,
) -> None:
    """Fail clearly when an automated test has loaded a real-input adapter."""

    loaded = loaded_real_input_modules(modules)
    if loaded:
        names = ", ".join(loaded)
        raise AssertionError(f"real OS-input adapter loaded during tests: {names}")


def module_for_guard_test(name: str) -> ModuleType:
    """Create a detached module object for testing the guard itself."""

    return ModuleType(name)
