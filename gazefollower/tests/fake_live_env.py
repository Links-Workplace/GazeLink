"""A complete fake world for a live session: no camera, window or OS input.

Built as a ``LiveEnvironment`` and injected, instead of replacing module
globals. ``real_input`` is False, so a session that somehow reached a real
sender would also meet the real-input interlock. Every fake counts its calls in
``calls`` so a test can prove the fake was the thing used (a canary against
wiring that silently bypasses it).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

from gazelink_core.app.environment import LiveEnvironment
from gazelink_core.domain.clock import Clock, MonotonicClock

MONITOR = SimpleNamespace(origin=(0, 0), width_px=5120, height_px=1440, name="FAKE")
DESKTOP = {"x": 0, "y": 0, "width": 5120, "height": 1440}


def fake_model(path: Any = None) -> Any:
    return SimpleNamespace(
        schema=SimpleNamespace(columns=range(4), head_names=()),
        support_activation=lambda design: [1.0],
    )


class FakeSource:
    """A tracking source that publishes only what a test hands it."""

    def __init__(self, world: FakeWorld) -> None:
        self.world = world
        self.stopped = False
        self.closed = False
        self.started = False
        self.errors = 0
        self.last_error = None

    def subscribe(self, sink: Any) -> None:
        self.world.subscribers.append(sink)

    def start(self) -> None:
        self.started = True

    def stop(self) -> bool:
        self.stopped = True
        return True

    def publish(self, obs: Any) -> None:
        if self.stopped:
            return
        for sink in list(self.world.subscribers):
            sink(obs)

    def close(self) -> Any:
        self.stop()
        if self.closed:
            return self.world.shutdown_result
        self.closed = True
        self.world._count("shutdown_library")
        if self.world.events is not None:
            self.world.events.append("library shutdown")
        return self.world.shutdown_result


@dataclass
class FakeWorld:
    runner: Any
    display: Any
    clock: Clock = field(default_factory=MonotonicClock)
    escape: Callable[[], bool] = lambda: False
    key_down: Callable[[int], bool] = lambda vk: False
    foreground: Callable[[], int] = lambda: 4242
    title: Callable[[int], str] = lambda hwnd: f"window {hwnd}"
    warm_up_escape: bool = False
    display_error: BaseException | None = None
    shutdown_result: Any = None
    # Recorded effects
    sends: list[int] = field(default_factory=list)
    wheels: list[int] = field(default_factory=list)
    keystrokes: list[tuple[int, int, int]] = field(default_factory=list)
    moves: list[tuple[int, int]] = field(default_factory=list)
    subscribers: list[Any] = field(default_factory=list)
    calls: Counter = field(default_factory=Counter)
    events: list[str] | None = None
    # Optional hooks so a harness can stamp effects with its own time.
    on_send: Callable[[str, tuple], None] | None = None
    source: Any = None

    def _count(self, name: str) -> None:
        self.calls[name] += 1

    def environment(self) -> LiveEnvironment:
        world = self

        def open_source(profile: Any) -> Any:
            world._count("open_library")
            world.source = FakeSource(world)
            return world.source

        def make_display(*a: Any, **kw: Any) -> Any:
            world._count("make_display")
            if world.display_error is not None:
                raise world.display_error
            return world.display

        def make_runner(*a: Any, **kw: Any) -> Any:
            world._count("make_runner")
            return world.runner

        def effect(name: str, store: list, value: Any) -> None:
            world._count(name)
            store.append(value)
            if world.on_send is not None:
                world.on_send(name, value if isinstance(value, tuple) else (value,))

        return LiveEnvironment(
            clock=self.clock,
            open_source=open_source,
            warm_up=lambda display, seconds: (world._count("warm_up"), world.warm_up_escape)[1],
            make_runner=make_runner,
            load_model=lambda path: (world._count("load_model"), fake_model(path))[1],
            make_display=make_display,
            pick_monitor=lambda selector: (world._count("pick_monitor"), MONITOR)[1],
            virtual_desktop=lambda: (world._count("virtual_desktop"), DESKTOP)[1],
            ensure_dpi_aware=lambda: (
                world._count("ensure_dpi_aware"),
                (True, "PER_MONITOR (fake)"),
            )[1],
            check_screen=lambda profile: (
                world._count("check_screen"),
                SimpleNamespace(verified=True),
            )[1],
            escape_is_down=lambda: (world._count("escape_is_down"), world.escape())[1],
            key_is_down=lambda vk: (world._count("key_is_down"), world.key_down(vk))[1],
            click_sender=lambda flag: effect("button", world.sends, flag),
            wheel_sender=lambda delta: effect("wheel", world.wheels, delta),
            key_sender=lambda vk, scan, flags: effect("key", world.keystrokes, (vk, scan, flags)),
            foreground_window=lambda: (world._count("foreground_window"), world.foreground())[1],
            window_title=lambda hwnd: world.title(hwnd),
            set_pointer=lambda x, y: effect("move", world.moves, (x, y)),
            get_pointer=lambda: (0, 0),
            real_input=False,
        )
