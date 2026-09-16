"""GAZELINK live core (ADR-0002, TECHNICAL_SPEC v2.0 §4).

The core never imports the ``gf_*`` tools (recording, fitting, probes, CLI).
Tools import the core. Sub-packages:

``domain``       contracts, clock, reason codes
``tracking``     TrackingSource adapters (gazefollower, replay)
``gaze``         prediction, sample gates, pipeline, preflight
``calibration``  fitted model, schema, profiles
``interaction``  gestures, dwell, menu, scroll, keyboard, safety, controller, executor
``platform``     Windows input adapters and screen geometry
``ui``           display adapters and presentation
``app``          LiveSession, options, lifecycle, telemetry
"""
