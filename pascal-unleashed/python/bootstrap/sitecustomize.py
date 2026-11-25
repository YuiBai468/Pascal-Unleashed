"""Loaded only by a Python application launched with the Pascal profile enabled."""

import os

if os.environ.get('PU_ENABLED')=='1':
    try:
        from pascal_unleashed.torch import activate_from_environment
        activate_from_environment()
    except (ImportError,OSError,RuntimeError,ValueError) as exc:
        import json
        from pathlib import Path
        telemetry=os.environ.get('PU_TELEMETRY')
        if telemetry:
            try:
                Path(telemetry).write_text(json.dumps(dict(loaded=False,pid=os.getpid(),error=str(exc))),encoding='utf-8')
            except OSError:
                pass
