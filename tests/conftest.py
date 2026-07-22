"""CPU-only test bootstrap.

`model/lavis/__init__.py` eagerly imports the dataset builders, which drag in the
whole training stack (torchvision, transformers, opencv, decord). The tests here
only touch `optims.py` and `base_task.py`, so register `model` and `model.lavis`
as path-only packages first: Python then resolves the submodules without ever
executing that `__init__`.

`dist_utils` also imports `timm.models.hub` for a checkpoint-download helper that
no test calls, so stub it when timm is absent.
"""

import sys
import types
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

for _name in ("model", "model.lavis"):
    if _name not in sys.modules:
        _pkg = types.ModuleType(_name)
        _pkg.__path__ = [str(_REPO_ROOT / Path(*_name.split(".")))]
        sys.modules[_name] = _pkg

try:  # pragma: no cover - exercised only on machines without timm
    import timm.models.hub  # noqa: F401
except ImportError:
    _timm = types.ModuleType("timm")
    _models = types.ModuleType("timm.models")
    _hub = types.ModuleType("timm.models.hub")
    _hub.get_cache_dir = lambda *args, **kwargs: "/tmp"
    _hub.download_cached_file = lambda *args, **kwargs: None
    _models.hub = _hub
    _timm.models = _models
    sys.modules.update({"timm": _timm, "timm.models": _models, "timm.models.hub": _hub})
