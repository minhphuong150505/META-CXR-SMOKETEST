"""Local replacement for health_multimodal.common.device.get_module_device.

BioViL-T only needed this single one-line helper from hi-ml-multimodal, whose
2022-era dependency pins (torch==1.9.0, transformers==4.17.0, huggingface-hub==0.6.0)
are incompatible with the rest of this project. Vendoring it removes that dependency.
"""

import torch


def get_module_device(module: torch.nn.Module) -> torch.device:
    """Device of the module's first parameter (matches the upstream helper)."""
    return next(module.parameters()).device
