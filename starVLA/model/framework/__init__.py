"""
Framework factory utilities.
Automatically builds registered framework implementations
based on configuration.

Each framework module (e.g., M1.py, QwenFast.py) should register itself:
    from starVLA.model.framework.framework_registry import FRAMEWORK_REGISTRY

    @FRAMEWORK_REGISTRY.register("InternVLA-M1")
    def build_model_framework(config):
        return InternVLA_M1(config=config)
"""

import pkgutil
import importlib
from starVLA.model.tools import FRAMEWORK_REGISTRY

from starVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)
_AUTO_IMPORT_ERRORS = {}
_PKG_NAME = (__package__ or __name__).replace(".__init__", "")

try:
    pkg_path = __path__
except NameError:
    pkg_path = None

# Auto-import all framework submodules to trigger registration
if pkg_path is not None:
    for _, module_name, _ in pkgutil.iter_modules(pkg_path):
        try:
            importlib.import_module(f"{_PKG_NAME}.{module_name}")
        except Exception as e:
            _AUTO_IMPORT_ERRORS[module_name] = str(e)
            logger.warning(f"Failed to auto-import framework submodule {module_name}: {e}")
        
def build_framework(cfg):
    """
    Build a framework model from config.
    Args:
        cfg: Config object (OmegaConf / namespace) containing:
             cfg.framework.name: Identifier string (e.g. "InternVLA-M1")
    Returns:
        nn.Module: Instantiated framework model.
    """

    if not hasattr(cfg.framework, "name"): 
        cfg.framework.name = cfg.framework.framework_py  # Backward compatibility for legacy config yaml
        
    if cfg.framework.name == "QwenOFT":
        from starVLA.model.framework.QwenOFT import Qwenvl_OFT
        return Qwenvl_OFT(cfg)
    elif cfg.framework.name == "QwenFast":
        from starVLA.model.framework.QwenFast import Qwenvl_Fast
        return Qwenvl_Fast(cfg)

    # auto detect from registry
    framework_id = cfg.framework.name
    if framework_id not in FRAMEWORK_REGISTRY._registry:
        # Try lazy import by module name == framework id (e.g. MapAnythingLlava3DPI).
        try:
            importlib.import_module(f"{_PKG_NAME}.{framework_id}")
        except Exception as e:
            lazy_err = str(e)
        else:
            lazy_err = None
        if framework_id in FRAMEWORK_REGISTRY._registry:
            MODLE_CLASS = FRAMEWORK_REGISTRY[framework_id]
            return MODLE_CLASS(cfg)
        detail = ""
        if framework_id in _AUTO_IMPORT_ERRORS:
            detail = f" auto_import_error={_AUTO_IMPORT_ERRORS[framework_id]!r}"
        elif lazy_err is not None:
            detail = f" lazy_import_error={lazy_err!r}"
        registered = sorted(list(FRAMEWORK_REGISTRY._registry.keys()))
        raise NotImplementedError(
            f"Framework {cfg.framework.name} is not implemented. "
            f"Registered={registered}.{detail} "
            f"Plz, python yourframework_py to specify framework module."
        )
    
    MODLE_CLASS = FRAMEWORK_REGISTRY[framework_id]
    return MODLE_CLASS(cfg)

__all__ = ["build_framework", "FRAMEWORK_REGISTRY"]
