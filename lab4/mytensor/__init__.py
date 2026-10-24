"""The public Python API; all arithmetic runs in the C++/CUDA extension."""
import os as _os
from pathlib import Path as _Path

# Windows CUDA 13 keeps cuBLAS DLLs in bin/x64. Retain the directory handles.
_dll_handles = []
if _os.name == "nt":
    _cuda = _os.environ.get("CUDA_PATH")
    if _cuda:
        for _directory in (_Path(_cuda) / "bin", _Path(_cuda) / "bin/x64"):
            if _directory.is_dir():
                _dll_handles.append(_os.add_dll_directory(str(_directory)))

from ._mytensor import *  # noqa: F401,F403,E402
from ._mytensor import __version__  # noqa: F401,E402
