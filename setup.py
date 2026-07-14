import os
import sys
from pathlib import Path

from setuptools import Distribution, setup
from setuptools.command.build_py import build_py as _build_py
from wheel.bdist_wheel import bdist_wheel as _bdist_wheel


def _native_name() -> str:
    if sys.platform == "win32":
        return "jxlpy_native.dll"
    if sys.platform == "darwin":
        return "libjxlpy_native.dylib"
    return "libjxlpy_native.so"


def _find_native(root: Path) -> Path:
    explicit = os.environ.get("JXLPY_NATIVE_LIB")
    if explicit:
        candidate = Path(explicit).expanduser().resolve()
        if not candidate.is_file():
            raise RuntimeError(f"JXLPY_NATIVE_LIB does not exist: {candidate}")
        if candidate.name != _native_name():
            raise RuntimeError(
                f"JXLPY_NATIVE_LIB must name {_native_name()}, got {candidate.name}"
            )
        return candidate

    candidates = sorted(
        path.resolve()
        for path in (root / "out" / "build").glob(f"**/{_native_name()}")
        if path.is_file()
    )
    if not candidates:
        raise RuntimeError(
            "native library not found under out/build; build jxlpy_native or "
            "set JXLPY_NATIVE_LIB before creating a wheel"
        )
    if len(candidates) != 1:
        listing = "\n  ".join(str(path) for path in candidates)
        raise RuntimeError(
            "multiple native libraries found; set JXLPY_NATIVE_LIB to the "
            f"one intended for this wheel:\n  {listing}"
        )
    return candidates[0]


class build_py(_build_py):
    def run(self):
        super().run()
        root = Path(__file__).resolve().parent
        package_dir = Path(self.build_lib) / "jxlpy"
        package_dir.mkdir(parents=True, exist_ok=True)
        candidate = _find_native(root)
        self.copy_file(str(candidate), str(package_dir / candidate.name))


class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return True


class bdist_wheel(_bdist_wheel):
    def finalize_options(self):
        super().finalize_options()
        self.root_is_pure = False

    def get_tag(self):
        _, _, platform_tag = super().get_tag()
        return "py3", "none", platform_tag


setup(
    cmdclass={"build_py": build_py, "bdist_wheel": bdist_wheel},
    distclass=BinaryDistribution,
)
