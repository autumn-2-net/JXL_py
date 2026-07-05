from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py as _build_py


class build_py(_build_py):
    def run(self):
        super().run()
        root = Path(__file__).resolve().parent
        package_dir = Path(self.build_lib) / "jxlpy"
        package_dir.mkdir(parents=True, exist_ok=True)
        names = {
            "jxlpy_native.dll",
            "libjxlpy_native.so",
            "libjxlpy_native.dylib",
        }
        copied = False
        for candidate in (root / "out" / "build").glob("**/*"):
            if candidate.name in names and candidate.is_file():
                self.copy_file(str(candidate), str(package_dir / candidate.name))
                copied = True
        if not copied:
            self.announce(
                "jxlpy_native was not found under out/build; wheel will be pure Python",
                level=2,
            )


setup(cmdclass={"build_py": build_py})
