"""Build the optional pybind11 Gaussian model extension in place."""

from __future__ import annotations

from pathlib import Path
import sys

from setuptools import Extension, setup
import pybind11


ROOT = Path(__file__).resolve().parents[1]

if sys.platform == "win32":
    compile_args = ["/O2", "/std:c++17"]
else:
    compile_args = ["-O3", "-std=c++17"]


setup(
    name="subpixel-gaussian-cpp",
    ext_modules=[
        Extension(
            "estimators._gaussian_model",
            [str(ROOT / "cpp" / "gaussian_model.cpp")],
            include_dirs=[pybind11.get_include()],
            language="c++",
            extra_compile_args=compile_args,
        )
    ],
    script_args=["build_ext", "--inplace"],
)
