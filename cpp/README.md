# Optional C++ Gaussian model backend

This branch contains an optional `pybind11` implementation of the
pixel-integrated Gaussian model and its analytical Jacobian. The Python
implementation remains the reference path and the default backend.

From the repository root:

```bash
python -m pip install -r requirements-cpp.txt
python cpp/build_extension.py
```

The extension is then selected through the estimator configuration:

```yaml
profiles:
  sensor:
    estimator:
      gaussian_backend: cpp
```

This preserves the existing SciPy bounded Trust Region Reflective optimizer
and Python IRLS loop. Only the model and Jacobian evaluation are compiled in
this first step, allowing direct numerical comparison with the reference
implementation before moving more of the optimization loop to C++.

The generated shared library and build directory are ignored by Git. The
extension must be rebuilt separately on each operating system and Python
environment, including the Jetson target.
