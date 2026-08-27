# Changelog

All notable changes to NablaNet are documented here. The project follows
semantic versioning; pre-1.0 minor releases may contain source-compatible
breaking changes.

## 0.1.0 - 2026-08-27

- Established the `nablanet` C++ namespace, `<nablanet/...>` public headers,
  and `NablaNet::NablaNet` exported CMake target.
- Added verified exponential and hinge binary objectives with signed-target
  documentation, finite-difference curvature coverage, and deterministic
  training checks.
- Marked hinge loss as first-order only; it no longer advertises an invalid
  Hessian-vector product.
- Added CMake install-consumer coverage and Conan 2 static-package support
  with an executable test package.
- Added cross-platform CI and sanitizer validation.
