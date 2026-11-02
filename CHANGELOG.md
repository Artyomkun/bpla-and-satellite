# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-02-12

### Added

- Complete rewrite in Julia 1.9+
- Custom Julia HTTP server for inference
- Production-ready Docker image
- Kubernetes deployment with Helm charts
- Circuit breaker pattern implementation
- Rate limiting middleware
- Prometheus metrics collection
- Structured JSON logging
- Comprehensive test suite
- CI/CD pipeline with GitHub Actions
- Security scanning with Trivy
- Multi-stage Docker build
- Non-root container execution
- Pre-commit hooks with JuliaFormatter

### Changed

- Migrated from Python to Julia 1.9+
- From FastAPI to native Julia HTTP
- From pip to Pkg.jl
- From Ruff to JuliaFormatter

### Fixed

- Memory leak in inference engine
- Race condition in circuit breaker

### Security

- Non-root container execution
- Trivy vulnerability scanning
- Rate limiting implementation
