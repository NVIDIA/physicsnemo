# Container dependency wheels

The release pipeline copies optional architecture-specific wheels into this
directory before building the container. Wheel files are intentionally ignored
by Git. When a wheel is absent, `docker/install-container-dependencies.sh` uses
the package's configured fallback.

This directory is tracked because the Dockerfile bind mount requires its source
to exist, including for builds that do not provide custom wheels.
