FROM mcr.microsoft.com/devcontainers/python:1-3.12-bookworm

WORKDIR /workspace

# Installing during the image build makes every device start from the same
# declared Python toolchain instead of relying on packages from the host.
COPY pyproject.toml README.md ./
COPY src ./src
RUN python -m pip install --no-cache-dir -e ".[dev]"

