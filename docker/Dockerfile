FROM python:3.12-slim

ARG CUSTOMICS_VERSION

# to avoid permission issues with the numba cache
ENV NUMBA_CACHE_DIR=/tmp/numba_cache

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
      --index-url https://test.pypi.org/simple/ \
      --extra-index-url https://pypi.org/simple \
      "customics==${CUSTOMICS_VERSION}"

ENV PYTHONUNBUFFERED=1
