# The agent, with no runtime dependencies to install.
#
# python:slim and a COPY, because the package is standard-library only. The
# frontend is pre-built into web/ and committed, so there is no node stage here
# either -- `make ui-build` is a developer step, not a deployment one.
FROM python:3.12-slim

# Not root. The agent writes files and runs generated code; the sandbox is
# containment rather than a security boundary (DESIGN 5.4), so the container's
# own user is one of the layers that has to do real work.
RUN useradd --create-home --uid 10001 distil
WORKDIR /app

COPY distil/ /app/distil/
COPY web/ /app/web/
COPY tests/ /app/tests/
COPY README.md DESIGN.md /app/

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DISTIL_HOME=/data

RUN mkdir -p /data && chown -R distil:distil /data /app
USER distil

EXPOSE 8765
# 0.0.0.0 inside the container, published to loopback by compose. The warning the
# server prints on a non-loopback bind is correct and deliberate: what makes this
# safe is the port mapping, not the bind address.
CMD ["python3", "-m", "distil.cli", "ui", "--bind", "0.0.0.0", "--port", "8765"]
