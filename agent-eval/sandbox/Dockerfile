# The sandbox of the run tool (plans/PLAN_AGENT_CAPACITY.md §5): Python, C++ and Rust, with no network at run time.
# Build: docker build -t agent-eval-sandbox:1 agent-eval/sandbox
FROM ubuntu:24.04
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      python3 python3-pytest python3-numpy g++ make cmake curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*
ENV RUSTUP_HOME=/opt/rustup CARGO_HOME=/opt/cargo
RUN curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain 1.90.0 --no-modify-path \
    && chmod -R a+rX /opt/rustup /opt/cargo
# At run time the root is read-only: cargo writes its cache to /tmp, and the binaries stay in /opt/cargo/bin.
ENV PATH=/opt/cargo/bin:$PATH CARGO_HOME=/tmp/cargo HOME=/tmp PYTHONDONTWRITEBYTECODE=1
WORKDIR /work
