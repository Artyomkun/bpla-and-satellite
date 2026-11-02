FROM registry.astralinux.ru/library/astra/ubi18@sha256:5ab19eade0294930e80e6564faede30cf35d142e660f80e31355e9b991deea1b

# Install build dependencies
RUN apt-get update && apt-get install -y \
    curl \
    gnupg \
    gcc \
    g++ \
    cmake \
    libc-dev \
    libssl-dev \
    libcurl4-openssl-dev \
    pkg-config \
    libopenblas0 \
    libomp5 \
    libcurl4 \
    libssl3 \
    tini \
    curl ca-certificates

ARG RUST_VERSION=1.97.1
ARG RUST_SHA256=ef5dd6a24cd8c2196fbdbbcf2f6c043fb95d52b99da030c6a7e02e1c94be90df

RUN set -eux; \
    curl --proto '=https' --tlsv1.3 -sSf -O "https://rust-lang.org{RUST_VERSION}-x86_64-unknown-linux-gnu.tar.xz"; \
    echo "${RUST_SHA256} *rust-${RUST_VERSION}-x86_64-unknown-linux-gnu.tar.xz" | sha256sum -c -; \
    tar -xf "rust-${RUST_VERSION}-x86_64-unknown-linux-gnu.tar.xz"; \
    ./rust-${RUST_VERSION}-x86_64-unknown-linux-gnu/install.sh; \
    rustc --version && cargo --version

ENV CARGO_HOME=/usr/local/cargo
ENV RUST_HOME=/usr/local/rust
ENV PATH="${CARGO_HOME}/bin:${PATH}"

ARG JULIA_VERSION=1.13.0

RUN set -eux; \
    curl -fsSL "https://julialang-s3.julialang.org/bin/linux/x64/1.13/julia-${JULIA_VERSION}-linux-x86_64.tar.gz" -o julia.tar.gz; \
    curl -fsSL "https://julialang-s3.julialang.org/bin/linux/x64/1.13/julia-${JULIA_VERSION}-linux-x86_64.tar.gz.asc" -o julia.tar.gz.asc; \
    curl -fsSL "https://julialang.org{JULIA_VERSION}-signing-key.asc" -o julia-key.asc; \
    gpg --import julia-key.asc; \
    gpg --verify julia.tar.gz.asc julia.tar.gz; \
    tar -xzf julia.tar.gz -C /opt; \
    ln -s /opt/julia-${JULIA_VERSION}/bin/julia /usr/local/bin/julia;

RUN mkdir -p /opt/.julia /opt/dev && chmod 777 /opt/.julia /opt/dev

ENV JULIA_DEPOT_PATH="/opt/.julia" \
    JULIA_PKG_DEVDIR="/opt/dev" \
    JULIA_NUM_THREADS=4

RUN julia -e 'using Pkg; Pkg.add(["Flux", "Images", "HTTP", "JSON", "Dates", "Logging", "Printf"]); Pkg.precompile()' 2>/dev/null || true

ARG UID=1000
ARG GID=1000
RUN groupadd --gid ${GID} bpla && \
    useradd --uid ${UID} --gid ${GID} --create-home --shell /bin/bash bpla

COPY --from=builder /tmp/.julia /home/bpla/.julia

ENV JULIA_DEPOT_PATH="/home/bpla/.julia" \
    JULIA_NUM_THREADS=4 \
    OPENBLAS_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 \
    OMP_NUM_THREADS=4 \
    PATH="/home/bpla/.julia/bin:$PATH"

WORKDIR /app

COPY --chown=bpla:bpla bpla_person_detection/ .

RUN mkdir -p /app/models /app/outputs /app/logs /tmp && \
    chown -R bpla:bpla /app

USER bpla

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 CMD julia -e 'using BPLAPersonDetection; @info "Health check passed"' 2>/dev/null || exit 1

EXPOSE 8000 7860

ENTRYPOINT ["/usr/bin/tini", "--"]

CMD ["julia", "--project=.", "-e", "using BPLAPersonDetection; BPLAPersonDetection.run_serve()"]
