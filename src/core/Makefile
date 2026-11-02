.PHONY: all build test check check-host fmt clippy clean help install-target ferrocene-check ferrocene-evidence

TARGET       := riscv64gc-unknown-linux-gnu
CARGO        ?= cargo
PROFILE      ?= release
RV_RUSTFLAGS := -C target-feature=-hard-float

FERROCENE_CARGO   ?= cargo +ferrocene
FERROCENE_TARGET ?= unknown-linux-gnu

help:
	@echo ""
	@echo "BPLA Core (Rust) Build Options"
	@echo "============================"
	@echo "  build              - Release build for $(TARGET)"
	@echo "  test               - Run host unit tests"
	@echo "  check              - cargo check for $(TARGET)"
	@echo "  check-host         - cargo check + clippy on the host"
	@echo "  fmt                - Format sources"
	@echo "  install-target     - Add $(TARGET) via rustup"
	@echo "  ferrocene-check    - check + clippy with the Ferrocene toolchain ($(FERROCENE_TARGET))"
	@echo "  ferrocene-evidence - Write toolchain and dependency evidence to evidence/"
	@echo "  clean              - Remove build artifacts"
	@echo ""

all: check

build:
	RUSTFLAGS="$(RV_RUSTFLAGS)" $(CARGO) build --$(PROFILE) --target $(TARGET)

check:
	RUSTFLAGS="$(RV_RUSTFLAGS)" $(CARGO) check --target $(TARGET)

install-target:
	rustup target add $(TARGET)

ferrocene-check:
	$(FERROCENE_CARGO) check --target $(FERROCENE_TARGET) --no-default-features
	$(FERROCENE_CARGO) clippy --target $(FERROCENE_TARGET) -- -D warnings

ferrocene-evidence:
	@mkdir -p evidence
	rustc +ferrocene -vV > evidence/rustc-version.txt
	rustc +ferrocene --print cfg --target $(FERROCENE_TARGET) > evidence/target-cfg.txt
	$(FERROCENE_CARGO) tree --target $(FERROCENE_TARGET) > evidence/dependencies.txt
	$(FERROCENE_CARGO) build --$(PROFILE) --target $(FERROCENE_TARGET) -v > evidence/build.log 2>&1
	@echo "Evidence written to evidence/"

test:
	$(CARGO) test

check-host:
	$(CARGO) check --all-targets --features std
	$(CARGO) clippy --all-targets --features std -- -D warnings

fmt:
	$(CARGO) fmt

clean:
	$(CARGO) clean
	@echo "Clean complete."