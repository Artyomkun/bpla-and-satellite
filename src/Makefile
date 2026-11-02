MAKEFILE_DIR := $(dir $(lastword $(MAKEFILE_LIST)))

include $(MAKEFILE_DIR)Make.user

export JULIA_CPU_TARGET
JULIA_PKG_SERVER :=
export JULIA_PKG_SERVER

.PHONY: julia-build project-compile

julia-build:
	$(MAKE) -j

project-compile:
	$(JULIA_SRC)/usr/bin/julia --project=$(MAKEFILE_DIR) -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'

