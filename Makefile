SHELL := /usr/bin/env bash

ROOT ?= $(CURDIR)
WASM_SCRIPT ?= $(ROOT)/scripts/build_wasm_fixed_bundles.sh
WASM_DIST ?= $(ROOT)/dist/wasm-fixed-bundles
BINDGEN_TARGET ?= web
RUST_TOOLCHAIN ?= nightly
RUST_WASM_TARGET ?= wasm32-unknown-unknown

.PHONY: wasm wasm-node wasm-clean wasm-check

wasm:
	BINDGEN_TARGET=$(BINDGEN_TARGET) \
	RUST_TOOLCHAIN=$(RUST_TOOLCHAIN) \
	RUST_WASM_TARGET=$(RUST_WASM_TARGET) \
	$(WASM_SCRIPT)

wasm-node:
	BINDGEN_TARGET=nodejs \
	RUST_TOOLCHAIN=$(RUST_TOOLCHAIN) \
	RUST_WASM_TARGET=$(RUST_WASM_TARGET) \
	$(WASM_SCRIPT)

wasm-clean:
	rm -rf $(WASM_DIST) $(ROOT)/pkg

wasm-check:
	test -f $(WASM_DIST)/pkg/encodec_rs.js
	test -f $(WASM_DIST)/pkg/encodec_rs_bg.wasm
	test -f $(WASM_DIST)/manifest.json
	test -d $(WASM_DIST)/bundles/encodec_48khz_6kbps_1333ms
	test -d $(WASM_DIST)/bundles/encodec_48khz_6kbps_1800ms
	test -d $(WASM_DIST)/bundles/encodec_48khz_12kbps_1333ms
	test -d $(WASM_DIST)/bundles/encodec_48khz_12kbps_1800ms
