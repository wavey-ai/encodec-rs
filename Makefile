SHELL := /usr/bin/env bash

ROOT ?= $(CURDIR)

WASM_SCRIPT ?= $(ROOT)/scripts/build_wasm_fixed_bundles.sh
BUNDLE_EXPORT_SCRIPT ?= $(ROOT)/scripts/export_encodec_fixed_chunk_bundles.sh
PREFIX_BUNDLE_SCRIPT ?= $(ROOT)/scripts/derive-codebook-prefix-bundle.py
PYTHON_BIN ?= /opt/anaconda3/envs/encodec-export/bin/python

WASM_DIST ?= $(ROOT)/dist/wasm-fixed-bundles
ONNX_BUNDLES ?= $(ROOT)/onnx-bundles

BINDGEN_TARGET ?= web
RUST_TOOLCHAIN ?= nightly
RUST_WASM_TARGET ?= wasm32-unknown-unknown

FIXED_BUNDLES := \
	encodec_48khz_3kbps_1333ms \
	encodec_48khz_6kbps_1333ms \
	encodec_48khz_12kbps_1333ms \
	encodec_48khz_12kbps_7cb_1333ms \
	encodec_48khz_24kbps_1333ms \
	encodec_48khz_3kbps_1800ms \
	encodec_48khz_6kbps_1800ms \
	encodec_48khz_12kbps_1800ms \
	encodec_48khz_12kbps_7cb_1800ms \
	encodec_48khz_24kbps_1800ms

.PHONY: \
	wasm \
	wasm-node \
	wasm-clean \
	wasm-check \
	bundles \
	bundles-clean \
	bundles-check \
	rebuild-bundles \
	benchmark-encodec-memory

wasm:
	BUNDLES="$(FIXED_BUNDLES)" \
	BINDGEN_TARGET="$(BINDGEN_TARGET)" \
	RUST_TOOLCHAIN="$(RUST_TOOLCHAIN)" \
	RUST_WASM_TARGET="$(RUST_WASM_TARGET)" \
	"$(WASM_SCRIPT)"

wasm-node:
	BUNDLES="$(FIXED_BUNDLES)" \
	BINDGEN_TARGET=nodejs \
	RUST_TOOLCHAIN="$(RUST_TOOLCHAIN)" \
	RUST_WASM_TARGET="$(RUST_WASM_TARGET)" \
	"$(WASM_SCRIPT)"

wasm-clean:
	rm -rf "$(WASM_DIST)" "$(ROOT)/pkg"

bundles-clean:
	rm -rf \
		"$(ONNX_BUNDLES)/encodec_48khz_3kbps_1333ms" \
		"$(ONNX_BUNDLES)/encodec_48khz_3kbps_1800ms" \
		"$(ONNX_BUNDLES)/encodec_48khz_6kbps_1333ms" \
		"$(ONNX_BUNDLES)/encodec_48khz_6kbps_1800ms" \
		"$(ONNX_BUNDLES)/encodec_48khz_12kbps_1333ms" \
		"$(ONNX_BUNDLES)/encodec_48khz_12kbps_1800ms" \
		"$(ONNX_BUNDLES)/encodec_48khz_12kbps_7cb_1333ms" \
		"$(ONNX_BUNDLES)/encodec_48khz_12kbps_7cb_1800ms" \
		"$(ONNX_BUNDLES)/encodec_48khz_24kbps_1333ms" \
		"$(ONNX_BUNDLES)/encodec_48khz_24kbps_1800ms"

bundles:
	CHUNKS="1333ms:64960:64000 1800ms:87360:86400" \
	ENCODEC_RS_REPO="$(ROOT)" \
	"$(BUNDLE_EXPORT_SCRIPT)"
	"$(PYTHON_BIN)" "$(PREFIX_BUNDLE_SCRIPT)" \
		"$(ONNX_BUNDLES)/encodec_48khz_12kbps_1333ms" \
		"$(ONNX_BUNDLES)/encodec_48khz_12kbps_7cb_1333ms" \
		--codebooks 7 --profile-bandwidth-kbps 12
	"$(PYTHON_BIN)" "$(PREFIX_BUNDLE_SCRIPT)" \
		"$(ONNX_BUNDLES)/encodec_48khz_12kbps_1800ms" \
		"$(ONNX_BUNDLES)/encodec_48khz_12kbps_7cb_1800ms" \
		--codebooks 7 --profile-bandwidth-kbps 12

bundles-check:
	@set -euo pipefail; \
	for bundle in $(FIXED_BUNDLES); do \
		dir="$(ONNX_BUNDLES)/$$bundle"; \
		test -f "$$dir/bundle.json"; \
		test -f "$$dir/encode_frame.onnx"; \
		test -f "$$dir/decode_frame.onnx"; \
		test -f "$$dir/lm_weights_q8.bin"; \
	done
	@if grep -RInE \
		'left_guard_samples|right_guard_samples|owned_samples|seam_repair|seam_repair_samples' \
		$(addprefix "$(ONNX_BUNDLES)/",$(FIXED_BUNDLES)); then \
		echo "obsolete guard metadata found in rebuilt bundles" >&2; \
		exit 1; \
	fi
	@echo "bundle checks passed"

rebuild-bundles:
	$(MAKE) bundles-clean
	$(MAKE) bundles
	$(MAKE) bundles-check
	$(MAKE) wasm-clean
	$(MAKE) wasm
	$(MAKE) wasm-check

benchmark-encodec-memory:
	/usr/bin/time -l node --expose-gc tools/benchmark-encodec-memory.mjs

wasm-check:
	test -f "$(WASM_DIST)/pkg/encodec_rs.js"
	test -f "$(WASM_DIST)/pkg/encodec_rs_bg.wasm"
	test -f "$(WASM_DIST)/pkg/package.json"
	test -f "$(WASM_DIST)/encodec-ecdc-runtime.js"
	test -f "$(WASM_DIST)/manifest.json"
	@set -euo pipefail; \
	for bundle in $(FIXED_BUNDLES); do \
		test -d "$(WASM_DIST)/bundles/$$bundle"; \
	done
	@if grep -RInE \
		'left_guard_samples|right_guard_samples|owned_samples|seam_repair|seam_repair_samples' \
		"$(WASM_DIST)"; then \
		echo "obsolete guard metadata found in WASM distribution" >&2; \
		exit 1; \
	fi
	@echo "WASM distribution checks passed"
