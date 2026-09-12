#ifndef ENCODEC_NATIVE_EMSCRIPTEN_H
#define ENCODEC_NATIVE_EMSCRIPTEN_H

/*
 * Native stand-in for the single Emscripten macro the custom EnCodec kernels
 * use. Those kernels already have external linkage, so the attribute only
 * documents the intent and keeps the symbols visible in a shared library.
 */
#ifndef EMSCRIPTEN_KEEPALIVE
#define EMSCRIPTEN_KEEPALIVE __attribute__((used, visibility("default")))
#endif

#endif
