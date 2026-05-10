const PAGES_ORIGIN = "https://encodec-rs-browser-smoke.pages.dev";
const PREFIX = "/code/encodec-rs";

export default {
  async fetch(request) {
    const incomingUrl = new URL(request.url);
    if (incomingUrl.pathname === PREFIX || incomingUrl.pathname === `${PREFIX}/`) {
      return Response.redirect(`${incomingUrl.origin}${PREFIX}/browser-smoke/`, 302);
    }

    const upstreamUrl = new URL(PAGES_ORIGIN);
    upstreamUrl.pathname = incomingUrl.pathname;
    upstreamUrl.search = incomingUrl.search;

    const upstreamResponse = await fetch(upstreamUrl, {
      cf: {
        cacheEverything: true,
      },
    });
    if (upstreamResponse.status === 404) {
      const partsResponse = await fetchPartsAsset(upstreamUrl);
      if (partsResponse) {
        return withDemoHeaders(partsResponse, incomingUrl);
      }
    }

    return withDemoHeaders(upstreamResponse, incomingUrl);
  },
};

async function fetchPartsAsset(upstreamUrl) {
  const manifestUrl = new URL(upstreamUrl);
  manifestUrl.pathname = `${upstreamUrl.pathname}.parts.json`;
  const manifestResponse = await fetch(manifestUrl);
  if (!manifestResponse.ok) {
    return null;
  }

  const manifest = await manifestResponse.json();
  if (!Array.isArray(manifest.parts) || !Number.isInteger(manifest.byteLength)) {
    return null;
  }

  const buffers = await Promise.all(
    manifest.parts.map(async (part) => {
      const partUrl = new URL(part, manifestUrl);
      const partResponse = await fetch(partUrl);
      if (!partResponse.ok) {
        throw new Error(`Failed to fetch asset part ${partUrl.pathname}: ${partResponse.status}`);
      }
      return partResponse.arrayBuffer();
    }),
  );
  const body = new Uint8Array(manifest.byteLength);
  let offset = 0;
  for (const buffer of buffers) {
    body.set(new Uint8Array(buffer), offset);
    offset += buffer.byteLength;
  }

  return new Response(body, {
    headers: {
      "Content-Length": String(body.byteLength),
      "Content-Type": contentTypeForPath(upstreamUrl.pathname),
    },
  });
}

function withDemoHeaders(response, incomingUrl) {
  const headers = new Headers(response.headers);
  headers.set("Cross-Origin-Opener-Policy", "same-origin");
  headers.set("Cross-Origin-Embedder-Policy", "require-corp");
  headers.set("Cross-Origin-Resource-Policy", "same-origin");
  headers.set("X-Encodec-Rs-Demo", "cloudflare-worker");
  if (incomingUrl.pathname === `${PREFIX}/browser-smoke/` || incomingUrl.pathname.endsWith("/index.html")) {
    headers.set("Cache-Control", "no-store");
  }

  return new Response(response.body, {
    status: response.status,
    statusText: response.statusText,
    headers,
  });
}

function contentTypeForPath(pathname) {
  if (pathname.endsWith(".wasm")) {
    return "application/wasm";
  }
  if (pathname.endsWith(".mjs") || pathname.endsWith(".js")) {
    return "text/javascript; charset=utf-8";
  }
  if (pathname.endsWith(".onnx")) {
    return "application/octet-stream";
  }
  return "application/octet-stream";
}
