use std::io::Cursor;

use wasm_bindgen::prelude::*;

use crate::binary::read_ecdc_header;
use crate::format::EcdcMetadata;
use crate::metadata::OnnxFrameBundleMetadata;
use crate::raw::{
    decode_raw_frames, encode_raw_ecdc, encode_raw_frame_payload, encode_raw_header,
    overlap_add_decoded_frames, raw_segment_count, RawEcdcFrame,
};

#[wasm_bindgen(js_name = initPanicHook)]
pub fn init_panic_hook() {
    console_error_panic_hook::set_once();
}

#[wasm_bindgen(js_name = bundleMetadata)]
pub fn bundle_metadata(bundle_json: &str) -> Result<JsValue, JsValue> {
    let meta = parse_bundle(bundle_json)?;
    to_js_value(&meta)
}

#[wasm_bindgen(js_name = rawEcdcHeader)]
pub fn raw_ecdc_header(bundle_json: &str, audio_length: usize) -> Result<Vec<u8>, JsValue> {
    let meta = parse_bundle(bundle_json)?;
    encode_raw_header(&meta, audio_length).map_err(to_js_error)
}

#[wasm_bindgen(js_name = rawEcdcFramePayload)]
pub fn raw_ecdc_frame_payload(
    bundle_json: &str,
    codes: &[u16],
    scale: f32,
    frame_length: usize,
) -> Result<Vec<u8>, JsValue> {
    let meta = parse_bundle(bundle_json)?;
    encode_raw_frame_payload(&meta, codes, scale, frame_length).map_err(to_js_error)
}

#[wasm_bindgen(js_name = rawEcdcEncode)]
pub fn raw_ecdc_encode(
    bundle_json: &str,
    audio_length: usize,
    frames: JsValue,
) -> Result<Vec<u8>, JsValue> {
    let meta = parse_bundle(bundle_json)?;
    let frames: Vec<RawEcdcFrame> = serde_wasm_bindgen::from_value(frames).map_err(to_js_error)?;
    encode_raw_ecdc(&meta, audio_length, &frames).map_err(to_js_error)
}

#[wasm_bindgen(js_name = rawEcdcMetadata)]
pub fn raw_ecdc_metadata(payload: &[u8]) -> Result<JsValue, JsValue> {
    let metadata: EcdcMetadata =
        read_ecdc_header(&mut Cursor::new(payload)).map_err(to_js_error)?;
    to_js_value(&metadata)
}

#[wasm_bindgen(js_name = rawEcdcDecodeFrames)]
pub fn raw_ecdc_decode_frames(bundle_json: &str, payload: &[u8]) -> Result<JsValue, JsValue> {
    let meta = parse_bundle(bundle_json)?;
    let frames = decode_raw_frames(&meta, payload).map_err(to_js_error)?;
    to_js_value(&frames)
}

#[wasm_bindgen(js_name = rawEcdcSegmentCount)]
pub fn raw_ecdc_segment_count(bundle_json: &str, audio_length: usize) -> Result<usize, JsValue> {
    let meta = parse_bundle(bundle_json)?;
    Ok(raw_segment_count(&meta, audio_length))
}

#[wasm_bindgen(js_name = rawEcdcOverlapAdd)]
pub fn raw_ecdc_overlap_add(
    bundle_json: &str,
    audio_length: usize,
    decoded_frames: &[f32],
) -> Result<Vec<f32>, JsValue> {
    let meta = parse_bundle(bundle_json)?;
    overlap_add_decoded_frames(&meta, audio_length, decoded_frames).map_err(to_js_error)
}

fn parse_bundle(bundle_json: &str) -> Result<OnnxFrameBundleMetadata, JsValue> {
    serde_json::from_str(bundle_json).map_err(to_js_error)
}

fn to_js_value<T: serde::Serialize + ?Sized>(value: &T) -> Result<JsValue, JsValue> {
    serde_wasm_bindgen::to_value(value).map_err(to_js_error)
}

fn to_js_error(error: impl std::fmt::Display) -> JsValue {
    JsValue::from_str(&error.to_string())
}
