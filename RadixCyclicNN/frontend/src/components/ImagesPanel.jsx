import { useEffect, useRef, useState } from "react";
import { api } from "../api.js";
import { useJob } from "../hooks/useJob.js";
import { asArray, fmtInt, jobIsRunning, parseInteger, parseNumber } from "../util.js";
import Alert from "./Alert.jsx";
import JobStatus from "./JobStatus.jsx";
import { NumberField, SelectField, TextArea } from "./Fields.jsx";

const SIZES = [
  ["64", "64 × 64 (latent 8 × 8)"],
  ["128", "128 × 128 (latent 16 × 16)"],
  ["256", "256 × 256 (latent 32 × 32)"],
  ["512", "512 × 512 (latent 64 × 64)"],
];

/**
 * Images as text: the Stable Diffusion VAE run backwards (image -> compressed
 * latent), quantised, base64-encoded and fed to the model as a text; the
 * forward process (text -> latent -> VAE decoder) shows an encoded or predicted
 * text as an image again. Without the diffusion weights a thumbnail stand-in
 * with the same 8x reduction is used.
 */
export default function ImagesPanel({ status }) {
  const [info, setInfo] = useState(null);
  const [infoError, setInfoError] = useState(null);
  const [encoder, setEncoder] = useState("auto");
  const [size, setSize] = useState("128");
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [encoding, setEncoding] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [roundTrip, setRoundTrip] = useState(null);
  const [decodeText, setDecodeText] = useState("");
  const [decoded, setDecoded] = useState(null);
  const [decodeError, setDecodeError] = useState(null);
  const [decoding, setDecoding] = useState(false);
  const [epochs, setEpochs] = useState("3");
  const [lr, setLr] = useState("0.5");
  const [batchSize, setBatchSize] = useState("8");
  const [saveName, setSaveName] = useState("");
  const [saveNotice, setSaveNotice] = useState(null);
  const [copied, setCopied] = useState(false);
  const inputRef = useRef(null);
  const { job, running, busy, error: jobError, start, stop, clearError } = useJob("train");

  const otherJobRunning = jobIsRunning(status) && !running;

  useEffect(() => {
    let alive = true;
    api
      .images()
      .then((data) => {
        if (!alive) return;
        setInfo(data && typeof data === "object" ? data : null);
        setInfoError(null);
      })
      .catch((err) => {
        if (alive) setInfoError(err.message);
      });
    return () => {
      alive = false;
    };
  }, []);

  useEffect(() => {
    if (!file) {
      setPreview(null);
      return undefined;
    }
    const url = URL.createObjectURL(file);
    setPreview(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  async function encode(chosen) {
    const target = chosen || file;
    if (!target) return;
    setEncoding(true);
    setError(null);
    setResult(null);
    setRoundTrip(null);
    setCopied(false);
    try {
      const data = await api.imageEncode(target, { size: parseInteger(size, 128), encoder });
      setResult(data && typeof data === "object" ? data : null);
      if (data && data.text) {
        setDecodeText(data.text);
        setSaveName(target.name.replace(/\.[^.]+$/, "") + ".txt");
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setEncoding(false);
    }
  }

  function pick(fileList) {
    const chosen = Array.from(fileList || [])[0];
    if (!chosen) return;
    setFile(chosen);
    encode(chosen);
  }

  async function decode(text, setter, setErr, setBusy) {
    if (!String(text || "").trim()) return;
    setBusy(true);
    setErr(null);
    try {
      const data = await api.imageDecode(text);
      setter(data && typeof data === "object" ? data : null);
    } catch (err) {
      setter(null);
      setErr(err.message);
    } finally {
      setBusy(false);
    }
  }

  async function train() {
    if (!result || !result.text) return;
    await start(() =>
      api.train({
        texts: [result.text],
        epochs: parseInteger(epochs, 3),
        lr: parseNumber(lr, 0.5),
        batch_size: parseInteger(batchSize, 8),
      }),
    );
  }

  async function saveUpload() {
    if (!result || !result.text) return;
    setSaveNotice(null);
    try {
      const data = await api.upload(saveName.trim() || "image.txt", result.text + "\n");
      const record = asArray(data && data.uploads)[0];
      setSaveNotice(record ? `Saved as upload ${record.name} (${fmtInt(record.chars)} chars); tick it on the Train tab.` : "Saved.");
    } catch (err) {
      setSaveNotice(`Could not save: ${err.message}`);
    }
  }

  async function copyText() {
    if (!result || !result.text) return;
    try {
      await navigator.clipboard.writeText(result.text);
      setCopied(true);
    } catch {
      setCopied(false);
    }
  }

  const encoderOptions = [
    ["auto", info ? `auto (${info.auto || "unavailable"})` : "auto"],
    ["sd", "Stable Diffusion VAE"],
    ["tiny", "thumbnail stand-in (no weights needed)"],
  ];
  const usable = !info || info.pillow;

  return (
    <>
      <div className="card">
        <h2>Images</h2>
        <p className="muted">
          The Stable Diffusion VAE is run backwards: an image becomes its compressed latent (4 channels at 1/8 of the
          size, 48× fewer numbers than the pixels), each number becomes one signed byte, the bytes become base64, and
          that text is what the model trains on and predicts. Decoding runs the forward process again, so an encoded
          or predicted text can be looked at.
        </p>
        {infoError ? <Alert message={`Cannot read the encoder status: ${infoError}`} /> : null}
        {info ? (
          <p className="muted">
            Pillow {info.pillow ? "✓" : "✗"} · torch {info.torch ? "✓" : "✗"} · diffusers {info.diffusers ? "✓" : "✗"} · VAE{" "}
            <code>{String(info.sd_model)}</code> {info.sd_loaded ? "(loaded)" : info.sd_error ? `(not loaded: ${info.sd_error})` : "(loads on first use)"}
            {" · "}auto → <b>{info.auto || "nothing (install pillow)"}</b>
          </p>
        ) : null}
        <div className="row">
          <SelectField label="Encoder" value={encoder} onChange={setEncoder} options={encoderOptions} disabled={encoding} />
          <SelectField label="Size" hint="the image is resized to a square" value={size} onChange={setSize} options={SIZES} disabled={encoding} />
        </div>
        <input
          ref={inputRef}
          type="file"
          accept="image/*"
          hidden
          onChange={(e) => {
            pick(e.target.files);
            e.target.value = "";
          }}
        />
        <div className="actions">
          <button type="button" className="primary" disabled={encoding || !usable} onClick={() => inputRef.current && inputRef.current.click()}>
            {encoding ? "Encoding…" : "Choose an image…"}
          </button>
          <button type="button" className="small" disabled={encoding || !file} onClick={() => encode()}>
            Re-encode
          </button>
        </div>
        <Alert message={error} onDismiss={() => setError(null)} />
        {preview ? (
          <div className="image-row">
            <figure>
              <img src={preview} alt="the chosen image" />
              <figcaption>{file ? file.name : ""}</figcaption>
            </figure>
            {roundTrip && roundTrip.png_base64 ? (
              <figure>
                <img src={`data:image/png;base64,${roundTrip.png_base64}`} alt="decoded from the text" />
                <figcaption>
                  decoded back ({roundTrip.encoder}, {fmtInt(roundTrip.width)}×{fmtInt(roundTrip.height)})
                </figcaption>
              </figure>
            ) : null}
          </div>
        ) : null}
        {result ? (
          <>
            <dl className="kv">
              <dt>encoder</dt>
              <dd>{String(result.encoder)}</dd>
              <dt>latent</dt>
              <dd>
                {asArray(result.latent_shape).join(" × ")} ({fmtInt(result.bytes)} bytes)
              </dd>
              <dt>text</dt>
              <dd>{fmtInt(result.chars)} chars</dd>
              <dt>source</dt>
              <dd>{asArray(result.source_size).join(" × ")} px</dd>
            </dl>
            <TextArea label="Encoded text (what the model sees)" value={result.text} onChange={() => {}} rows={5} />
            <div className="actions">
              <button type="button" className="small" onClick={copyText}>
                {copied ? "Copied" : "Copy text"}
              </button>
              <button type="button" className="small" disabled={decoding} onClick={() => decode(result.text, setRoundTrip, setError, setDecoding)}>
                Decode back to an image
              </button>
            </div>
          </>
        ) : null}
      </div>

      <div className="card">
        <h2>Feed it into the model</h2>
        <div className="row">
          <NumberField label="Epochs" value={epochs} onChange={setEpochs} min={1} step={1} disabled={running} />
          <NumberField label="Learning rate" value={lr} onChange={setLr} min={0} disabled={running} />
          <NumberField label="Batch size" value={batchSize} onChange={setBatchSize} min={1} step={1} disabled={running} />
        </div>
        <div className="actions">
          <button type="button" className="primary" disabled={!result || running || busy || otherJobRunning} onClick={train}>
            {busy ? "Starting…" : "Train on this image"}
          </button>
          <button type="button" className="danger" disabled={!running} onClick={() => stop()}>
            Stop
          </button>
        </div>
        {otherJobRunning ? <p className="muted">Another job is running; wait for it to finish.</p> : null}
        <JobStatus job={job} emptyText="No training job yet. Encode an image, then press Train." />
        <Alert message={jobError} onDismiss={clearError} />
        <div className="row">
          <label className="field">
            <span>Save as upload</span>
            <input type="text" value={saveName} onChange={(e) => setSaveName(e.target.value)} placeholder="photo.txt" disabled={!result} />
          </label>
          <div className="actions">
            <button type="button" className="small" disabled={!result} onClick={saveUpload}>
              Save
            </button>
          </div>
        </div>
        {saveNotice ? <p className="muted">{saveNotice}</p> : null}
        <p className="muted">
          Saved texts appear in every upload picker, so several images can be trained on together; a prediction that
          starts with an encoded text (Predict tab) can be pasted below to see it.
        </p>
      </div>

      <div className="card">
        <h2>Decode a text</h2>
        <TextArea
          label="Encoded or predicted text"
          hint="img:<encoder>:<w>x<h>:<base64>; a cut-off tail is padded"
          value={decodeText}
          onChange={setDecodeText}
          rows={4}
          placeholder="img:sd:128x128:AAAA…"
        />
        <div className="actions">
          <button type="button" className="primary" disabled={decoding || !decodeText.trim()} onClick={() => decode(decodeText, setDecoded, setDecodeError, setDecoding)}>
            {decoding ? "Decoding…" : "Decode"}
          </button>
        </div>
        <Alert message={decodeError} onDismiss={() => setDecodeError(null)} />
        {decoded && decoded.png_base64 ? (
          <div className="image-row">
            <figure>
              <img src={`data:image/png;base64,${decoded.png_base64}`} alt="decoded text" />
              <figcaption>
                {decoded.encoder}, {fmtInt(decoded.width)}×{fmtInt(decoded.height)}
                {decoded.repaired ? " · the text was repaired (padded or truncated)" : ""}
              </figcaption>
            </figure>
          </div>
        ) : null}
      </div>
    </>
  );
}
