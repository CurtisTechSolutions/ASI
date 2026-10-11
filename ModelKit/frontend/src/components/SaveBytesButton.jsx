import { base64Bytes, bytesName } from "../decoded.js";

/**
 * 💾 Save: the output decoder of a model of base64 symbols. The answer's
 * symbols are decoded in the browser, exactly as the servers decode them
 * (`base64Bytes`), and handed over as a file - an image, a recording, a text
 * again - so what the model said can be opened in whatever reads it.
 */
export default function SaveBytesButton({ text, name, label, compact = false }) {
  const bytes = base64Bytes(text);

  function save() {
    const url = URL.createObjectURL(new Blob([bytes], { type: "application/octet-stream" }));
    const link = document.createElement("a");
    link.href = url;
    link.download = bytesName(name);
    document.body.appendChild(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 0);
  }

  return (
    <button
      type="button"
      className="rate save"
      aria-label={`save ${label} as bytes`}
      title={`Save: the ${bytes.length} bytes the symbols hold, as a file`}
      disabled={bytes.length === 0}
      onClick={save}
    >
      💾{compact ? "" : " Save bytes"}
    </button>
  );
}
