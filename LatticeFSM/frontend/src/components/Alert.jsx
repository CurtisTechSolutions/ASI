/** Inline message box. kind: "error" | "ok" | "info". Renders nothing without a message. */
export default function Alert({ kind = "error", message, onDismiss }) {
  if (!message) return null;
  return (
    <div className={`alert ${kind}`} role={kind === "error" ? "alert" : "status"}>
      <span>{String(message)}</span>
      {onDismiss ? (
        <button type="button" className="link" onClick={onDismiss} aria-label="Dismiss">
          ×
        </button>
      ) : null}
    </div>
  );
}
