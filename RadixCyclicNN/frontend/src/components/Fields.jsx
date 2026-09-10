/**
 * Minimal labelled form controls. Panels keep numeric values as strings so
 * the user can clear a field while typing; parsing happens on submit.
 */

function Label({ label, hint }) {
  return (
    <span>
      {label}
      {hint ? <em> ({hint})</em> : null}
    </span>
  );
}

export function TextField({ label, value, onChange, placeholder, disabled, hint }) {
  return (
    <label className="field">
      <Label label={label} hint={hint} />
      <input
        type="text"
        value={value}
        placeholder={placeholder}
        disabled={disabled}
        onChange={(e) => onChange(e.target.value)}
      />
    </label>
  );
}

export function NumberField({ label, value, onChange, step = "any", min, max, disabled, hint, placeholder }) {
  return (
    <label className="field">
      <Label label={label} hint={hint} />
      <input
        type="number"
        value={value}
        step={step}
        min={min}
        max={max}
        placeholder={placeholder}
        disabled={disabled}
        onChange={(e) => onChange(e.target.value)}
      />
    </label>
  );
}

export function SelectField({ label, value, onChange, options, disabled, hint }) {
  return (
    <label className="field">
      <Label label={label} hint={hint} />
      <select value={value} disabled={disabled} onChange={(e) => onChange(e.target.value)}>
        {options.map(([optionValue, text]) => (
          <option key={optionValue} value={optionValue}>
            {text}
          </option>
        ))}
      </select>
    </label>
  );
}

export function CheckField({ label, checked, onChange, disabled }) {
  return (
    <label className="field inline">
      <input type="checkbox" checked={checked} disabled={disabled} onChange={(e) => onChange(e.target.checked)} />
      <span>{label}</span>
    </label>
  );
}

export function TextArea({ label, value, onChange, placeholder, rows = 6, disabled, hint }) {
  return (
    <label className="field">
      <Label label={label} hint={hint} />
      <textarea
        value={value}
        rows={rows}
        placeholder={placeholder}
        disabled={disabled}
        spellCheck={false}
        onChange={(e) => onChange(e.target.value)}
      />
    </label>
  );
}
