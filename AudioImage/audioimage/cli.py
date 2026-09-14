"""Command line: ``python -m audioimage <command> [...]`` / ``audioimage``.

    audioimage tone -o tune.wav --kind melody
    audioimage encode tune.wav -o tune.png
    audioimage decode tune.png -o back.wav
    audioimage roundtrip tune.wav            # both, and how much survived
    audioimage view tune.wav -o view.png     # the same plane, labelled
    audioimage info tune.png
    audioimage serve                         # the same thing, in a browser

Every command prints a short human-readable report by default and, with
``--json``, exactly one JSON document on stdout - progress and notices then go
to stderr so stdout stays machine-readable.

Exit codes: ``0`` success, ``1`` a handled error (one line on stderr), ``130``
interrupted.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Callable, NoReturn, Sequence, TextIO

from . import __version__
from .backend import BACKENDS, describe_backends
from .codec import (
    EncodeConfig,
    compare,
    decode,
    encode,
    image_to_plane,
    read_meta,
)
from .colormap import COLORMAPS
from .dsp import WINDOWS
from .png import Image, PngError, SIGNATURE, read_png, write_png
from .render import ViewConfig, render_view, render_waveform
from .spectrogram import FREQ_SCALES, ORIGINS, SCALES, PlaneConfig
from .server import DEFAULT_HOST, DEFAULT_PORT, frontend_dir, run_server
from .synth import KINDS, generate
from .wav import Audio, WavError, read_wav, write_wav

__all__ = ["CliError", "EXIT_ABORTED", "EXIT_ERROR", "EXIT_OK", "build_parser", "main"]

PROG = "audioimage"
EXIT_OK = 0
EXIT_ERROR = 1
EXIT_ABORTED = 130

_HANDLED = (ValueError, TypeError, OSError, LookupError, ImportError, ArithmeticError)


class CliError(Exception):
    """A user-facing problem: one line on stderr, exit code 1."""


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def _die(message: str) -> NoReturn:
    raise CliError(message)


def _out(args: argparse.Namespace) -> TextIO:
    """Where human-readable text goes: stderr while stdout is reserved for JSON."""
    return sys.stderr if args.json else sys.stdout


def _report(args: argparse.Namespace, payload: dict[str, Any], lines: Sequence[str]) -> None:
    """Print either the JSON document or the human-readable lines."""
    if args.json:
        json.dump(payload, sys.stdout, indent=2, sort_keys=True, default=float)
        sys.stdout.write("\n")
    else:
        for line in lines:
            print(line)


def _looks_like_png(path: str) -> bool:
    try:
        with open(path, "rb") as fh:
            return fh.read(8) == SIGNATURE
    except OSError as exc:
        _die(f"cannot read {path}: {exc}")


def _default_out(path: str, suffix: str, extension: str) -> str:
    stem = os.path.splitext(path)[0]
    return f"{stem}{suffix}{extension}"


def _plane_config(args: argparse.Namespace) -> PlaneConfig:
    return PlaneConfig(
        scale=args.scale,
        top_db=args.top_db,
        ref=args.ref,
        freq_scale=args.freq_scale,
        f_min=args.f_min,
        f_max=args.f_max,
        height=args.height,
        width=args.width,
        origin=args.origin,
    )


def _encode_config(args: argparse.Namespace) -> EncodeConfig:
    hop = args.hop if args.hop is not None else max(1, args.n_fft // 4)
    return EncodeConfig(
        n_fft=args.n_fft,
        hop=hop,
        window=args.window,
        center=not args.no_center,
        plane=_plane_config(args),
        colormap=args.colormap,
        depth=args.depth,
    )


def _progress(args: argparse.Namespace) -> Callable[[int, int, float], None] | None:
    """A one-line, self-overwriting progress report on stderr."""
    if args.quiet:
        return None
    stream = sys.stderr
    if not stream.isatty() and not args.verbose:
        return None
    last = [0.0]

    def report(step: int, total: int, error: float) -> None:
        now = time.monotonic()
        if step != total and now - last[0] < 0.1:
            return
        last[0] = now
        stream.write(f"\r  griffin-lim {step}/{total}  error {error:.4f}   ")
        if step == total:
            stream.write("\n")
        stream.flush()

    return report


def _load_audio(path: str) -> Audio:
    if not os.path.exists(path):
        _die(f"no such file: {path}")
    if _looks_like_png(path):
        _die(f"{path} is a PNG; this command wants a WAV")
    return read_wav(path)


def _load_image(path: str) -> Image:
    if not os.path.exists(path):
        _die(f"no such file: {path}")
    if not _looks_like_png(path):
        _die(f"{path} is not a PNG")
    return read_png(path)


def _fmt_size(path: str) -> str:
    try:
        return f"{os.path.getsize(path):,} bytes"
    except OSError:  # pragma: no cover - the file was just written
        return "?"


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------


def cmd_tone(args: argparse.Namespace) -> int:
    """Synthesise a test clip."""
    notes = tuple(n for n in (args.notes or "").replace(",", " ").split() if n)
    extra: dict[str, Any] = {"amplitude": args.amplitude, "seed": args.seed, "noise": args.noise, "method": args.method}
    if notes:
        extra["notes"] = notes
    if args.f1 is not None:
        extra["f1"] = args.f1
    audio = generate(args.kind, duration=args.duration, rate=args.rate, freq=args.freq, **extra)
    write_wav(args.output, audio, bits=args.bits)
    payload = {"command": "tone", "output": args.output, "kind": args.kind, **audio.describe()}
    _report(
        args,
        payload,
        [
            f"wrote {args.output}  ({_fmt_size(args.output)})",
            f"  {args.kind}: {audio.duration:.2f}s at {audio.sample_rate} Hz, peak {audio.peak:.3f}",
        ],
    )
    return EXIT_OK


def cmd_encode(args: argparse.Namespace) -> int:
    """A WAV to a picture of its spectrum."""
    audio = _load_audio(args.input)
    cfg = _encode_config(args)
    started = time.monotonic()
    image = encode(audio, cfg, backend=args.backend)
    write_png(args.output, image)
    elapsed = time.monotonic() - started
    meta = read_meta(image)
    payload = {
        "command": "encode",
        "input": args.input,
        "output": args.output,
        "seconds": round(elapsed, 3),
        "image": image.describe(),
        "audio": audio.describe(),
        "meta": meta,
    }
    _report(
        args,
        payload,
        [
            f"wrote {args.output}  ({_fmt_size(args.output)}) in {elapsed:.2f}s",
            f"  {image.width} columns x {image.height} rows, {image.depth}-bit {image.mode}, colormap {cfg.colormap}",
            f"  {audio.duration:.2f}s at {audio.sample_rate} Hz -> n_fft {cfg.n_fft}, hop {cfg.hop}, {cfg.window}",
            f"  amplitude scale {cfg.plane.scale}"
            + (f" over {cfg.plane.top_db:.0f} dB" if cfg.plane.scale == "db" else "")
            + f", frequency axis {cfg.plane.freq_scale}",
        ],
    )
    return EXIT_OK


def cmd_decode(args: argparse.Namespace) -> int:
    """A picture back to a WAV."""
    image = _load_image(args.input)
    started = time.monotonic()
    result = decode(
        image,
        fallback=_encode_config(args),
        sample_rate=args.rate,
        iterations=args.iters,
        momentum=args.momentum,
        seed=args.seed,
        init=args.init,
        backend=args.backend,
        level=args.level,
        progress=_progress(args),
    )
    write_wav(args.output, result.audio, bits=args.bits)
    elapsed = time.monotonic() - started
    payload = {
        "command": "decode",
        "input": args.input,
        "output": args.output,
        "seconds": round(elapsed, 3),
        **result.describe(),
    }
    notes = [] if result.had_meta else ["  (no settings stored in the picture - defaults were used)"]
    _report(
        args,
        payload,
        [
            f"wrote {args.output}  ({_fmt_size(args.output)}) in {elapsed:.2f}s",
            f"  {result.audio.duration:.2f}s at {result.audio.sample_rate} Hz",
            f"  {args.iters} griffin-lim passes, final error {result.error:.4f}",
            *notes,
        ],
    )
    return EXIT_OK


def cmd_roundtrip(args: argparse.Namespace) -> int:
    """Encode, decode, and measure what survived."""
    audio = _load_audio(args.input)
    cfg = _encode_config(args)
    png_path = args.image or _default_out(args.input, "", ".png")
    wav_path = args.output or _default_out(args.input, "-decoded", ".wav")

    started = time.monotonic()
    image = encode(audio, cfg, backend=args.backend)
    write_png(png_path, image)
    encode_time = time.monotonic() - started

    started = time.monotonic()
    result = decode(
        image,
        iterations=args.iters,
        momentum=args.momentum,
        seed=args.seed,
        init=args.init,
        backend=args.backend,
        level=args.level,
        progress=_progress(args),
    )
    write_wav(wav_path, result.audio, bits=args.bits)
    decode_time = time.monotonic() - started

    metrics = compare(audio, result.audio, cfg, backend=args.backend)
    payload = {
        "command": "roundtrip",
        "input": args.input,
        "image": png_path,
        "output": wav_path,
        "encode_seconds": round(encode_time, 3),
        "decode_seconds": round(decode_time, 3),
        "iterations": args.iters,
        "metrics": metrics,
        "picture": image.describe(),
    }
    snr = metrics["waveform_snr"]
    _report(
        args,
        payload,
        [
            f"{args.input} -> {png_path} -> {wav_path}",
            f"  picture   {image.width}x{image.height}, {image.depth}-bit, {_fmt_size(png_path)}",
            f"  encode    {encode_time:.2f}s     decode {decode_time:.2f}s ({args.iters} passes)",
            f"  spectrum  {metrics['spectral_convergence'] * 100:.2f}% relative error"
            f"   ({metrics['log_spectral_distance']:.2f} dB log distance)",
            f"  waveform  {snr:.2f} dB SNR - low is expected: the phase was rebuilt, not stored",
        ],
    )
    return EXIT_OK


def cmd_view(args: argparse.Namespace) -> int:
    """Draw a labelled chart of a clip or of a picture."""
    cfg = _encode_config(args)
    if _looks_like_png(args.input):
        image = _load_image(args.input)
        meta = read_meta(image)
        plane_cfg = PlaneConfig.from_meta(meta) if meta else cfg.plane
        plane = image_to_plane(image, str(meta.get("colormap", "gray")))
        rate = args.rate or int(meta.get("sample_rate", 0)) or 22050
        hop = int(meta.get("hop", cfg.hop))
        title = args.title if args.title is not None else os.path.basename(args.input)
    else:
        audio = _load_audio(args.input)
        image = encode(audio, cfg, backend=args.backend)
        plane = image_to_plane(image, cfg.colormap)
        plane_cfg = PlaneConfig.from_meta(read_meta(image))
        rate, hop = audio.sample_rate, cfg.hop
        title = args.title if args.title is not None else (
            f"{os.path.basename(args.input)}  {audio.duration:.2f}s  {rate} Hz  n_fft {cfg.n_fft}"
        )
    view = ViewConfig(
        colormap=args.view_colormap,
        title=title,
        axes=not args.no_axes,
        legend=not args.no_legend,
        scale=args.zoom,
        width=args.view_width,
        height=args.view_height,
    )
    picture = render_view(plane, rate, hop, plane_cfg, view)
    write_png(args.output, picture)
    payload = {"command": "view", "input": args.input, "output": args.output, "image": picture.describe()}
    _report(
        args,
        payload,
        [
            f"wrote {args.output}  ({_fmt_size(args.output)})",
            f"  {picture.width}x{picture.height}, colormap {args.view_colormap}",
        ],
    )
    return EXIT_OK


def cmd_waveform(args: argparse.Namespace) -> int:
    """Draw the samples themselves."""
    audio = _load_audio(args.input)
    title = args.title if args.title is not None else (
        f"{os.path.basename(args.input)}  {audio.duration:.2f}s  {audio.sample_rate} Hz"
    )
    picture = render_waveform(
        audio,
        args.view_width or 900,
        args.view_height or 220,
        ViewConfig(colormap="gray", title=title, axes=not args.no_axes, legend=False),
    )
    write_png(args.output, picture)
    payload = {"command": "waveform", "input": args.input, "output": args.output, "image": picture.describe()}
    _report(args, payload, [f"wrote {args.output}  ({_fmt_size(args.output)})", f"  {picture.width}x{picture.height}"])
    return EXIT_OK


def cmd_info(args: argparse.Namespace) -> int:
    """Describe a WAV or a picture."""
    path = args.input
    if not os.path.exists(path):
        _die(f"no such file: {path}")
    payload: dict[str, Any] = {"command": "info", "input": path, "backends": describe_backends()}
    lines = [f"{path}  ({_fmt_size(path)})"]
    if _looks_like_png(path):
        image = read_png(path)
        meta = read_meta(image)
        payload["kind"] = "image"
        payload["image"] = image.describe()
        payload["meta"] = meta
        lines.append(f"  picture   {image.width} columns x {image.height} rows, {image.depth}-bit {image.mode}")
        if meta:
            rate = meta.get("sample_rate", "?")
            lines.append(f"  encoded   {meta.get('samples', '?')} samples at {rate} Hz, {meta.get('frames', '?')} frames")
            lines.append(
                f"  settings  n_fft {meta.get('n_fft')}, hop {meta.get('hop')}, {meta.get('window')} window,"
                f" {meta.get('scale')} scale, {meta.get('freq_scale')} frequency axis"
            )
            lines.append(f"  written   by {meta.get('tool', 'unknown')}")
        else:
            lines.append("  no audioimage settings in this file - decoding will use defaults")
    else:
        audio = read_wav(path)
        payload["kind"] = "audio"
        payload["audio"] = audio.describe()
        lines.append(
            f"  audio     {audio.duration:.2f}s at {audio.sample_rate} Hz,"
            f" {len(audio.samples)} samples, peak {audio.peak:.3f}, rms {audio.rms:.3f}"
        )
        lines.append(f"  source    {audio.meta.get('bits', '?')}-bit {audio.meta.get('format', '?')}, {audio.meta.get('channels', '?')} channel(s)")
    backend = payload["backends"]
    lines.append(f"  backend   {backend['selected']}" + (f" (numpy {backend.get('numpy_version')})" if backend.get("numpy") else " (numpy not installed)"))
    _report(args, payload, lines)
    return EXIT_OK


def cmd_serve(args: argparse.Namespace) -> int:
    """Put the whole tool behind a page in the browser."""
    directory = frontend_dir(args.frontend_dir)
    if not os.path.isfile(os.path.join(directory, "index.html")):
        _die(f"the front end is not in {directory}")
    url = f"http://{args.host}:{args.port}/"
    payload = {
        "command": "serve",
        "host": args.host,
        "port": args.port,
        "url": url,
        "frontend": directory,
        "backend": describe_backends(),
    }
    _report(
        args,
        payload,
        [
            f"audioimage {__version__} is serving on {url}",
            f"  front end  {directory}",
            f"  transforms {describe_backends()['selected']}",
            "  press Ctrl-C to stop",
        ],
    )
    sys.stdout.flush()
    try:
        run_server(args.host, args.port, directory, backend=args.backend, quiet=args.quiet)
    except KeyboardInterrupt:  # pragma: no cover - interactive
        pass
    return EXIT_OK


# ---------------------------------------------------------------------------
# the parser
# ---------------------------------------------------------------------------


def _add_encode_options(parser: argparse.ArgumentParser, full: bool = True) -> None:
    group = parser.add_argument_group("analysis")
    group.add_argument("--n-fft", type=int, default=1024, help="window length in samples, a power of two (default: 1024)")
    group.add_argument("--hop", type=int, default=None, help="samples between frames (default: n_fft // 4)")
    group.add_argument("--window", choices=WINDOWS, default="hann", help="window shape (default: hann)")
    group.add_argument("--no-center", action="store_true", help="do not pad half a window at each end")
    if not full:
        return
    plane = parser.add_argument_group("the plane")
    plane.add_argument("--scale", choices=SCALES, default="db", help="how amplitude becomes darkness (default: db)")
    plane.add_argument("--top-db", type=float, default=80.0, help="dB below the peak that reaches white (default: 80)")
    plane.add_argument("--ref", type=float, default=None, help="magnitude that maps to black (default: the clip's peak)")
    plane.add_argument("--freq-scale", choices=FREQ_SCALES, default="linear", help="frequency axis (default: linear, the only exact one)")
    plane.add_argument("--f-min", type=float, default=0.0, help="lowest frequency in the picture (default: 0)")
    plane.add_argument("--f-max", type=float, default=None, help="highest frequency (default: the Nyquist frequency)")
    plane.add_argument("--height", type=int, default=None, help="rows (default: one per frequency bin)")
    plane.add_argument("--width", type=int, default=None, help="columns (default: one per frame)")
    plane.add_argument("--origin", choices=ORIGINS, default="lower", help="which end of the picture is 0 Hz (default: lower)")
    plane.add_argument("--colormap", choices=COLORMAPS, default="gray", help="pixel colours (default: gray, the invertible one)")
    plane.add_argument("--depth", type=int, choices=(8, 16), default=16, help="bits per pixel (default: 16)")


def _add_decode_options(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("phase reconstruction")
    group.add_argument("--iters", type=int, default=64, help="Griffin-Lim passes (default: 64)")
    group.add_argument("--momentum", type=float, default=0.99, help="fast Griffin-Lim over-relaxation, 0 to 1 (default: 0.99)")
    group.add_argument("--init", choices=("random", "zeros"), default="random", help="starting phase (default: random)")
    group.add_argument("--seed", type=int, default=0, help="seed for the starting phase (default: 0)")
    group.add_argument("--level", choices=("auto", "peak", "none"), default="auto", help="output loudness (default: auto, the original's)")
    group.add_argument("--bits", type=int, choices=(8, 16, 24, 32), default=16, help="WAV bit depth (default: 16)")


def _add_global_options(parser: argparse.ArgumentParser, suppress: bool = False) -> None:
    """The options that work before *and* after the command name.

    On the sub-parsers the default is ``SUPPRESS``: without it, a flag that was
    given before the command ("--json encode ...") would be overwritten by the
    sub-parser's own default when it is not repeated after it.
    """
    default: Any = argparse.SUPPRESS if suppress else None
    parser.add_argument("--json", action="store_true", default=False if not suppress else default, help="print one JSON document on stdout")
    parser.add_argument("--backend", choices=BACKENDS, default="auto" if not suppress else default, help="which transform implementation to use")
    parser.add_argument("--quiet", "-q", action="store_true", default=False if not suppress else default, help="no progress reporting")
    parser.add_argument("--verbose", "-v", action="store_true", default=False if not suppress else default, help="progress even when stderr is not a terminal")


def build_parser() -> argparse.ArgumentParser:
    """The whole command line."""
    parser = argparse.ArgumentParser(
        prog=PROG,
        description="Encode a waveform into a picture - time across, frequency up, loudness as ink - and decode it back.",
    )
    parser.add_argument("--version", action="version", version=f"{PROG} {__version__}")
    _add_global_options(parser)
    subs = parser.add_subparsers(dest="command", metavar="command")

    p = subs.add_parser("tone", help="synthesise a test clip")
    p.add_argument("-o", "--output", default="tone.wav", help="where to write the WAV (default: tone.wav)")
    p.add_argument("--kind", choices=KINDS, default="melody", help="what to make (default: melody)")
    p.add_argument("--duration", type=float, default=2.0, help="seconds (default: 2)")
    p.add_argument("--rate", type=int, default=22050, help="sample rate (default: 22050)")
    p.add_argument("--freq", type=float, default=440.0, help="frequency, or where a sweep starts (default: 440)")
    p.add_argument("--f1", type=float, default=None, help="where a sweep ends (default: a quarter of the sample rate)")
    p.add_argument("--notes", default="", help="notes for a chord or melody, e.g. 'C4 E4 G4'")
    p.add_argument("--amplitude", type=float, default=0.6, help="peak amplitude (default: 0.6)")
    p.add_argument("--noise", choices=("white", "pink"), default="white", help="noise colour (default: white)")
    p.add_argument("--method", choices=("linear", "log"), default="linear", help="how a sweep rises (default: linear)")
    p.add_argument("--seed", type=int, default=0, help="seed for noise (default: 0)")
    p.add_argument("--bits", type=int, choices=(8, 16, 24, 32), default=16, help="WAV bit depth (default: 16)")
    _add_global_options(p, suppress=True)
    p.set_defaults(func=cmd_tone)

    p = subs.add_parser("encode", help="a WAV to a picture of its spectrum")
    p.add_argument("input", help="the WAV to read")
    p.add_argument("-o", "--output", default=None, help="where to write the PNG (default: alongside the input)")
    _add_encode_options(p)
    _add_global_options(p, suppress=True)
    p.set_defaults(func=cmd_encode)

    p = subs.add_parser("decode", help="a picture back to a WAV")
    p.add_argument("input", help="the PNG to read")
    p.add_argument("-o", "--output", default=None, help="where to write the WAV (default: alongside the input)")
    p.add_argument("--rate", type=int, default=None, help="sample rate to assume when the picture does not say")
    _add_encode_options(p)
    _add_decode_options(p)
    _add_global_options(p, suppress=True)
    p.set_defaults(func=cmd_decode)

    p = subs.add_parser("roundtrip", help="encode, decode, and measure what survived")
    p.add_argument("input", help="the WAV to read")
    p.add_argument("-o", "--output", default=None, help="where to write the decoded WAV")
    p.add_argument("--image", default=None, help="where to write the picture")
    _add_encode_options(p)
    _add_decode_options(p)
    _add_global_options(p, suppress=True)
    p.set_defaults(func=cmd_roundtrip)

    p = subs.add_parser("view", help="draw a labelled chart of a clip or a picture")
    p.add_argument("input", help="a WAV or a PNG")
    p.add_argument("-o", "--output", default=None, help="where to write the view")
    p.add_argument("--view-colormap", choices=COLORMAPS, default="magma", help="colours for the view (default: magma)")
    p.add_argument("--title", default=None, help="title text (default: the file name and its settings)")
    p.add_argument("--zoom", type=int, default=1, help="enlarge every plane pixel by this much (default: 1)")
    p.add_argument("--view-width", type=int, default=None, help="resize the plane to this many columns first")
    p.add_argument("--view-height", type=int, default=None, help="resize the plane to this many rows first")
    p.add_argument("--no-axes", action="store_true", help="no axes or labels")
    p.add_argument("--no-legend", action="store_true", help="no colour legend")
    p.add_argument("--rate", type=int, default=None, help="sample rate to assume when a picture does not say")
    _add_encode_options(p)
    _add_global_options(p, suppress=True)
    p.set_defaults(func=cmd_view)

    p = subs.add_parser("waveform", help="draw the samples themselves")
    p.add_argument("input", help="the WAV to read")
    p.add_argument("-o", "--output", default=None, help="where to write the picture")
    p.add_argument("--view-width", type=int, default=900, help="plot width (default: 900)")
    p.add_argument("--view-height", type=int, default=220, help="plot height (default: 220)")
    p.add_argument("--title", default=None, help="title text")
    p.add_argument("--no-axes", action="store_true", help="no axes or labels")
    _add_global_options(p, suppress=True)
    p.set_defaults(func=cmd_waveform)

    p = subs.add_parser("serve", help="open the whole tool in a browser")
    p.add_argument("--host", default=DEFAULT_HOST, help=f"interface to listen on (default: {DEFAULT_HOST})")
    p.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"port (default: {DEFAULT_PORT})")
    p.add_argument("--frontend-dir", default=None, help="serve the page from here instead of the packaged copy")
    _add_global_options(p, suppress=True)
    p.set_defaults(func=cmd_serve)

    p = subs.add_parser("info", help="describe a WAV or a picture")
    p.add_argument("input", help="the file to describe")
    _add_global_options(p, suppress=True)
    p.set_defaults(func=cmd_info)
    return parser


def _fill_defaults(args: argparse.Namespace) -> None:
    """Work out the output path when one was not given."""
    command = getattr(args, "command", "")
    if command == "encode" and not args.output:
        args.output = _default_out(args.input, "", ".png")
    elif command == "decode" and not args.output:
        args.output = _default_out(args.input, "-decoded", ".wav")
    elif command in ("view", "waveform") and not args.output:
        args.output = _default_out(args.input, f"-{command}", ".png")


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point.  Returns the process exit code."""
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    if not getattr(args, "command", None):
        parser.print_help()
        return EXIT_OK
    try:
        _fill_defaults(args)
        return int(args.func(args))
    except CliError as exc:
        print(f"{PROG}: error: {exc}", file=sys.stderr)
        return EXIT_ERROR
    except (WavError, PngError) as exc:
        print(f"{PROG}: error: {exc}", file=sys.stderr)
        return EXIT_ERROR
    except KeyboardInterrupt:
        print(f"\n{PROG}: interrupted", file=sys.stderr)
        return EXIT_ABORTED
    except BrokenPipeError:
        # the reader went away ("| head"): say nothing, and keep the interpreter
        # from reporting the same thing again while it flushes stdout at exit
        try:
            devnull = os.open(os.devnull, os.O_WRONLY)
            os.dup2(devnull, sys.stdout.fileno())
        except OSError:  # pragma: no cover - nothing useful to do about it
            pass
        return EXIT_OK
    except _HANDLED as exc:
        print(f"{PROG}: error: {exc}", file=sys.stderr)
        return EXIT_ERROR
