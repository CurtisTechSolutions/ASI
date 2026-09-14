"""audioimage - a waveform encoded into a picture, and decoded back out of it.

The plane is the point: **X is time, Y is frequency, and the darkness of a
point is how much of that frequency is sounding at that moment** - black is
1.0, white is 0.0.  Every part of the package serves that one mapping.

    from audioimage import Audio, EncodeConfig, decode, encode, read_wav

    audio = read_wav("voice.wav")
    picture = encode(audio)          # -> a PNG image of the spectrum
    back = decode(picture).audio     # -> a waveform again

Standard library only.  numpy is used when it is installed, for the transforms
(:mod:`audioimage.backend`) and for reading colours back out of a picture; the
same code runs without it, more slowly and with identical results.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .backend import BACKENDS, describe_backends, get_backend, numpy_available
from .colormap import COLORMAPS, GRAY, ColorMap, NearestColor, get_colormap
from .codec import (
    META_KEY,
    DecodeResult,
    EncodeConfig,
    compare,
    decode,
    decode_file,
    encode,
    encode_file,
    image_to_plane,
    plane_to_image,
    read_meta,
)
from .dsp import WINDOWS, griffin_lim, istft, magnitudes, rfft, irfft, stft, window_values
from .png import Image, PngError, read_png, write_png
from .spectrogram import (
    FREQ_SCALES,
    ORIGINS,
    SCALES,
    PlaneConfig,
    Spectrogram,
    from_plane,
    hz_to_mel,
    mel_to_hz,
    to_plane,
)
from .wav import Audio, WavError, read_wav, write_wav

__all__ = [
    "__version__",
    "Audio", "WavError", "read_wav", "write_wav",
    "Image", "PngError", "read_png", "write_png",
    "EncodeConfig", "DecodeResult", "encode", "decode", "encode_file", "decode_file",
    "compare", "read_meta", "plane_to_image", "image_to_plane", "META_KEY",
    "Spectrogram", "PlaneConfig", "to_plane", "from_plane",
    "SCALES", "FREQ_SCALES", "ORIGINS", "hz_to_mel", "mel_to_hz",
    "ColorMap", "NearestColor", "get_colormap", "COLORMAPS", "GRAY",
    "WINDOWS", "stft", "istft", "rfft", "irfft", "magnitudes", "griffin_lim", "window_values",
    "BACKENDS", "get_backend", "describe_backends", "numpy_available",
]
