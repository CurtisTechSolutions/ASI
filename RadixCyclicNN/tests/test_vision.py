"""Tests for radixnet.vision - images as text through the Stable Diffusion VAE (or the stand-in) and base64.

The diffusion weights cannot be downloaded in every environment, so the SD
path is exercised with a fake VAE (average pooling in, nearest-neighbour
upsampling out) that has the real interface; the stand-in encoder needs only
Pillow (tests skip without it).
"""

import base64
import io
import json
import os
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from PIL import Image
except ImportError:  # pragma: no cover - the suite skips without Pillow
    Image = None

from radixnet import vision  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def gradient(width=100, height=60):
    image = Image.new("RGB", (width, height))
    for x in range(width):
        for y in range(height):
            image.putpixel((x, y), ((x * 255) // width, (y * 255) // height, 128))
    return image


def png_bytes(image):
    buf = io.BytesIO()
    image.save(buf, "PNG")
    return buf.getvalue()


@unittest.skipIf(Image is None, "Pillow is not installed")
class TestTextFormat(unittest.TestCase):
    def test_pack_and_parse(self):
        text = vision.pack_text("tiny", 32, 32, bytes(range(48)))
        self.assertTrue(text.startswith("img:tiny:32x32:"))
        encoder, width, height, payload, repaired = vision.parse_text(text)
        self.assertEqual((encoder, width, height, payload, repaired), ("tiny", 32, 32, bytes(range(48)), False))

    def test_parse_repairs_junk_whitespace_and_truncation(self):
        text = vision.pack_text("sd", 64, 64, bytes(range(256)))
        cut = text[:-30] + "\n"
        encoder, _, _, payload, repaired = vision.parse_text(cut)
        self.assertEqual(encoder, "sd")
        self.assertTrue(repaired)
        self.assertLess(len(payload), 256)
        self.assertEqual(payload, bytes(range(256))[: len(payload)])
        spaced = text[:20] + " " + text[20:40] + "\n" + text[40:]
        self.assertEqual(vision.parse_text(spaced)[3], bytes(range(256)))
        for bad in ("hello", "img:tiny:32x32", "img:tiny:0x32:AAAA", ""):
            with self.assertRaises(vision.VisionError):
                vision.parse_text(bad)


@unittest.skipIf(Image is None, "Pillow is not installed")
class TestTinyEncoder(unittest.TestCase):
    def test_encode_decode_round_trip(self):
        result = vision.encode_image(png_bytes(gradient()), size=32, encoder="tiny")
        self.assertEqual(result["encoder"], "tiny")
        self.assertEqual((result["width"], result["height"], result["latent_shape"], result["bytes"]), (32, 32, [3, 4, 4], 48))
        self.assertEqual(result["source_size"], [100, 60])
        self.assertTrue(result["text"].startswith("img:tiny:32x32:"))
        self.assertEqual(result["chars"], len(result["text"]))
        decoded = vision.decode_text(result["text"])
        self.assertEqual((decoded["encoder"], decoded["width"], decoded["height"], decoded["repaired"]), ("tiny", 32, 32, False))
        self.assertTrue(decoded["png"].startswith(b"\x89PNG"))
        image = Image.open(io.BytesIO(decoded["png"]))
        self.assertEqual(image.size, (32, 32))
        # the same image always gives the same text
        self.assertEqual(vision.encode_image(png_bytes(gradient()), size=32, encoder="tiny")["text"], result["text"])

    def test_decode_pads_or_truncates_a_predicted_text(self):
        text = vision.encode_image(png_bytes(gradient()), size=32, encoder="tiny")["text"]
        short = vision.decode_text(text[:-24])
        self.assertTrue(short["repaired"])
        self.assertEqual(short["width"], 32)
        long = vision.decode_text(text + "AAAA")
        self.assertTrue(long["repaired"])

    def test_errors(self):
        with self.assertRaises(vision.VisionError):
            vision.encode_image(b"not an image", size=32, encoder="tiny")
        with self.assertRaises(vision.VisionError):
            vision.encode_image(png_bytes(gradient()), size=30, encoder="tiny")  # not a multiple of 8
        with self.assertRaises(vision.VisionError):
            vision.encode_image(png_bytes(gradient()), size=32, encoder="nope")
        with self.assertRaises(vision.VisionError):
            vision.decode_text("img:tiny:12x12:AAAA")  # not a multiple of 8
        with self.assertRaises(vision.VisionError):
            vision.decode_text("img:other:32x32:AAAA")
        with self.assertRaises(vision.VisionError):
            vision.decode_text("hello")

    def test_describe(self):
        info = vision.describe()
        self.assertTrue(info["pillow"])
        self.assertEqual(info["encoders"], ["auto", "sd", "tiny"])
        self.assertIn(info["auto"], ("sd", "tiny"))
        self.assertEqual(info["default_size"], vision.DEFAULT_SIZE)


class _FakeDist:
    def __init__(self, mean):
        self.mean = mean


class _FakeEncoded:
    def __init__(self, mean):
        self.latent_dist = _FakeDist(mean)


class _FakeDecoded:
    def __init__(self, sample):
        self.sample = sample


class _FakeVae:
    """A stand-in for diffusers' AutoencoderKL: average-pool to 4 channels, nearest-neighbour back."""

    class config:  # noqa: N801 - mirrors the diffusers attribute
        scaling_factor = 0.18215

    def encode(self, x):
        import torch

        pooled = torch.nn.functional.avg_pool2d(x, 8)  # (1, 3, h/8, w/8)
        extra = pooled.mean(dim=1, keepdim=True)
        return _FakeEncoded(torch.cat([pooled, extra], dim=1) / self.config.scaling_factor)

    def decode(self, z):
        import torch

        rgb = z[:, :3] * self.config.scaling_factor  # the encoder side divided by it: back to pixel range
        return _FakeDecoded(torch.nn.functional.interpolate(rgb, scale_factor=8, mode="nearest"))


def _torch_available():
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


@unittest.skipIf(Image is None or not _torch_available(), "Pillow and torch are needed")
class TestSDEncoderPlumbing(unittest.TestCase):
    """The quantisation / base64 / decode plumbing around the VAE, with a fake VAE in place of the weights."""

    def fake_encoder(self):
        enc = vision.SDVaeEncoder(model="fake/vae", device="cpu")
        enc.vae = _FakeVae()
        enc.scaling = 0.18215
        return enc

    def test_round_trip_through_a_fake_vae(self):
        enc = self.fake_encoder()
        with mock.patch.object(vision, "_instance", lambda name: enc if name == "sd" else vision.TinyEncoder()):
            result = vision.encode_image(png_bytes(gradient()), size=64, encoder="sd")
            self.assertEqual(result["encoder"], "sd")
            self.assertEqual((result["latent_shape"], result["bytes"]), ([4, 8, 8], 256))
            self.assertTrue(result["text"].startswith("img:sd:64x64:"))
            payload = base64.b64decode(result["text"].split(":", 3)[3])
            self.assertEqual(len(payload), 256)
            decoded = vision.decode_text(result["text"])
            self.assertEqual((decoded["encoder"], decoded["width"], decoded["repaired"]), ("sd", 64, False))
            image = Image.open(io.BytesIO(decoded["png"]))
            self.assertEqual(image.size, (64, 64))
            # a flat image survives the fake VAE almost unchanged (quantisation aside)
            flat = Image.new("RGB", (64, 64), (200, 40, 90))
            back = Image.open(io.BytesIO(vision.decode_text(vision.encode_image(png_bytes(flat), size=64, encoder="sd")["text"])["png"]))
            r, g, b = back.getpixel((10, 10))
            self.assertLess(abs(r - 200) + abs(g - 40) + abs(b - 90), 40)

    def test_missing_weights_fall_back_to_the_stand_in_for_auto(self):
        enc = vision.SDVaeEncoder(model="/nonexistent/vae", device="cpu")
        with mock.patch.object(vision, "_instance", lambda name: enc if name == "sd" else vision.TinyEncoder()):
            with self.assertRaises(vision.VisionError):
                vision.get_encoder("sd")
            self.assertIsNotNone(enc.error)
            self.assertEqual(vision.encode_image(png_bytes(gradient()), size=32, encoder="auto")["encoder"], "tiny")


@unittest.skipIf(Image is None, "Pillow is not installed")
class TestApi(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tests.test_api import start_server

        cls.tmp = tempfile.TemporaryDirectory(prefix="radixnet-vision-api-")
        cls.addClassCleanup(cls.tmp.cleanup)
        cls.client, cls.server, cls.service = start_server(cls.addClassCleanup, upload_dir=os.path.join(cls.tmp.name, "uploads"))
        cls.png = png_bytes(gradient())

    def test_describe_endpoint(self):
        status, data, _ = self.client.get("/api/images")
        self.assertEqual(status, 200)
        self.assertTrue(data["pillow"])
        self.assertIn("text_format", data)

    def test_encode_raw_multipart_json_and_decode(self):
        status, data, _ = self.client.request(
            "POST", "/api/images/encode?size=32&encoder=tiny", raw=self.png, headers={"Content-Type": "image/png"}
        )
        self.assertEqual(status, 200, data)
        self.assertEqual((data["encoder"], data["width"], data["latent_shape"], data["name"]), ("tiny", 32, [3, 4, 4], "image"))
        self.assertTrue(data["text"].startswith("img:tiny:32x32:"))
        self.assertIsNone(data["job"])
        self.assertIsNone(data["upload"])
        text = data["text"]

        boundary = "----radixnet-image"
        body = (
            f"--{boundary}\r\n".encode("ascii")
            + b'Content-Disposition: form-data; name="file"; filename="photo.png"\r\nContent-Type: image/png\r\n\r\n'
            + self.png + f"\r\n--{boundary}--\r\n".encode("ascii")
        )
        status, data, _ = self.client.request(
            "POST", "/api/images/encode?size=32&encoder=tiny", raw=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
        self.assertEqual(status, 200, data)
        self.assertEqual((data["name"], data["text"]), ("photo.png", text))

        status, data, _ = self.client.post(
            "/api/images/encode", {"name": "b64.png", "content_base64": base64.b64encode(self.png).decode("ascii"), "size": 32, "encoder": "tiny"}
        )
        self.assertEqual(status, 200, data)
        self.assertEqual(data["text"], text)

        status, data, _ = self.client.post("/api/images/decode", {"text": text})
        self.assertEqual(status, 200, data)
        self.assertEqual((data["encoder"], data["width"], data["height"], data["repaired"]), ("tiny", 32, 32, False))
        self.assertTrue(base64.b64decode(data["png_base64"]).startswith(b"\x89PNG"))
        status, data, _ = self.client.post("/api/images/decode", {"text": text[:-20]})
        self.assertEqual(status, 200)
        self.assertTrue(data["repaired"])

    def test_encode_can_save_and_train(self):
        from tests.test_api import wait_for_job

        status, data, _ = self.client.request(
            "POST", "/api/images/encode?size=32&encoder=tiny&train=true&save_as=photo.txt&epochs=2&batch_size=8&lr=0.5",
            raw=self.png, headers={"Content-Type": "image/png"},
        )
        self.assertEqual(status, 202, data)
        self.assertEqual(data["upload"]["name"], "photo.txt")
        self.assertEqual(data["job"]["type"], "train")
        job = wait_for_job(self.client)
        self.assertEqual(job["state"], "done", job)
        self.assertEqual(len(job["history"]), 2)
        status, uploads, _ = self.client.get("/api/uploads")
        self.assertEqual([u["name"] for u in uploads["uploads"]], ["photo.txt"])
        status, stats, _ = self.client.get("/api/status")
        self.assertEqual(stats["trained_texts"], 1)
        self.assertEqual(stats["trained_chars"], data["chars"])
        # the model can now continue the encoded text
        status, pred, _ = self.client.post("/api/predict", {"prefix": data["text"][:20], "length": 10})
        self.assertEqual(status, 200)
        self.assertTrue(pred["full_text"].startswith("img:tiny:32x32:"))

    def test_errors(self):
        status, data, _ = self.client.request(
            "POST", "/api/images/encode?encoder=tiny", raw=b"not an image", headers={"Content-Type": "image/png"}
        )
        self.assertEqual(status, 400, data)
        self.assertIn("not a readable image", data["error"])
        status, data, _ = self.client.request(
            "POST", "/api/images/encode?encoder=tiny&size=30", raw=self.png, headers={"Content-Type": "image/png"}
        )
        self.assertEqual(status, 400, data)
        self.assertIn("multiple of 8", data["error"])
        status, data, _ = self.client.post("/api/images/decode", {"text": "hello"})
        self.assertEqual(status, 400, data)
        status, data, _ = self.client.post("/api/images/encode", {"name": "x.png", "content": "text, not bytes"})
        self.assertEqual(status, 400, data)
        self.assertIn("bytes", data["error"])


@unittest.skipIf(Image is None, "Pillow is not installed")
class TestCli(unittest.TestCase):
    def test_encode_decode_and_train(self):
        from tests.test_cli import run_cli, run_json

        with tempfile.TemporaryDirectory(prefix="radixnet-vision-cli-") as tmp:
            source = os.path.join(tmp, "photo.png")
            with open(source, "wb") as fh:
                fh.write(png_bytes(gradient()))
            doc = run_json("image", "encode", source, "--size", 32, "--encoder", "tiny", "--out", os.path.join(tmp, "photo.txt"))
            self.assertEqual((doc["encoder"], doc["width"], doc["latent_shape"]), ("tiny", 32, [3, 4, 4]))
            self.assertTrue(doc["text"].startswith("img:tiny:32x32:"))
            with open(os.path.join(tmp, "photo.txt"), encoding="utf-8") as fh:
                self.assertEqual(fh.read().strip(), doc["text"])
            out = os.path.join(tmp, "back.png")
            doc2 = run_json("image", "decode", "--data", os.path.join(tmp, "photo.txt"), "--out", out)
            self.assertEqual((doc2["encoder"], doc2["width"], doc2["out"]), ("tiny", 32, out))
            self.assertEqual(Image.open(out).size, (32, 32))
            doc3 = run_json("image", "decode", "--text", doc["text"][:-16], "--out", os.path.join(tmp, "cut.png"))
            self.assertTrue(doc3["repaired"])
            model = os.path.join(tmp, "m.json")
            doc4 = run_json("image", "encode", source, "--size", 32, "--encoder", "tiny", "--train", "--epochs", 2, "--batch-size", 8, model=model)
            self.assertEqual(doc4["trained"]["epochs"], 2)
            self.assertTrue(os.path.exists(model))
            info = run_json("image", "info")
            self.assertTrue(info["pillow"])
            proc = run_cli("image", "encode", os.path.join(tmp, "missing.png"), expect=1)
            self.assertIn("not found", proc.stderr)
