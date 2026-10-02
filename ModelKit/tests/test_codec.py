"""Tests for modelkit.codec - the diffusion codec: the generator run backwards, its predictions corrected step by step.

The diffusion weights cannot be downloaded in every environment, so the
``sd`` process is exercised with a fake UNet, scheduler and VAE that have the
real interfaces (and skips without torch); the ``pyramid`` stand-in needs
only Pillow (the suite skips without it).  The arithmetic coder needs nothing.
"""

import base64
import io
import os
import random
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # ModelKit
# the model the kit is tested against: the RadixCyclicNN checkout beside this one
sys.path.insert(1, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "RadixCyclicNN"))

try:
    from PIL import Image, ImageDraw, ImageFilter
except ImportError:  # pragma: no cover - the suite skips without Pillow
    Image = ImageDraw = ImageFilter = None

from modelkit import codec  # noqa: E402


def test_card(width=96, height=64, seed=0):
    """A picture with something in it: a gradient, a check, circles, lines, text, a little blur and a little grain."""
    rng = random.Random(seed)
    image = Image.new("RGB", (width, height))
    px = image.load()
    for y in range(height):
        for x in range(width):
            px[x, y] = (x * 255 // width, y * 255 // height, 128 + 60 * ((x // 8 + y // 8) % 2))
    draw = ImageDraw.Draw(image)
    for _ in range(6):
        cx, cy, r = rng.randint(0, width), rng.randint(0, height), rng.randint(4, 20)
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=tuple(rng.randint(0, 255) for _ in range(3)), outline=(0, 0, 0))
    for _ in range(4):
        draw.line((rng.randint(0, width), rng.randint(0, height), rng.randint(0, width), rng.randint(0, height)), fill=(255, 255, 255), width=2)
    draw.text((4, 4), "rdc", fill=(0, 0, 0))
    image = image.filter(ImageFilter.GaussianBlur(0.6))
    px = image.load()
    for y in range(height):
        for x in range(width):
            if rng.random() < 0.3:
                n = rng.randint(-12, 12)
                px[x, y] = tuple(max(0, min(255, v + n)) for v in px[x, y])
    return image


def png_bytes(image):
    buf = io.BytesIO()
    image.save(buf, "PNG")
    return buf.getvalue()


class TestArithmeticCoder(unittest.TestCase):
    def test_random_symbols_round_trip(self):
        rng = random.Random(7)
        for trial in range(40):
            count, width = rng.randint(1, 300), rng.randint(1, 17)
            spread = rng.choice([0.2, 1.0, 4.0, 40.0, 5000.0])
            symbols = [int(rng.gauss(0, spread)) for _ in range(count)]
            data = codec._write_symbols(symbols, width)
            self.assertEqual(codec._read_symbols(data, count, width), symbols, trial)

    def test_zeros_are_cheap_and_wild_values_escape(self):
        zeros = [0] * 4096
        self.assertLess(len(codec._write_symbols(zeros, 64)), 40)  # a few bits for four thousand symbols
        wild = [1 << 20, -(1 << 20), codec._MAG_CAP, codec._MAG_CAP + 1, -codec._MAG_CAP, 0, 1, -1]
        self.assertEqual(codec._read_symbols(codec._write_symbols(wild, 3), len(wild), 3), wild)

    def test_a_short_stream_decodes_as_zeros_past_its_end(self):
        data = codec._write_symbols([3, 0, -2], 3)
        self.assertEqual(codec._read_symbols(data, 3, 3), [3, 0, -2])
        self.assertEqual(codec._read_symbols(b"", 5, 5), [0, 0, 0, 0, 0])

    def test_varints(self):
        for n in (0, 1, 127, 128, 300, 1 << 20, (1 << 32) - 1):
            data = codec._varint(n) + b"tail"
            self.assertEqual(codec._read_varint(data, 0), (n, len(data) - 4))
        with self.assertRaises(codec.CodecError):
            codec._read_varint(b"\x80\x80", 0)

    def test_quantiser(self):
        self.assertEqual([codec._quantise(x) for x in (0.0, 0.5, 0.79, 0.81, 1.5, 1.81, -0.81, -1.81)], [0, 0, 0, 1, 1, 2, -1, -2])


@unittest.skipIf(Image is None, "Pillow is not installed")
class TestPyramidProcess(unittest.TestCase):
    def test_sizes_and_blocks(self):
        self.assertEqual(codec.PyramidProcess.sizes(96, 64), [(2, 1), (3, 2), (6, 4), (12, 8), (24, 16), (48, 32), (96, 64)])
        self.assertEqual(codec.PyramidProcess.sizes(1, 1), [(1, 1)])
        self.assertEqual(codec.PyramidProcess.sizes(5, 3), [(2, 1), (3, 2), (5, 3)])
        blocks = codec.PyramidProcess().blocks(96, 64)
        self.assertEqual(len(blocks), 21)
        self.assertEqual((blocks[0].label, blocks[0].count, blocks[0].width), ("2x1/Y", 2, 2))
        self.assertEqual((blocks[-1].label, blocks[-1].count, blocks[-1].width), ("96x64/Cr", 96 * 64, 96))
        self.assertEqual(blocks[-1].scale, codec.CHROMA_STEP)
        self.assertEqual(blocks[-3].scale, 1.0)
        self.assertLess(blocks[0].scale, blocks[-3].scale)  # coarse levels are quantised finer

    def test_round_trip_is_deterministic_and_close(self):
        image = test_card()
        png = png_bytes(image)
        result = codec.encode(png, process="pyramid", step=4.0)
        self.assertEqual((result["process"], result["width"], result["height"], result["step"]), ("pyramid", 96, 64, 4.0))
        self.assertEqual(result["source_size"], [96, 64])
        self.assertEqual(len(result["blocks"]), 21)
        self.assertEqual(result["size"], len(result["bytes"]))
        self.assertAlmostEqual(result["bits_per_pixel"], 8.0 * result["size"] / (96 * 64))
        self.assertTrue(result["bytes"].startswith(codec.MAGIC))
        self.assertLess(result["size"], len(png))
        self.assertEqual(codec.encode(png, process="pyramid", step=4.0)["bytes"], result["bytes"])
        back = codec.decode(result["bytes"])
        self.assertEqual((back["process"], back["width"], back["height"], back["blocks"], back["blocks_decoded"], back["truncated"]),
                         ("pyramid", 96, 64, 21, 21, False))
        self.assertTrue(back["png"].startswith(b"\x89PNG"))
        self.assertEqual(back["image"].size, (96, 64))
        self.assertGreater(codec.psnr(image, back["image"]), 33.0)
        self.assertLess(codec.psnr(image, Image.new("RGB", image.size, (128, 128, 128))), 20.0)

    def test_a_finer_step_costs_more_bytes_and_errs_less(self):
        image = test_card()
        png = png_bytes(image)
        sizes, errors = [], []
        for step in (2.0, 8.0, 32.0):
            result = codec.encode(png, process="pyramid", step=step)
            sizes.append(result["size"])
            errors.append(codec.psnr(image, codec.decode(result["bytes"])["image"]))
        self.assertEqual(sizes, sorted(sizes, reverse=True))
        self.assertEqual(errors, sorted(errors, reverse=True))
        self.assertGreater(errors[0], 38.0)

    def test_progressive_a_cut_off_stream_still_decodes(self):
        image = test_card()
        full = codec.encode(png_bytes(image), process="pyramid", step=4.0)["bytes"]
        quality = []
        for blocks in (3, 9, 15, 21):
            cut = codec.truncate(full, blocks)
            back = codec.decode(cut)
            self.assertEqual(back["blocks_decoded"], blocks)
            self.assertEqual(back["truncated"], blocks < 21)
            self.assertEqual(back["image"].size, (96, 64))
            quality.append(codec.psnr(image, back["image"]))
        self.assertEqual(quality, sorted(quality))  # every block makes it better
        self.assertEqual(codec.truncate(full, 21), full)
        self.assertEqual(len(list(codec.iter_blocks(full))), 21)
        # a stream cut in the middle of a block keeps the complete blocks before it
        middle = full[: len(codec.truncate(full, 10)) + 3]
        back = codec.decode(middle)
        self.assertEqual((back["blocks_decoded"], back["truncated"]), (10, True))
        # the header alone: the generator runs free (a grey picture)
        header_only = codec.truncate(full, 0)
        back = codec.decode(header_only)
        self.assertEqual((back["blocks_decoded"], back["truncated"]), (0, True))
        self.assertEqual(back["image"].getpixel((10, 10)), (128, 128, 128))

    def test_sizes_odd_tiny_and_capped(self):
        for size in ((1, 1), (3, 5), (17, 2)):
            image = test_card(*size)
            result = codec.encode(png_bytes(image), process="pyramid", step=1.0)
            back = codec.decode(result["bytes"])
            self.assertEqual(back["image"].size, size)
            self.assertGreater(codec.psnr(image, back["image"]), 30.0)
        result = codec.encode(png_bytes(test_card(96, 64)), process="pyramid", size=48)
        self.assertEqual((result["width"], result["height"], result["source_size"]), (48, 32, [96, 64]))
        self.assertEqual(codec.decode(result["bytes"])["image"].size, (48, 32))

    def test_text_form(self):
        png = png_bytes(test_card())
        result = codec.encode_text(png, process="pyramid", step=8.0)
        self.assertTrue(result["text"].startswith("img:rdc:96x64:"))
        self.assertEqual(base64.b64decode(result["text"].split(":", 3)[3]), result["bytes"])
        back = codec.decode_text(result["text"])
        self.assertEqual((back["blocks_decoded"], back["repaired"]), (21, False))
        cut = codec.decode_text(result["text"][:-41] + "\n")
        self.assertTrue(cut["repaired"])
        self.assertTrue(cut["truncated"])
        self.assertLess(cut["blocks_decoded"], 21)
        with self.assertRaises(codec.CodecError):
            codec.decode_text("img:tiny:32x32:AAAA")
        with self.assertRaises(codec.CodecError):
            codec.decode_text("hello")

    def test_compare_images(self):
        image = test_card()
        png = png_bytes(image)
        result = codec.encode(png, process="pyramid", step=8.0)
        back = codec.decode(result["bytes"])
        report = codec.compare_images(png, back["image"], result["size"])
        self.assertEqual(report["bytes"], result["size"])
        self.assertGreater(report["psnr"], 28.0)
        if "jpeg" in report:
            self.assertLessEqual(report["jpeg"]["bytes"], max(result["size"], report["jpeg"]["bytes"]))
            self.assertGreater(report["jpeg"]["psnr"], 20.0)
        self.assertEqual(codec.psnr(image, image), float("inf"))
        with self.assertRaises(codec.CodecError):
            codec.psnr(image, Image.new("RGB", (2, 2)))

    def test_errors(self):
        png = png_bytes(test_card())
        with self.assertRaises(codec.CodecError):
            codec.encode(b"not an image", process="pyramid")
        with self.assertRaises(codec.CodecError):
            codec.encode(png, process="nope")
        with self.assertRaises(codec.CodecError):
            codec.encode(png, process="pyramid", step=0)
        with self.assertRaises(codec.CodecError):
            codec.encode(png, process="pyramid", size=0)
        with self.assertRaises(codec.CodecError):
            codec.decode(b"PNG\x01 nope")
        with self.assertRaises(codec.CodecError):
            codec.decode(codec.MAGIC + b"\x00\x05" + b"{}")  # truncated header
        with self.assertRaises(codec.CodecError):
            codec.decode(codec.MAGIC + b"\x00\x02" + b"{}")  # a header without the fields
        with self.assertRaises(codec.CodecError):
            codec.read_header(b"")
        good = codec.encode(png, process="pyramid")["bytes"]
        with self.assertRaises(codec.CodecError):
            codec.decode(good, process="nope")

    def test_describe(self):
        info = codec.describe()
        self.assertTrue(info["pillow"])
        self.assertEqual(info["processes"], ["auto", "sd", "pyramid"])
        self.assertIn(info["auto"], ("sd", "pyramid"))
        self.assertEqual(info["default_step"], codec.DEFAULT_STEP)
        self.assertIn("rdc", info["text_format"])

    def test_auto_is_the_stand_in_without_the_diffusion_model(self):
        with mock.patch.object(codec.DiffusionProcess, "importable", staticmethod(lambda: False)):
            self.assertIs(codec.get_process("auto"), codec.get_process("pyramid"))


def _torch_available():
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


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

        pooled = torch.nn.functional.avg_pool2d(x, 8)
        extra = pooled.mean(dim=1, keepdim=True)
        return _FakeEncoded(torch.cat([pooled, extra], dim=1) / self.config.scaling_factor)

    def decode(self, z):
        import torch

        rgb = z[:, :3] * self.config.scaling_factor
        return _FakeDecoded(torch.nn.functional.interpolate(rgb, scale_factor=8, mode="nearest"))


class _FakeOutput:
    def __init__(self, sample):
        self.sample = sample


class _FakeUnet:
    """A stand-in for the UNet: predicts the noise as a blurred version of its input (a weak but real prior)."""

    class config:  # noqa: N801
        cross_attention_dim = 768

    def __call__(self, x, t, encoder_hidden_states=None):
        import torch

        assert encoder_hidden_states is not None and tuple(encoder_hidden_states.shape[1:]) == (77, 768)
        blurred = torch.nn.functional.avg_pool2d(x, 3, stride=1, padding=1, count_include_pad=False)
        return _FakeOutput(x - 0.5 * blurred)


class _FakeScheduler:
    """A stand-in for DDIMScheduler: a linear beta schedule, leading spacing, epsilon prediction."""

    class config:  # noqa: N801
        prediction_type = "epsilon"

    def __init__(self, train_steps=1000):
        import torch

        betas = torch.linspace(0.00085, 0.012, train_steps)
        self.alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)
        self.final_alpha_cumprod = torch.tensor(1.0)
        self.train_steps = train_steps
        self.timesteps = None

    def set_timesteps(self, steps):
        import torch

        stride = self.train_steps // steps
        self.timesteps = torch.tensor([i * stride + 1 for i in range(steps)][::-1])


@unittest.skipIf(Image is None or not _torch_available(), "Pillow and torch are needed")
class TestDiffusionProcessPlumbing(unittest.TestCase):
    """The DDIM steps, the corrections and the stream around the model, with fakes in place of the weights."""

    def fake_process(self, steps=4, prompt=""):
        from modelkit import vision

        proc = codec.DiffusionProcess(model="fake/sd", device="cpu", steps=steps, prompt=prompt)
        vae = vision.SDVaeEncoder(model="fake/vae", device="cpu")
        vae.vae = _FakeVae()
        vae.scaling = 0.18215
        proc.vae = vae
        proc.unet = _FakeUnet()
        proc.scheduler = _FakeScheduler()
        return proc

    def test_blocks_follow_the_noise_schedule(self):
        proc = self.fake_process(steps=4)
        proc._step_hint = 4.0
        blocks = proc.blocks(64, 48)
        self.assertEqual([b.label for b in blocks], ["t=751", "t=501", "t=251", "t=1"])
        self.assertEqual([(b.count, b.width) for b in blocks], [(4 * 8 * 6, 8)] * 4)
        scales = [b.scale for b in blocks]
        self.assertEqual(scales, sorted(scales, reverse=True))  # coarse first ...
        self.assertEqual(scales[-1], 1.0)  # ... and the last block at the step itself
        self.assertGreater(scales[0], 5.0)  # the first block is quantised many times coarser

    def test_round_trip_and_the_header(self):
        proc = self.fake_process(steps=4, prompt="a test card")
        image = test_card(64, 48)
        with mock.patch.object(codec, "_instance", lambda name: proc if name == "sd" else codec.PyramidProcess()):
            result = codec.encode(png_bytes(image), process="sd", step=2.0, steps=4, prompt="a test card", seed=3)
            self.assertEqual((result["process"], result["width"], result["height"], result["steps"], result["seed"], result["prompt"]),
                             ("sd", 64, 48, 4, 3, "a test card"))
            self.assertEqual(result["model"], "fake/sd")
            self.assertEqual(len(result["blocks"]), 4)
            header, _ = codec.read_header(result["bytes"])
            self.assertEqual((header["process"], header["steps"], header["seed"], header["prompt"]), ("sd", 4, 3, "a test card"))
            self.assertEqual(codec.encode(png_bytes(image), process="sd", step=2.0, steps=4, prompt="a test card", seed=3)["bytes"], result["bytes"])
            back = codec.decode(result["bytes"])
            self.assertEqual((back["process"], back["blocks"], back["blocks_decoded"], back["truncated"], back["prompt"]),
                             ("sd", 4, 4, False, "a test card"))
            self.assertEqual(back["image"].size, (64, 48))
            # the fake VAE keeps the 8x8 means: a flat picture comes back flat
            flat = Image.new("RGB", (64, 48), (200, 40, 90))
            fine = codec.encode(png_bytes(flat), process="sd", step=0.5, steps=4)
            r, g, b = codec.decode(fine["bytes"])["image"].getpixel((10, 10))
            self.assertLess(abs(r - 200) + abs(g - 40) + abs(b - 90), 40)
            # a finer step is more bytes and a closer picture
            coarse = codec.encode(png_bytes(image), process="sd", step=8.0, steps=4)
            self.assertLess(coarse["size"], result["size"])
            self.assertGreater(codec.psnr(image, codec.decode(result["bytes"])["image"]),
                               codec.psnr(image, codec.decode(coarse["bytes"])["image"]))
            # cut off after two blocks: the generator runs the other two free
            cut = codec.decode(codec.truncate(result["bytes"], 2))
            self.assertEqual((cut["blocks_decoded"], cut["truncated"]), (2, True))

    def test_sizes_are_multiples_of_eight(self):
        proc = self.fake_process()
        self.assertEqual(proc.coded_size(100, 60, None), (96, 56))
        self.assertEqual(proc.coded_size(100, 60, 48), (48, 24))
        self.assertEqual(proc.coded_size(3, 3, None), (8, 8))
        with mock.patch.object(codec, "_instance", lambda name: proc if name == "sd" else codec.PyramidProcess()):
            result = codec.encode(png_bytes(test_card(100, 60)), process="sd", steps=2)
            self.assertEqual((result["width"], result["height"], result["source_size"]), (96, 56, [100, 60]))

    def test_missing_weights_fall_back_to_the_stand_in_for_auto(self):
        proc = codec.DiffusionProcess(model="/nonexistent/sd", device="cpu")
        with mock.patch.object(codec, "_instance", lambda name: proc if name == "sd" else codec.PyramidProcess()):
            with self.assertRaises(codec.CodecError):
                codec.get_process("sd")
            self.assertIsNotNone(proc.error)
            self.assertEqual(codec.encode(png_bytes(test_card()), process="auto")["process"], "pyramid")


@unittest.skipIf(Image is None, "Pillow is not installed")
class TestApi(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tests.test_api import start_server

        cls.tmp = tempfile.TemporaryDirectory(prefix="radixnet-codec-api-")
        cls.addClassCleanup(cls.tmp.cleanup)
        cls.client, cls.server, cls.service = start_server(cls.addClassCleanup, upload_dir=os.path.join(cls.tmp.name, "uploads"))
        cls.image = test_card()
        cls.png = png_bytes(cls.image)

    def test_describe_endpoint(self):
        status, data, _ = self.client.get("/api/images/codec")
        self.assertEqual(status, 200)
        self.assertTrue(data["pillow"])
        self.assertEqual(data["processes"], ["auto", "sd", "pyramid"])

    def test_compress_raw_multipart_json_and_decompress(self):
        status, data, _ = self.client.request(
            "POST", "/api/images/compress?process=pyramid&step=8", raw=self.png, headers={"Content-Type": "image/png"}
        )
        self.assertEqual(status, 200, data)
        self.assertEqual((data["process"], data["width"], data["height"], data["step"], data["name"]), ("pyramid", 96, 64, 8.0, "image"))
        self.assertTrue(data["text"].startswith("img:rdc:96x64:"))
        blob = base64.b64decode(data["content_base64"])
        self.assertEqual(len(blob), data["size"])
        self.assertTrue(blob.startswith(codec.MAGIC))

        boundary = "----radixnet-codec"
        body = (
            f"--{boundary}\r\n".encode("ascii")
            + b'Content-Disposition: form-data; name="file"; filename="card.png"\r\nContent-Type: image/png\r\n\r\n'
            + self.png + f"\r\n--{boundary}--\r\n".encode("ascii")
        )
        status, data2, _ = self.client.request(
            "POST", "/api/images/compress?process=pyramid&step=8", raw=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
        self.assertEqual(status, 200, data2)
        self.assertEqual((data2["name"], data2["content_base64"]), ("card.png", data["content_base64"]))

        status, data3, _ = self.client.post(
            "/api/images/compress", {"name": "b64.png", "content_base64": base64.b64encode(self.png).decode("ascii"), "process": "pyramid", "step": 8}
        )
        self.assertEqual(status, 200, data3)
        self.assertEqual(data3["content_base64"], data["content_base64"])

        status, back, _ = self.client.post("/api/images/decompress", {"content_base64": data["content_base64"]})
        self.assertEqual(status, 200, back)
        self.assertEqual((back["process"], back["width"], back["height"], back["blocks_decoded"], back["truncated"]), ("pyramid", 96, 64, 21, False))
        png = base64.b64decode(back["png_base64"])
        self.assertTrue(png.startswith(b"\x89PNG"))
        self.assertGreater(codec.psnr(self.image, Image.open(io.BytesIO(png))), 28.0)
        status, back2, _ = self.client.post("/api/images/decompress", {"text": data["text"]})
        self.assertEqual(status, 200, back2)
        self.assertEqual(back2["png_base64"], back["png_base64"])
        status, back3, _ = self.client.post("/api/images/decompress", {"text": data["text"][:200]})
        self.assertEqual(status, 200, back3)
        self.assertTrue(back3["truncated"])

    def test_errors(self):
        status, data, _ = self.client.request(
            "POST", "/api/images/compress?process=pyramid", raw=b"not an image", headers={"Content-Type": "image/png"}
        )
        self.assertEqual(status, 400, data)
        self.assertIn("not a readable image", data["error"])
        status, data, _ = self.client.request(
            "POST", "/api/images/compress?process=pyramid&step=zero", raw=self.png, headers={"Content-Type": "image/png"}
        )
        self.assertEqual(status, 400, data)
        status, data, _ = self.client.post("/api/images/decompress", {"content_base64": "AAAA"})
        self.assertEqual(status, 400, data)
        self.assertIn("rdc", data["error"])
        status, data, _ = self.client.post("/api/images/decompress", {"text": "hello"})
        self.assertEqual(status, 400, data)
        status, data, _ = self.client.post("/api/images/decompress", {})
        self.assertEqual(status, 400, data)


@unittest.skipIf(Image is None, "Pillow is not installed")
class TestCli(unittest.TestCase):
    def test_compress_decompress_and_compare(self):
        from tests.test_cli import run_cli, run_json

        with tempfile.TemporaryDirectory(prefix="radixnet-codec-cli-") as tmp:
            image = test_card()
            source = os.path.join(tmp, "card.png")
            image.save(source)
            rdc, text = os.path.join(tmp, "card.rdc"), os.path.join(tmp, "card.txt")
            doc = run_json("image", "compress", source, "--process", "pyramid", "--step", 8, "--out", rdc, "--text-out", text, "--compare")
            self.assertEqual((doc["process"], doc["width"], doc["height"], doc["out"], doc["text_out"]), ("pyramid", 96, 64, rdc, text))
            self.assertGreater(doc["compare"]["psnr"], 28.0)
            self.assertEqual(os.path.getsize(rdc), doc["size"])
            with open(text, encoding="utf-8") as fh:
                self.assertTrue(fh.read().startswith("img:rdc:96x64:"))
            out = os.path.join(tmp, "back.png")
            doc2 = run_json("image", "decompress", rdc, "--out", out)
            self.assertEqual((doc2["process"], doc2["blocks_decoded"], doc2["truncated"], doc2["out"]), ("pyramid", 21, False, out))
            self.assertGreater(codec.psnr(image, Image.open(out)), 28.0)
            doc3 = run_json("image", "decompress", text, "--out", os.path.join(tmp, "back2.png"))
            self.assertEqual(doc3["blocks_decoded"], 21)
            with open(text, encoding="utf-8") as fh:
                cut = fh.read()[:300]
            doc4 = run_json("image", "decompress", "--text", cut, "--out", os.path.join(tmp, "cut.png"))
            self.assertTrue(doc4["truncated"])
            proc = run_cli("image", "compress", source, "--process", "pyramid", "--step", 16, json_mode=False)
            self.assertIn("img:rdc:96x64:", proc.stdout)
            info = run_json("image", "codec")
            self.assertTrue(info["pillow"])
            proc = run_cli("image", "compress", os.path.join(tmp, "missing.png"), expect=1)
            self.assertIn("not found", proc.stderr)
            proc = run_cli("image", "decompress", source, "--out", os.path.join(tmp, "x.png"), expect=1)
            self.assertIn("rdc", proc.stderr)


if __name__ == "__main__":
    unittest.main()
