package radixnet

import (
	"bytes"
	"encoding/base64"
	"fmt"
	"image"
	_ "image/gif"  // so load can read whatever the browser sends
	_ "image/jpeg" //
	"image/png"
	"regexp"
	"strconv"
	"strings"
)

// Images as text: the thumbnail encoder, the text format and the decoder.
//
// Stable Diffusion generates an image by decoding a latent - 4 x H/8 x W/8
// numbers - with its VAE, and vision.py runs that backwards.  The VAE needs
// torch and diffusers, so what is here is the other encoder: "tiny", an RGB
// thumbnail at 1/8 of the size (3 channels, one byte each), which is the same
// 8x spatial reduction and the same text format.
//
// One difference from Python is worth stating plainly, because it is the only
// place in this port where the two sides do not produce identical bytes: the
// *pixel reduction* differs.  Python's thumbnail goes through Pillow's Lanczos
// filter; this one uses a box filter, because reproducing Pillow's exact
// coefficients and rounding without Pillow to check against would be a
// guess.  So the same source image gives two different - equally valid - texts.
// Everything around that is shared: each side parses, trains on, predicts and
// decodes the other's texts, and an "sd" text can be parsed, measured and
// compared here too, it just cannot be decoded back into a picture without the
// weights.
//
//	img:tiny:128x128:AAECAwQF...

// Image encoder names; "auto" resolves to "tiny" here, because "sd" needs the
// diffusion weights and therefore the Python side.
var ImageEncoders = []string{"auto", "tiny"}

// DefaultImageSize is the square images are resized to before encoding.
const DefaultImageSize = 128

// MinImageSize and MaxImageSize bound it (and it must be a multiple of 8).
const (
	MinImageSize = 8
	MaxImageSize = 1024
)

const imageHeader = "img"

var imageTextRe = regexp.MustCompile(`(?s)^\s*img:([a-z0-9_]+):(\d+)x(\d+):(.*)$`)

// VisionError is an unreadable image, an unknown encoder or a text that is not
// an encoded image.
type VisionError struct{ Message string }

func (e *VisionError) Error() string { return e.Message }

func visionErrorf(format string, args ...any) error {
	return &VisionError{Message: fmt.Sprintf(format, args...)}
}

// PackImageText is img:<encoder>:<w>x<h>:<base64 of the quantised latent>.
func PackImageText(encoder string, width, height int, payload []byte) string {
	return fmt.Sprintf("%s:%s:%dx%d:%s", imageHeader, encoder, width, height,
		base64.StdEncoding.EncodeToString(payload))
}

// ParseImageText reads (encoder, width, height, payload, repaired) out of an
// encoded-image text.  The base64 is repaired when it is not clean - whitespace,
// stray characters, a missing tail, all of which a predicted text has - and the
// payload is not padded here (the decoder fits it to the latent it needs).
func ParseImageText(text string) (string, int, int, []byte, bool, error) {
	match := imageTextRe.FindStringSubmatch(text)
	if match == nil {
		return "", 0, 0, nil, false, visionErrorf("not an encoded image: expected 'img:<encoder>:<w>x<h>:<base64>'")
	}
	width, _ := strconv.Atoi(match[2])
	height, _ := strconv.Atoi(match[3])
	payload, repaired, err := RepairBase64(match[4])
	if err != nil {
		return "", 0, 0, nil, false, visionErrorf("%v", err)
	}
	if width <= 0 || height <= 0 {
		return "", 0, 0, nil, false, visionErrorf("invalid image size %dx%d", width, height)
	}
	return match[1], width, height, payload, repaired, nil
}

// checkImageSize validates a requested size.
func checkImageSize(size int) (int, error) {
	if size < MinImageSize || size > MaxImageSize || size%8 != 0 {
		return 0, visionErrorf("size must be a multiple of 8 between %d and %d, got %d",
			MinImageSize, MaxImageSize, size)
	}
	return size, nil
}

// NormaliseImageEncoder resolves "auto" and refuses what this side cannot do.
func NormaliseImageEncoder(name string) (string, error) {
	key := strings.ToLower(strings.TrimSpace(name))
	if key == "" || key == "auto" {
		return "tiny", nil
	}
	if key == "tiny" {
		return "tiny", nil
	}
	if key == "sd" {
		return "", visionErrorf("the Stable Diffusion encoder needs torch and diffusers: use the Python side " +
			"for it, or --encoder tiny here")
	}
	return "", visionErrorf("unknown image encoder %q; expected one of: %s", name, strings.Join(ImageEncoders, ", "))
}

// tinyLatentShape is (channels, H/8, W/8) of the thumbnail encoder.
func tinyLatentShape(width, height int) [3]int { return [3]int{3, height / 8, width / 8} }

// EncodedImage is what EncodeImage produces.
type EncodedImage struct {
	Text        string `json:"text"`
	Encoder     string `json:"encoder"`
	Width       int    `json:"width"`
	Height      int    `json:"height"`
	LatentShape []int  `json:"latent_shape"`
	Bytes       int    `json:"bytes"`
	Chars       int    `json:"chars"`
	SourceSize  []int  `json:"source_size"`
}

// EncodeImage turns image bytes (PNG, JPEG, GIF) into the text the network
// trains on: resized to size x size, reduced to a 1/8 RGB thumbnail, one byte
// per channel value, base64.
func EncodeImage(data []byte, size int, encoder string) (*EncodedImage, error) {
	if size == 0 {
		size = DefaultImageSize
	}
	size, err := checkImageSize(size)
	if err != nil {
		return nil, err
	}
	name, err := NormaliseImageEncoder(encoder)
	if err != nil {
		return nil, err
	}
	src, _, err := image.Decode(bytes.NewReader(data))
	if err != nil {
		return nil, visionErrorf("not a readable image: %v", err)
	}
	small := size / 8
	payload := resampleRGB(src, small, small)
	return &EncodedImage{
		Text: PackImageText(name, size, size, payload), Encoder: name, Width: size, Height: size,
		LatentShape: []int{3, small, small}, Bytes: len(payload), Chars: len(PackImageText(name, size, size, payload)),
		SourceSize: []int{src.Bounds().Dx(), src.Bounds().Dy()},
	}, nil
}

// resampleRGB box-filters src down to width x height and returns it as
// interleaved RGB bytes.  A box filter over the source pixels of each target
// cell is what a thumbnail is; it needs no dependencies and is stable, so the
// same image always gives the same text.
func resampleRGB(src image.Image, width, height int) []byte {
	bounds := src.Bounds()
	sw, sh := bounds.Dx(), bounds.Dy()
	out := make([]byte, 0, width*height*3)
	for y := 0; y < height; y++ {
		y0 := bounds.Min.Y + y*sh/height
		y1 := bounds.Min.Y + (y+1)*sh/height
		if y1 <= y0 {
			y1 = y0 + 1
		}
		for x := 0; x < width; x++ {
			x0 := bounds.Min.X + x*sw/width
			x1 := bounds.Min.X + (x+1)*sw/width
			if x1 <= x0 {
				x1 = x0 + 1
			}
			var rs, gs, bs, n uint64
			for py := y0; py < y1; py++ {
				for px := x0; px < x1; px++ {
					r, g, b, _ := src.At(px, py).RGBA() // 16-bit per channel
					rs += uint64(r >> 8)
					gs += uint64(g >> 8)
					bs += uint64(b >> 8)
					n++
				}
			}
			if n == 0 {
				n = 1
			}
			out = append(out, byte(rs/n), byte(gs/n), byte(bs/n))
		}
	}
	return out
}

// DecodedImage is what DecodeImageText produces.
type DecodedImage struct {
	PNG      []byte `json:"-"`
	Encoder  string `json:"encoder"`
	Width    int    `json:"width"`
	Height   int    `json:"height"`
	Bytes    int    `json:"bytes"`
	Repaired bool   `json:"repaired"`
}

// DecodeImageText turns an encoded - or predicted - text back into a PNG.  A
// payload that is too short or too long (a cut-off or rambling prediction) is
// padded with zeros or truncated, and Repaired says so.
func DecodeImageText(text, encoder string) (*DecodedImage, error) {
	headerEncoder, width, height, payload, repaired, err := ParseImageText(text)
	if err != nil {
		return nil, err
	}
	name := encoder
	if strings.TrimSpace(name) == "" {
		name = headerEncoder
	}
	if resolved, err := NormaliseImageEncoder(name); err != nil {
		return nil, err
	} else {
		name = resolved
	}
	if width%8 != 0 || height%8 != 0 {
		return nil, visionErrorf("the image size %dx%d is not a multiple of 8", width, height)
	}
	shape := tinyLatentShape(width, height)
	length := shape[0] * shape[1] * shape[2]
	payload, changed := fitPayload(payload, length)
	small := image.NewRGBA(image.Rect(0, 0, shape[2], shape[1]))
	for i := 0; i < shape[1]*shape[2]; i++ {
		small.Pix[i*4+0] = payload[i*3+0]
		small.Pix[i*4+1] = payload[i*3+1]
		small.Pix[i*4+2] = payload[i*3+2]
		small.Pix[i*4+3] = 255
	}
	full := image.NewRGBA(image.Rect(0, 0, width, height))
	for y := 0; y < height; y++ {
		sy := y * shape[1] / height
		for x := 0; x < width; x++ {
			sx := x * shape[2] / width
			at, to := (sy*shape[2]+sx)*4, (y*width+x)*4
			copy(full.Pix[to:to+4], small.Pix[at:at+4])
		}
	}
	var buf bytes.Buffer
	if err := png.Encode(&buf, full); err != nil {
		return nil, visionErrorf("cannot encode the PNG: %v", err)
	}
	return &DecodedImage{
		PNG: buf.Bytes(), Encoder: name, Width: width, Height: height, Bytes: length,
		Repaired: repaired || changed,
	}, nil
}

// fitPayload pads with zeros or truncates to length; returns (payload, changed).
func fitPayload(payload []byte, length int) ([]byte, bool) {
	if len(payload) == length {
		return payload, false
	}
	if len(payload) > length {
		return payload[:length], true
	}
	return append(append([]byte{}, payload...), make([]byte, length-len(payload))...), true
}

// DescribeVision reports what this side can do with images.
func DescribeVision() map[string]any {
	return map[string]any{
		"engine": "go", "encoders": ImageEncoders, "auto": "tiny", "default_size": DefaultImageSize,
		"sd": false, "resampling": "box",
		"resampling_note": "Python's thumbnail uses Pillow's Lanczos filter and this one a box filter, so the " +
			"same image gives a different (equally valid) text on each side; the format itself is shared, and " +
			"either side reads, trains on and decodes the other's texts",
		"sd_note": "the Stable Diffusion encoder needs torch and diffusers: use the Python server for it " +
			"(the text format is the same, so a model trained on sd texts by either side is read by both)",
		"text_format": imageHeader + ":<encoder>:<w>x<h>:<base64 of one byte per latent number>",
		"formats":     "PNG, JPEG and GIF",
	}
}
