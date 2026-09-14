package radixnet

import (
	"bytes"
	"image"
	"image/color"
	"image/png"
	"strings"
	"testing"
)

func gradientPNG(t *testing.T, width, height int) []byte {
	t.Helper()
	img := image.NewRGBA(image.Rect(0, 0, width, height))
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			img.Set(x, y, color.RGBA{uint8(x * 255 / width), uint8(y * 255 / height), 128, 255})
		}
	}
	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		t.Fatalf("png.Encode: %v", err)
	}
	return buf.Bytes()
}

func TestImageTextPackAndParse(t *testing.T) {
	payload := []byte{0, 1, 2, 3, 250, 251}
	text := PackImageText("tiny", 32, 32, payload)
	if !strings.HasPrefix(text, "img:tiny:32x32:") {
		t.Fatalf("text: %q", text)
	}
	encoder, width, height, back, repaired, err := ParseImageText(text)
	if err != nil {
		t.Fatalf("ParseImageText: %v", err)
	}
	if encoder != "tiny" || width != 32 || height != 32 || repaired {
		t.Errorf("parsed %q %d %d repaired=%v", encoder, width, height, repaired)
	}
	if !bytes.Equal(back, payload) {
		t.Errorf("payload: %v", back)
	}
}

func TestImageTextRepairsAPredictedTail(t *testing.T) {
	text := PackImageText("tiny", 32, 32, []byte{1, 2, 3, 4, 5, 6})
	cut := text[:len(text)-3] // a prediction rarely ends on a byte boundary
	_, _, _, payload, repaired, err := ParseImageText(cut)
	if err != nil {
		t.Fatalf("ParseImageText: %v", err)
	}
	if !repaired {
		t.Error("a cut-off tail must be reported as repaired")
	}
	if len(payload) == 0 {
		t.Error("what survived must still decode")
	}
}

func TestNotAnImageText(t *testing.T) {
	if _, _, _, _, _, err := ParseImageText("the cat sat on the mat"); err == nil {
		t.Error("prose is not an encoded image")
	}
	if _, _, _, _, _, err := ParseImageText("img:tiny:0x0:AAAA"); err == nil {
		t.Error("a zero size must be refused")
	}
}

func TestEncodeImageRoundTrips(t *testing.T) {
	encoded, err := EncodeImage(gradientPNG(t, 64, 40), 64, "auto")
	if err != nil {
		t.Fatalf("EncodeImage: %v", err)
	}
	if encoded.Encoder != "tiny" || encoded.Width != 64 || encoded.Height != 64 {
		t.Fatalf("encoded: %+v", encoded)
	}
	if got, want := encoded.Bytes, 3*8*8; got != want {
		t.Errorf("a 64x64 image is a 3 x 8 x 8 latent: got %d bytes, want %d", got, want)
	}
	if encoded.SourceSize[0] != 64 || encoded.SourceSize[1] != 40 {
		t.Errorf("source size: %v", encoded.SourceSize)
	}
	decoded, err := DecodeImageText(encoded.Text, "")
	if err != nil {
		t.Fatalf("DecodeImageText: %v", err)
	}
	if decoded.Width != 64 || decoded.Height != 64 || decoded.Repaired {
		t.Fatalf("decoded: %+v", decoded)
	}
	img, err := png.Decode(bytes.NewReader(decoded.PNG))
	if err != nil {
		t.Fatalf("the decoded PNG must be readable: %v", err)
	}
	if img.Bounds().Dx() != 64 || img.Bounds().Dy() != 64 {
		t.Errorf("bounds: %v", img.Bounds())
	}
}

func TestEncodeImageIsDeterministic(t *testing.T) {
	data := gradientPNG(t, 64, 64)
	first, err := EncodeImage(data, 64, "tiny")
	if err != nil {
		t.Fatalf("EncodeImage: %v", err)
	}
	second, _ := EncodeImage(data, 64, "tiny")
	if first.Text != second.Text {
		t.Error("the same image must always give the same text")
	}
}

func TestEncodeImageRefusesBadInput(t *testing.T) {
	data := gradientPNG(t, 32, 32)
	if _, err := EncodeImage(data, 30, "tiny"); err == nil {
		t.Error("a size that is not a multiple of 8 must be refused")
	}
	if _, err := EncodeImage(data, 64, "sd"); err == nil {
		t.Error("the sd encoder must say it needs the Python side")
	}
	if _, err := EncodeImage(data, 64, "sideways"); err == nil {
		t.Error("an unknown encoder must be refused")
	}
	if _, err := EncodeImage([]byte("not an image"), 64, "tiny"); err == nil {
		t.Error("unreadable bytes must be refused")
	}
}

func TestDecodeFitsAShortOrLongPayload(t *testing.T) {
	short := PackImageText("tiny", 64, 64, []byte{1, 2, 3})
	decoded, err := DecodeImageText(short, "")
	if err != nil {
		t.Fatalf("DecodeImageText: %v", err)
	}
	if !decoded.Repaired || decoded.Bytes != 3*8*8 {
		t.Errorf("a short payload is padded to the latent size: %+v", decoded)
	}
	long := PackImageText("tiny", 64, 64, make([]byte, 3*8*8+50))
	decoded, err = DecodeImageText(long, "")
	if err != nil {
		t.Fatalf("DecodeImageText: %v", err)
	}
	if !decoded.Repaired {
		t.Error("a rambling payload is truncated and reported")
	}
}

func TestDecodeSdTextSaysWhereToGo(t *testing.T) {
	_, err := DecodeImageText(PackImageText("sd", 64, 64, []byte{1, 2, 3}), "")
	if err == nil || !strings.Contains(err.Error(), "Python") {
		t.Errorf("an sd text must name the Python side: %v", err)
	}
}

func TestRepairBase64(t *testing.T) {
	payload, repaired, err := RepairBase64("AAECAwQF")
	if err != nil || repaired || len(payload) != 6 {
		t.Errorf("a clean payload comes back unchanged: %v %v %v", payload, repaired, err)
	}
	if _, repaired, _ := RepairBase64("AAEC AwQF"); !repaired {
		t.Error("whitespace must be reported as a repair")
	}
	if _, repaired, _ := RepairBase64("AAECA"); !repaired {
		t.Error("a dangling character must be dropped and reported")
	}
	if payload, _, err := RepairBase64(""); err != nil || len(payload) != 0 {
		t.Errorf("an empty payload decodes to nothing: %v %v", payload, err)
	}
}

func TestDescribeVision(t *testing.T) {
	info := DescribeVision()
	if info["auto"] != "tiny" || info["sd"] != false {
		t.Errorf("describe: %v", info)
	}
}
