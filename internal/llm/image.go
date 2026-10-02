package llm

import (
	"bytes"
	"encoding/base64"
	"errors"
	"fmt"
	"image"
	_ "image/gif"  // register the decoder so DecodeConfig can size a GIF
	_ "image/jpeg" // ditto JPEG
	_ "image/png"  // ditto PNG
	"io"
	"net/http"
	"os"
	"strings"
)

// MaxImageBytes caps every image kloo reads, whether from --image or view_image.
// Same figure as read_file: every file read is size-capped (an uncapped read once
// OOM-killed kloo), and an image is inflated ~4/3 by base64 on top, so a 5 MiB file
// is already a ~7 MB request body.
const MaxImageBytes = 5 << 20

// ImagePart is one image attached to a user message. It is held as raw bytes and
// only encoded to a data URL when the request is serialized, so the transcript
// never carries a megabyte-sized string through the token estimator.
type ImagePart struct {
	// Source names where the image came from (the --image path or the workspace
	// path given to view_image). Provenance only: it is reported in the --json
	// result and in compaction stubs, never sent to the endpoint.
	Source string
	MIME   string
	Data   []byte
	// Width/Height are 0 when the format has no stdlib decoder (WebP); the token
	// estimate then falls back to its conservative ceiling.
	Width, Height int
}

// imageMIMEs are the formats the OpenAI-compatible image_url part is specified
// for. Anything else (SVG, BMP, a PDF renamed .png) is refused up front: sending
// it would cost a request and come back as an opaque 4xx.
var imageMIMEs = map[string]bool{"image/png": true, "image/jpeg": true, "image/gif": true, "image/webp": true}

// NewImagePart validates data as a supported image and sizes it. The MIME type is
// sniffed from the bytes, not taken from the file extension: a mislabelled file is
// exactly the case where trusting the name produces a request the endpoint rejects.
func NewImagePart(source string, data []byte) (ImagePart, error) {
	if len(data) == 0 {
		return ImagePart{}, fmt.Errorf("image %s: file is empty", source)
	}
	if len(data) > MaxImageBytes {
		return ImagePart{}, fmt.Errorf("image %s: %d bytes is over the %d-byte image cap", source, len(data), MaxImageBytes)
	}
	mime := http.DetectContentType(data)
	if !imageMIMEs[mime] {
		return ImagePart{}, fmt.Errorf("image %s: unsupported type %s (want png, jpeg, gif or webp)", source, mime)
	}
	p := ImagePart{Source: source, MIME: mime, Data: data}
	if cfg, _, err := image.DecodeConfig(bytes.NewReader(data)); err == nil {
		p.Width, p.Height = cfg.Width, cfg.Height
	}
	return p, nil
}

// ReadImageFile reads an image from an absolute or caller-resolved path under the
// MaxImageBytes cap. The cap is checked on the bytes actually read, not only on a
// Stat, so a file that grows between the two cannot slip past it.
func ReadImageFile(source, path string) (ImagePart, error) {
	f, err := os.Open(path)
	if err != nil {
		return ImagePart{}, fmt.Errorf("image %s: %w", source, err)
	}
	defer f.Close()
	if info, err := f.Stat(); err == nil {
		if info.IsDir() {
			return ImagePart{}, fmt.Errorf("image %s: is a directory", source)
		}
		if info.Size() > MaxImageBytes {
			return ImagePart{}, fmt.Errorf("image %s: %d bytes is over the %d-byte image cap", source, info.Size(), MaxImageBytes)
		}
	}
	data, err := io.ReadAll(io.LimitReader(f, MaxImageBytes+1))
	if err != nil {
		return ImagePart{}, fmt.Errorf("image %s: %w", source, err)
	}
	return NewImagePart(source, data)
}

// DataURL is the base64 data URL the image_url part carries.
func (p ImagePart) DataURL() string {
	return "data:" + p.MIME + ";base64," + base64.StdEncoding.EncodeToString(p.Data)
}

// Image token accounting. There is no shared tokenizer for pixels any more than
// for text: Qwen-VL charges one token per 28×28 patch, OpenAI charges per 512px
// tile, and every server resizes to its own pixel ceiling first. So the estimate
// is the patch count, clamped to the ceiling a typical local VLM resizes down to,
// plus a small framing overhead. It errs high on purpose — undercounting is what
// overflows a real window; overcounting only compacts a little early.
const (
	imagePatchPixels = 28 * 28
	imageMinTokens   = 64
	imageMaxTokens   = 1280 // ≈ Qwen2.5-VL's default max_pixels / patch area
	imageFrameTokens = 8    // vision start/end markers and the part wrapper
)

// EstimateTokens is the estimated prompt cost of one image.
func (p ImagePart) EstimateTokens() int {
	if p.Width <= 0 || p.Height <= 0 {
		return imageMaxTokens + imageFrameTokens
	}
	n := (p.Width*p.Height + imagePatchPixels - 1) / imagePatchPixels
	n = max(imageMinTokens, min(n, imageMaxTokens))
	return n + imageFrameTokens
}

// ImageTokens is the estimated prompt cost of a message's images (0 when none).
func ImageTokens(images []ImagePart) int {
	n := 0
	for _, p := range images {
		n += p.EstimateTokens()
	}
	return n
}

// CountImages reports how many image parts msgs carry.
func CountImages(msgs []Message) int {
	n := 0
	for _, m := range msgs {
		n += len(m.Images)
	}
	return n
}

// ErrImageRejected marks an endpoint refusing a request because it carried image
// parts. It is deterministic — the same request will be refused again — so the
// retry ladder must never retry it.
var ErrImageRejected = errors.New("endpoint rejected image input")

// ImageRejectedError is the loud failure for a text-only endpoint. Without it the
// run would end on an opaque "api error 400" (or, worse, a 500 the retry ladder
// burns its budget on) with nothing pointing at the image as the cause.
type ImageRejectedError struct {
	Images int
	Model  string
	API    *APIError
}

func (e *ImageRejectedError) Error() string {
	return fmt.Sprintf("llm: endpoint rejected a request carrying %d image part(s) — model %q appears to be text-only (%s). "+
		"Drop --image / --view-image for this model, or mark it \"textOnly\": true in the profile so kloo refuses images up front",
		e.Images, e.Model, firstLineOf(e.API.Error()))
}

func (e *ImageRejectedError) Unwrap() []error { return []error{ErrImageRejected, e.API} }

// mentionsImageRejection reports whether an error body blames the image input.
// Wording varies by server ("image input is not supported", "does not support
// vision", "multimodal", "unknown content type image_url"), so match the nouns.
func mentionsImageRejection(body string) bool {
	s := strings.ToLower(body)
	for _, needle := range []string{"image", "vision", "multimodal", "multi-modal", "mmproj"} {
		if strings.Contains(s, needle) {
			return true
		}
	}
	return false
}

// classifyAPIError upgrades an APIError on a request that carried images into an
// ImageRejectedError when the body blames the images. A request with no images,
// or a failure about something else (auth, context overflow), is returned as is.
func (c *Client) classifyAPIError(req ChatRequest, apiErr *APIError) error {
	n := CountImages(req.Messages)
	if n == 0 || !mentionsImageRejection(apiErr.Body) {
		return apiErr
	}
	model := req.Model
	if model == "" {
		model = c.model
	}
	return &ImageRejectedError{Images: n, Model: model, API: apiErr}
}
