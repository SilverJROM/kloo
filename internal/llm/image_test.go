package llm

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"image"
	"image/color"
	"image/png"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// testPNG returns a real w×h PNG, so the MIME sniff and DecodeConfig paths run
// against genuine bytes rather than a stub.
func testPNG(t *testing.T, w, h int) []byte {
	t.Helper()
	img := image.NewRGBA(image.Rect(0, 0, w, h))
	for x := 0; x < w; x++ {
		img.Set(x, 0, color.RGBA{R: 255, A: 255})
	}
	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		t.Fatal(err)
	}
	return buf.Bytes()
}

func testImage(t *testing.T) ImagePart {
	t.Helper()
	p, err := NewImagePart("ref.png", testPNG(t, 4, 2))
	if err != nil {
		t.Fatal(err)
	}
	return p
}

// wireContent marshals m and returns its "content" field decoded generically.
func wireContent(t *testing.T, m Message) any {
	t.Helper()
	b, err := json.Marshal(m)
	if err != nil {
		t.Fatal(err)
	}
	var got map[string]any
	if err := json.Unmarshal(b, &got); err != nil {
		t.Fatal(err)
	}
	return got["content"]
}

func TestMessageWithImagesSerializesAsParts(t *testing.T) {
	img := testImage(t)
	wantURL := "data:image/png;base64," + base64.StdEncoding.EncodeToString(img.Data)

	t.Run("text part first, then one image_url part per image", func(t *testing.T) {
		parts, ok := wireContent(t, Message{Role: RoleUser, Content: "describe it", Images: []ImagePart{img, img}}).([]any)
		if !ok || len(parts) != 3 {
			t.Fatalf("content = %#v, want 3 parts", parts)
		}
		text := parts[0].(map[string]any)
		if text["type"] != "text" || text["text"] != "describe it" {
			t.Errorf("first part = %v, want the task text", text)
		}
		for _, p := range parts[1:] {
			ip := p.(map[string]any)
			if ip["type"] != "image_url" {
				t.Errorf("part type = %v, want image_url", ip["type"])
			}
			if url := ip["image_url"].(map[string]any)["url"]; url != wantURL {
				t.Errorf("url = %.60v…, want the base64 data URL", url)
			}
		}
	})

	t.Run("image-only message has no empty text part", func(t *testing.T) {
		parts := wireContent(t, Message{Role: RoleUser, Images: []ImagePart{img}}).([]any)
		if len(parts) != 1 || parts[0].(map[string]any)["type"] != "image_url" {
			t.Errorf("content = %v, want a single image part", parts)
		}
	})

	t.Run("cache breakpoint lands on the last part", func(t *testing.T) {
		parts := wireContent(t, Message{Role: RoleUser, Content: "x", Images: []ImagePart{img}, CacheControl: CacheControlEphemeral()}).([]any)
		if _, ok := parts[0].(map[string]any)["cache_control"]; ok {
			t.Errorf("text part carries cache_control; it belongs on the last part")
		}
		if _, ok := parts[1].(map[string]any)["cache_control"]; !ok {
			t.Errorf("last (image) part lacks cache_control")
		}
	})

	t.Run("message without images is still a plain string", func(t *testing.T) {
		if c := wireContent(t, Message{Role: RoleUser, Content: "hi"}); c != "hi" {
			t.Errorf("content = %#v, want plain string", c)
		}
	})

	t.Run("the raw bytes never leak as a JSON key", func(t *testing.T) {
		b, _ := json.Marshal(Message{Role: RoleUser, Content: "x", Images: []ImagePart{img}})
		if strings.Contains(string(b), `"Images"`) || strings.Contains(string(b), `"Source"`) {
			t.Errorf("serialized message leaks ImagePart fields: %s", b)
		}
	})
}

func TestNormalizeMergeKeepsImages(t *testing.T) {
	img := testImage(t)
	in := []Message{
		{Role: RoleUser, Content: "task", Images: []ImagePart{img}},
		{Role: RoleUser, Content: "more", Images: []ImagePart{img}},
	}
	out := normalizeMessages(in)
	if len(out) != 1 || len(out[0].Images) != 2 {
		t.Fatalf("merged = %d msgs / %d images, want 1 / 2", len(out), len(out[0].Images))
	}
	if len(in[0].Images) != 1 {
		t.Errorf("merge mutated the caller's slice: %d images", len(in[0].Images))
	}
}

func TestNewImagePartValidation(t *testing.T) {
	cases := []struct {
		name    string
		data    []byte
		wantErr string
	}{
		{"png accepted and sized", testPNG(t, 30, 20), ""},
		{"empty file refused", nil, "empty"},
		{"non-image refused by content, not name", []byte("<svg xmlns='http://www.w3.org/2000/svg'/>"), "unsupported type"},
		{"over the cap refused", append(testPNG(t, 2, 2), make([]byte, MaxImageBytes)...), "image cap"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			p, err := NewImagePart("x.png", tc.data)
			if tc.wantErr != "" {
				if err == nil || !strings.Contains(err.Error(), tc.wantErr) {
					t.Fatalf("err = %v, want containing %q", err, tc.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			if p.MIME != "image/png" || p.Width != 30 || p.Height != 20 {
				t.Errorf("part = %s %dx%d, want image/png 30x20", p.MIME, p.Width, p.Height)
			}
		})
	}
}

func TestReadImageFileCap(t *testing.T) {
	dir := t.TempDir()
	big := filepath.Join(dir, "big.png")
	if err := os.WriteFile(big, append(testPNG(t, 2, 2), make([]byte, MaxImageBytes)...), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := ReadImageFile("big.png", big); err == nil || !strings.Contains(err.Error(), "image cap") {
		t.Errorf("err = %v, want the size cap", err)
	}
	if _, err := ReadImageFile("dir", dir); err == nil || !strings.Contains(err.Error(), "directory") {
		t.Errorf("err = %v, want a directory refusal", err)
	}
	if _, err := ReadImageFile("missing.png", filepath.Join(dir, "missing.png")); !errors.Is(err, os.ErrNotExist) {
		t.Errorf("err = %v, want ErrNotExist", err)
	}
}

func TestImageTokenEstimate(t *testing.T) {
	cases := []struct {
		name string
		w, h int
		want int
	}{
		{"tiny image hits the floor", 10, 10, imageMinTokens + imageFrameTokens},
		{"mid image counts 28px patches", 280, 280, 100 + imageFrameTokens},
		{"huge image clamps to the resize ceiling", 4000, 4000, imageMaxTokens + imageFrameTokens},
		{"unknown size assumes the ceiling", 0, 0, imageMaxTokens + imageFrameTokens},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := (ImagePart{Width: tc.w, Height: tc.h}).EstimateTokens(); got != tc.want {
				t.Errorf("tokens = %d, want %d", got, tc.want)
			}
		})
	}
}

// A text-only endpoint must fail LOUDLY and attributably, not as a bare 400 and
// never as a retryable 500.
func TestImageRejectionIsLoud(t *testing.T) {
	cases := []struct {
		name         string
		status       int
		body         string
		images       bool
		wantRejected bool
	}{
		{"llama.cpp without mmproj (500)", 500, `{"error":{"message":"image input is not supported - hint: if this is unexpected, you may need to provide the mmproj"}}`, true, true},
		{"vision-less model (400)", 400, `{"error":"model does not support vision"}`, true, true},
		{"unrelated 400 with images stays an APIError", 400, `{"error":"invalid api key"}`, true, false},
		{"image wording without images is not blamed on images", 400, `{"error":"image input is not supported"}`, false, false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				_, _ = io.Copy(io.Discard, r.Body)
				w.WriteHeader(tc.status)
				_, _ = w.Write([]byte(tc.body))
			}))
			defer srv.Close()
			msg := Message{Role: RoleUser, Content: "describe"}
			if tc.images {
				msg.Images = []ImagePart{testImage(t)}
			}
			c := New(srv.URL+"/v1", "text-model")
			for _, call := range []func() error{
				func() error {
					_, err := c.Complete(context.Background(), ChatRequest{Messages: []Message{msg}})
					return err
				},
				func() error {
					_, err := c.Stream(context.Background(), ChatRequest{Messages: []Message{msg}}, func(Delta) error { return nil })
					return err
				},
			} {
				err := call()
				var apiErr *APIError
				if !errors.As(err, &apiErr) || apiErr.StatusCode != tc.status {
					t.Fatalf("err = %v, want an APIError %d underneath", err, tc.status)
				}
				if got := errors.Is(err, ErrImageRejected); got != tc.wantRejected {
					t.Fatalf("ErrImageRejected = %t, want %t (err: %v)", got, tc.wantRejected, err)
				}
				if tc.wantRejected && (!strings.Contains(err.Error(), "text-only") || !strings.Contains(err.Error(), `"text-model"`)) {
					t.Errorf("error does not name the cause and model: %v", err)
				}
			}
		})
	}
}
