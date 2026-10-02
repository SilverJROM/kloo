package tools

import (
	"bytes"
	"context"
	"errors"
	"image"
	"image/png"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/lokalhub/kloo/internal/llm"
)

func pngBytes(t *testing.T, w, h int) []byte {
	t.Helper()
	var buf bytes.Buffer
	if err := png.Encode(&buf, image.NewRGBA(image.Rect(0, 0, w, h))); err != nil {
		t.Fatal(err)
	}
	return buf.Bytes()
}

func TestViewImageTool(t *testing.T) {
	ws, root := wsAt(t)
	if err := os.MkdirAll(filepath.Join(root, "shots"), 0o755); err != nil {
		t.Fatal(err)
	}
	mustWriteBytes := func(rel string, b []byte) {
		t.Helper()
		if err := os.WriteFile(filepath.Join(root, rel), b, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	mustWriteBytes("shots/page.png", pngBytes(t, 64, 32))
	mustWriteBytes("notes.txt", []byte("not an image"))
	mustWriteBytes("huge.png", append(pngBytes(t, 2, 2), make([]byte, llm.MaxImageBytes)...))
	outside := filepath.Join(t.TempDir(), "secret.png")
	if err := os.WriteFile(outside, pngBytes(t, 2, 2), 0o644); err != nil {
		t.Fatal(err)
	}

	tool := NewViewImageTool(ws)
	call := func(path string) (Result, error) {
		return tool.Invoke(context.Background(), Call{Name: NameViewImage, Args: map[string]any{"path": path}})
	}

	t.Run("workspace png is returned as an image part", func(t *testing.T) {
		res, err := call("shots/page.png")
		if err != nil {
			t.Fatal(err)
		}
		if len(res.Images) != 1 || res.Images[0].Width != 64 || res.Images[0].Source != "shots/page.png" {
			t.Fatalf("images = %+v, want one 64px image sourced from shots/page.png", res.Images)
		}
		if !strings.Contains(res.Output, "64x32") {
			t.Errorf("output %q does not describe the image", res.Output)
		}
	})

	t.Run("relative escape is refused by the jail", func(t *testing.T) {
		if _, err := call("../escape.png"); !errors.Is(err, ErrPathEscape) {
			t.Errorf("err = %v, want ErrPathEscape", err)
		}
	})

	t.Run("absolute path outside the workspace is refused", func(t *testing.T) {
		if _, err := call(outside); !errors.Is(err, ErrPathEscape) {
			t.Errorf("err = %v, want ErrPathEscape", err)
		}
	})

	t.Run("symlink pointing outside is refused", func(t *testing.T) {
		if err := os.Symlink(outside, filepath.Join(root, "link.png")); err != nil {
			t.Skip("symlinks unavailable:", err)
		}
		if _, err := call("link.png"); !errors.Is(err, ErrPathEscape) {
			t.Errorf("err = %v, want ErrPathEscape", err)
		}
	})

	t.Run("over-cap file is refused before it is sent", func(t *testing.T) {
		if _, err := call("huge.png"); err == nil || !strings.Contains(err.Error(), "image cap") {
			t.Errorf("err = %v, want the size cap", err)
		}
	})

	t.Run("non-image is refused loudly", func(t *testing.T) {
		if _, err := call("notes.txt"); err == nil || !strings.Contains(err.Error(), "unsupported type") {
			t.Errorf("err = %v, want unsupported type", err)
		}
	})

	t.Run("missing file is an error, not an empty result", func(t *testing.T) {
		if _, err := call("shots/nope.png"); err == nil {
			t.Error("want an error for a missing file")
		}
	})
}

// view_image is opt-in: the default vocabulary must not offer it, or a text-only
// model would call it and have its next request refused.
func TestViewImageNotInDefaultRegistry(t *testing.T) {
	ws, _ := wsAt(t)
	if _, ok := DefaultRegistry(ws).Lookup(NameViewImage); ok {
		t.Error("view_image is registered by default; it must be opt-in")
	}
}
