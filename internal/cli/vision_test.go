package cli

import (
	"bytes"
	"encoding/json"
	"image"
	"image/png"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/lokalhub/kloo/internal/config"
	"github.com/lokalhub/kloo/internal/llm/llmtest"
	"github.com/lokalhub/kloo/internal/tools"
)

// writePNG writes a real w×h PNG at path.
func writePNG(t *testing.T, path string, w, h int) {
	t.Helper()
	var buf bytes.Buffer
	if err := png.Encode(&buf, image.NewRGBA(image.Rect(0, 0, w, h))); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, buf.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}
}

func visionCfg(endpoint string) config.Config {
	return config.Config{
		Endpoint:                endpoint,
		Model:                   "vl-model",
		ToolFormat:              config.DefaultToolFormat,
		Effort:                  config.DefaultEffort,
		MaxSteps:                2,
		MaxContextTokens:        config.DefaultMaxContextTokens,
		ChurnRounds:             config.DefaultChurnRounds,
		JSONSummary:             true,
		LLMMaxRetries:           3,
		LLMRetryableStatusCodes: append([]int(nil), config.DefaultLLMRetryableStatusCodes...),
		LLMRetryBaseDelay:       time.Millisecond,
		LLMRetryMaxDelay:        time.Millisecond,
		LLMColdLoadTimeout:      config.DefaultLLMColdLoadTimeout,
		LLMStreamIdleTimeout:    config.DefaultLLMStreamIdle,
	}
}

// lastJSONLine returns the run summary JSON object printed by --json.
func lastJSONLine(t *testing.T, out string) map[string]any {
	t.Helper()
	lines := strings.Split(strings.TrimSpace(out), "\n")
	for i := len(lines) - 1; i >= 0; i-- {
		l := strings.TrimSpace(lines[i])
		l = strings.TrimPrefix(l, "KLOO_RESULT_JSON ")
		if strings.HasPrefix(l, "{") {
			var m map[string]any
			if err := json.Unmarshal([]byte(l), &m); err == nil {
				return m
			}
		}
	}
	t.Fatalf("no JSON summary in output:\n%s", out)
	return nil
}

func TestHeadlessImageIsSentAndRecorded(t *testing.T) {
	dir := t.TempDir()
	t.Chdir(dir)
	writePNG(t, filepath.Join(dir, "ref.png"), 40, 20)
	srv := llmtest.Sequence(t, llmtest.Mock{Body: `{"choices":[{"message":{"role":"assistant","content":"The bar is red."}}],"usage":{"prompt_tokens":90,"completion_tokens":5,"total_tokens":95}}`})
	cfg := visionCfg(srv.URL + "/v1")
	cfg.Images = []string{"ref.png"}
	var out bytes.Buffer
	_ = defaultRunHeadless(cfg, "what colour is the bar?", "", lintOpts{}, &out)

	calls := srv.ModelCalls()
	if len(calls) == 0 || !strings.Contains(string(calls[0]), `"type":"image_url"`) || !strings.Contains(string(calls[0]), "data:image/png;base64,") {
		t.Fatalf("first model call carries no image part:\n%.400s", calls)
	}
	if !strings.Contains(out.String(), "image: ref.png (image/png 40x20") {
		t.Errorf("run header does not list the image:\n%s", out.String())
	}
	sum := lastJSONLine(t, out.String())
	imgs, _ := sum["images"].([]any)
	if len(imgs) != 1 {
		t.Fatalf("--json images = %v, want one record", sum["images"])
	}
	rec := imgs[0].(map[string]any)
	if rec["path"] != "ref.png" || rec["mime"] != "image/png" || rec["width"] != float64(40) {
		t.Errorf("image record = %v", rec)
	}
}

// The text-only failure must be loud, attributable, and cost exactly one call:
// llama.cpp reports it as a 500, which the default ladder would otherwise retry.
func TestHeadlessTextOnlyEndpointFailsLoudly(t *testing.T) {
	dir := t.TempDir()
	t.Chdir(dir)
	writePNG(t, filepath.Join(dir, "ref.png"), 8, 8)
	srv := llmtest.Sequence(t,
		llmtest.Mock{Status: 500, Body: `{"error":{"code":500,"message":"image input is not supported - hint: if this is unexpected, you may need to provide the mmproj"}}`},
		llmtest.Mock{Body: `{"choices":[{"message":{"role":"assistant","content":"unused"}}]}`},
	)
	cfg := visionCfg(srv.URL + "/v1")
	cfg.Images = []string{"ref.png"}
	var out bytes.Buffer
	if err := defaultRunHeadless(cfg, "describe", "", lintOpts{}, &out); err == nil {
		t.Fatal("a rejected image must fail the run")
	}
	if n := len(srv.ModelCalls()); n != 1 {
		t.Errorf("model calls = %d, want 1 — an image rejection must not be retried\n%s", n, out.String())
	}
	if !strings.Contains(out.String(), "text-only") {
		t.Errorf("output does not name the cause:\n%s", out.String())
	}
	sum := lastJSONLine(t, out.String())
	if sum["failure_code"] != "image_rejected" {
		t.Errorf("failure_code = %v, want image_rejected", sum["failure_code"])
	}
}

func TestHeadlessBadImageFailsBeforeAnyModelCall(t *testing.T) {
	dir := t.TempDir()
	t.Chdir(dir)
	if err := os.WriteFile(filepath.Join(dir, "notes.png"), []byte("not really a png"), 0o644); err != nil {
		t.Fatal(err)
	}
	for _, path := range []string{"missing.png", "notes.png"} {
		srv := llmtest.Sequence(t, llmtest.Mock{Body: `{"choices":[{"message":{"role":"assistant","content":"unused"}}]}`})
		cfg := visionCfg(srv.URL + "/v1")
		cfg.Images = []string{path}
		var out bytes.Buffer
		err := defaultRunHeadless(cfg, "describe", "", lintOpts{}, &out)
		if err == nil || !strings.Contains(err.Error(), "--image") {
			t.Errorf("%s: err = %v, want an --image error", path, err)
		}
		if n := len(srv.ModelCalls()); n != 0 {
			t.Errorf("%s: model calls = %d, want 0", path, n)
		}
	}
}

func TestRegisterVisionTools(t *testing.T) {
	ws, err := tools.NewWorkspace(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		name string
		cfg  config.Config
		want bool
	}{
		{"off by default", config.Config{}, false},
		{"on when opted in", config.Config{ViewImage: true}, true},
		{"never for a text-only model", config.Config{ViewImage: true, TextOnly: true}, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			reg := tools.DefaultRegistry(ws)
			registerVisionTools(tc.cfg, reg, ws)
			if _, ok := reg.Lookup(tools.NameViewImage); ok != tc.want {
				t.Errorf("view_image registered = %t, want %t", ok, tc.want)
			}
		})
	}
}
