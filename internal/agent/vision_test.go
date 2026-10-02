package agent

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"image"
	"image/png"
	"strings"
	"testing"

	"github.com/lokalhub/kloo/internal/llm"
	"github.com/lokalhub/kloo/internal/llm/llmtest"
	"github.com/lokalhub/kloo/internal/tools"
)

func visionImage(t *testing.T, source string, w, h int) llm.ImagePart {
	t.Helper()
	var buf bytes.Buffer
	if err := png.Encode(&buf, image.NewRGBA(image.Rect(0, 0, w, h))); err != nil {
		t.Fatal(err)
	}
	p, err := llm.NewImagePart(source, buf.Bytes())
	if err != nil {
		t.Fatal(err)
	}
	return p
}

// wireMessages decodes a captured request body's messages generically.
func wireMessages(t *testing.T, body []byte) []map[string]any {
	t.Helper()
	var req struct {
		Messages []map[string]any `json:"messages"`
	}
	if err := json.Unmarshal(body, &req); err != nil {
		t.Fatal(err)
	}
	return req.Messages
}

// imageURLParts counts the image_url parts in one wire message.
func imageURLParts(m map[string]any) int {
	parts, ok := m["content"].([]any)
	if !ok {
		return 0
	}
	n := 0
	for _, p := range parts {
		if p.(map[string]any)["type"] == "image_url" {
			n++
		}
	}
	return n
}

// --image: the images go on the FIRST user message, next to the task text, on
// both the working-memory path and the legacy bounded-history path.
func TestTaskImagesRideOnTheTaskMessage(t *testing.T) {
	for _, withMemory := range []bool{true, false} {
		name := "legacy history"
		if withMemory {
			name = "working memory"
		}
		t.Run(name, func(t *testing.T) {
			srv := llmtest.Sequence(t, llmtest.Mock{Body: proseResp(t, "The bar is red.")})
			loop, _ := newLoop(t, srv, nil, &stubBudget{}, &stubChurn{})
			if withMemory {
				loop.Memory = NewWorkingMemory()
			}
			loop.TaskImages = []llm.ImagePart{visionImage(t, "ref.png", 8, 8), visionImage(t, "cur.png", 8, 8)}
			if _, err := loop.Run(context.Background(), "what colour is the bar?"); err != nil {
				t.Fatal(err)
			}
			msgs := wireMessages(t, srv.ModelCalls()[0])
			var task map[string]any
			for _, m := range msgs {
				if m["role"] == llm.RoleUser {
					task = m
					break
				}
			}
			parts, ok := task["content"].([]any)
			if !ok {
				t.Fatalf("task message content = %#v, want parts", task["content"])
			}
			if first := parts[0].(map[string]any); first["type"] != "text" || !strings.Contains(first["text"].(string), "what colour is the bar?") {
				t.Errorf("first part = %v, want the task text", first)
			}
			if n := imageURLParts(task); n != 2 {
				t.Errorf("task carries %d image parts, want 2", n)
			}
			total := 0
			for _, m := range msgs {
				total += imageURLParts(m)
			}
			if total != 2 {
				t.Errorf("request carries %d image parts in total, want exactly the 2 task images", total)
			}
		})
	}
}

// imageTool stands in for view_image: it returns one image on its Result.
type imageTool struct{ img llm.ImagePart }

func (t imageTool) Name() string        { return tools.NameViewImage }
func (t imageTool) Description() string { return "view" }
func (t imageTool) Schema() tools.ParamSchema {
	return tools.ParamSchema{Properties: map[string]tools.Property{"path": {Type: "string"}}, Required: []string{"path"}}
}
func (t imageTool) Invoke(ctx context.Context, c tools.Call) (tools.Result, error) {
	return tools.Result{Output: "showing shot.png", Images: []llm.ImagePart{t.img}}, nil
}

// view_image: the image reaches the model on the observation turn and is counted.
func TestViewImageResultReachesTheModel(t *testing.T) {
	srv := llmtest.Sequence(t,
		llmtest.Mock{Body: toolResp(t, 10, tcSpec{name: tools.NameViewImage, args: map[string]any{"path": "shot.png"}})},
		llmtest.Mock{Body: proseResp(t, "The header is blue.")},
	)
	loop, _ := newLoop(t, srv, nil, &stubBudget{}, &stubChurn{})
	loop.Memory = NewWorkingMemory()
	loop.Registry.Register(imageTool{img: visionImage(t, "shot.png", 16, 16)})
	rep, err := loop.Run(context.Background(), "look at shot.png")
	if err != nil {
		t.Fatal(err)
	}
	calls := srv.ModelCalls()
	if len(calls) < 2 {
		t.Fatalf("model calls = %d, want ≥2", len(calls))
	}
	found := false
	for _, m := range wireMessages(t, calls[1]) {
		if imageURLParts(m) == 1 {
			if m["role"] != llm.RoleUser {
				t.Errorf("image rides on a %v message, want user", m["role"])
			}
			found = true
		}
	}
	if !found {
		t.Error("second request carries no image part — the view_image result was dropped")
	}
	if rep.ToolCounters.ImagesViewed != 1 {
		t.Errorf("ImagesViewed = %d, want 1", rep.ToolCounters.ImagesViewed)
	}
}

func TestObservationAttachesImagesOnlyOnSuccess(t *testing.T) {
	img := visionImage(t, "a.png", 4, 4)
	call := tools.Call{Name: tools.NameViewImage}
	if got := observation(call, tools.Result{Output: "ok", Images: []llm.ImagePart{img}}, nil); len(got.Images) != 1 {
		t.Errorf("success observation carries %d images, want 1", len(got.Images))
	}
	if got := observation(call, tools.Result{Images: []llm.ImagePart{img}}, errors.New("boom")); len(got.Images) != 0 {
		t.Errorf("error observation carries %d images, want 0", len(got.Images))
	}
}

func TestViewImageIsReadOnly(t *testing.T) {
	if !isReadOnlyTool(tools.NameViewImage) {
		t.Error("view_image must count as read-only: looking at a picture changes nothing to verify")
	}
}

// Compaction must deal with an image deliberately: the task's images are pinned,
// an observation's image is folded WITH its message and replaced by a stub that
// names it, and the fold is counted.
func TestCompactionFoldsImagesDeliberately(t *testing.T) {
	const window = 4000
	task := "match the reference"
	taskImg := visionImage(t, "ref.png", 56, 56)
	shot := visionImage(t, "shots/old.png", 56, 56)
	convo := []llm.Message{{Role: llm.RoleUser, Content: task, Images: []llm.ImagePart{taskImg}},
		{Role: llm.RoleUser, Content: "tool view_image result:\nshowing shots/old.png", Images: []llm.ImagePart{shot}},
	}
	// Read dumps fold to one-line stubs, so the summary stays small and the test
	// isolates the image handling from the summary's own shedding.
	for i := 0; i < 40; i++ {
		convo = append(convo, userMsg("tool read_file result:\n"+strings.Repeat("line of file text ", 60)))
	}
	wm := NewWorkingMemory()
	out, err := wm.Assemble(MemoryInput{Task: task, TaskImages: []llm.ImagePart{taskImg}, Convo: convo, WindowTokens: window, SystemTokens: 100})
	if err != nil {
		t.Fatal(err)
	}
	if len(out[0].Images) != 1 || out[0].Images[0].Source != "ref.png" {
		t.Errorf("task images after compaction = %+v, want ref.png pinned", out[0].Images)
	}
	if n := llm.CountImages(out); n != 1 {
		t.Errorf("assembled prompt carries %d images, want only the pinned task image", n)
	}
	sum, ok := hasSummary(out)
	if !ok || !strings.Contains(sum, "[image shots/old.png removed from context by compaction") {
		t.Errorf("summary does not record the folded image:\n%s", sum)
	}
	if got := wm.Stats().ImagesFolded; got != 1 {
		t.Errorf("ImagesFolded = %d, want 1", got)
	}
	if got := wm.Stats().PromptTokens; got > window {
		t.Errorf("PromptTokens = %d over the window %d", got, window)
	}
}

func TestImagesCountTowardTheBudget(t *testing.T) {
	img := visionImage(t, "a.png", 280, 280)
	plain := []llm.Message{userMsg("hello")}
	with := []llm.Message{{Role: llm.RoleUser, Content: "hello", Images: []llm.ImagePart{img}}}
	if d := tokensOf(with) - tokensOf(plain); d != img.EstimateTokens() {
		t.Errorf("image adds %d tokens to the estimate, want %d", d, img.EstimateTokens())
	}

	t.Run("a task image the window cannot hold is a config error, not a dropped image", func(t *testing.T) {
		big := visionImage(t, "big.png", 2000, 2000)
		_, err := NewWorkingMemory().Assemble(MemoryInput{Task: "t", TaskImages: []llm.ImagePart{big}, Convo: []llm.Message{userMsg("t")}, WindowTokens: 500, SystemTokens: 10})
		if !errors.Is(err, ErrWindowTooSmall) {
			t.Errorf("err = %v, want ErrWindowTooSmall", err)
		}
	})
}

func TestImageRejectionIsNeverRetried(t *testing.T) {
	err := &llm.ImageRejectedError{Images: 1, Model: "m", API: &llm.APIError{StatusCode: 500, Body: "image input is not supported"}}
	if retryableLLMError(err, []int{500}) {
		t.Error("an image rejection was classed retryable; it is deterministic")
	}
	if !retryableLLMError(&llm.APIError{StatusCode: 500, Body: "boom"}, []int{500}) {
		t.Error("control: a plain 500 should still retry")
	}
}
