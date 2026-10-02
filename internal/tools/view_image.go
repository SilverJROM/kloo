package tools

import (
	"context"
	"fmt"

	"github.com/lokalhub/kloo/internal/llm"
)

// NameViewImage is the opt-in vision tool.
const NameViewImage = "view_image"

// viewImageTool loads a workspace image into the conversation, so a model can look
// at a screenshot its own check produced rather than read a description of one.
//
// It is never part of DefaultRegistry: a text-only model offered it would call it
// and the next request would be refused. The CLI registers it only when vision is
// opted into AND the model is not marked text-only.
//
// The image cannot ride on the tool result itself — most OpenAI-compatible
// servers require a tool/observation turn to be text — so the Result carries it
// in Images and the loop attaches it to the observation message it already builds.
type viewImageTool struct{ ws Workspace }

// NewViewImageTool builds the view_image tool over ws. Reads go through the same
// jail as read_file, so a path outside the workspace is refused before any I/O.
func NewViewImageTool(ws Workspace) Tool { return viewImageTool{ws} }

func (t viewImageTool) Name() string { return NameViewImage }
func (t viewImageTool) Description() string {
	return "Look at an image file in the workspace (png, jpeg, gif or webp, at most 5 MiB). " +
		"The image is shown to you in the next message. Use it to check a screenshot or a reference design."
}
func (t viewImageTool) Schema() ParamSchema {
	return ParamSchema{
		Properties: map[string]Property{
			"path": {Type: "string", Description: "Workspace-relative path to the image."},
		},
		Required: []string{"path"},
	}
}

func (t viewImageTool) Invoke(ctx context.Context, c Call) (Result, error) {
	path, _ := argString(c.Args, "path")
	img, err := ViewImage(t.ws, path)
	if err != nil {
		return Result{}, err
	}
	size := "unknown size"
	if img.Width > 0 {
		size = fmt.Sprintf("%dx%d", img.Width, img.Height)
	}
	return Result{
		Output: fmt.Sprintf("showing %s (%s, %s, %d bytes) — the image is attached to this message", path, img.MIME, size, len(img.Data)),
		Images: []llm.ImagePart{img},
	}, nil
}

// ViewImage reads a workspace image through the jail, under llm.MaxImageBytes.
func ViewImage(ws Workspace, relPath string) (llm.ImagePart, error) {
	if relPath == "" {
		return llm.ImagePart{}, fmt.Errorf("tools: view_image: missing path")
	}
	abs, err := ws.Resolve(relPath)
	if err != nil {
		return llm.ImagePart{}, err
	}
	img, err := llm.ReadImageFile(relPath, abs)
	if err != nil {
		return llm.ImagePart{}, fmt.Errorf("tools: view_image: %w", err)
	}
	return img, nil
}
