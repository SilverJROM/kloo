package agent

import (
	"context"
	"testing"

	"github.com/lokalhub/kloo/internal/llm/llmtest"
)

// A reply carrying no structured tool_calls but GLM's text dialect in content is
// dispatched like a native call, and counted as text-recovered. Before the GLM
// extractor this ended the run on "no usable tool call after re-prompt".
func TestGLMTextDialectIsDispatchedAndCounted(t *testing.T) {
	srv := llmtest.Sequence(t,
		llmtest.Mock{Body: proseResp(t, "<tool_call>read_file<arg_key>path</arg_key><arg_value>main.go</arg_value></tool_call>")},
		llmtest.Mock{Body: toolResp(t, 10, tcSpec{name: "read_file", args: map[string]any{"path": "b.go"}})},
		llmtest.Mock{Body: proseResp(t, "Done reading.")},
	)
	loop, calls := newLoop(t, srv, nil, &stubBudget{}, &stubChurn{})
	rep, err := loop.Run(context.Background(), "read main.go")
	if err != nil {
		t.Fatal(err)
	}
	if len(*calls) < 2 || (*calls)[0].Name != "read_file" || (*calls)[0].Args["path"] != "main.go" {
		t.Fatalf("dispatched calls = %+v, want read_file main.go first", *calls)
	}
	if got := rep.ToolCounters.TextToolCalls; got != 1 {
		t.Errorf("TextToolCalls = %d, want 1 (the native call must not count)", got)
	}
}
