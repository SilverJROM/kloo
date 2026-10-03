package agent

import (
	"context"
	"strings"
	"testing"

	"github.com/lokalhub/kloo/internal/llm/llmtest"
	"github.com/lokalhub/kloo/internal/tools"
)

// The exact batched reply glm-5.3-flash sent on the live planner run.
const glmBatchedReply = `<tool_call>read_file<arg_key>limit</arg_key><arg_value>120</arg_value><arg_key>offset</arg_key><arg_value>60</arg_value><arg_key>path</arg_key><arg_value>docs/contract/products-seed.json</arg_value></tool_call><tool_call>list_dir<arg_key>path</arg_key><arg_value>.</arg_value></tool_call><tool_call>search<arg_key>query</arg_key><arg_value>plan-lint|PLAN-DRAFT</arg_value></tool_call>`

// Several calls in one turn: the first runs, the model is told by name which did
// NOT, and the run carries on — it is never failed for batching.
func TestExtraCallsRunFirstAndTellTheModel(t *testing.T) {
	srv := llmtest.Sequence(t,
		llmtest.Mock{Body: proseResp(t, glmBatchedReply)},
		llmtest.Mock{Body: toolResp(t, 10, tcSpec{name: "read_file", args: map[string]any{"path": "b.go"}})},
		llmtest.Mock{Body: proseResp(t, "Done.")},
	)
	loop, calls := newLoop(t, srv, nil, &stubBudget{}, &stubChurn{})
	var seen []tools.Call
	reg := loop.Registry
	reg.Register(recordTool{name: tools.NameListDir, calls: &seen})
	reg.Register(recordTool{name: tools.NameSearch, calls: &seen})
	rep, err := loop.Run(context.Background(), "plan the change")
	if err != nil {
		t.Fatal(err)
	}
	if rep.Reason == ReasonError {
		t.Fatalf("run failed on a batched turn: %v", rep.Err)
	}
	if len(*calls) == 0 || (*calls)[0].Name != "read_file" || (*calls)[0].Args["path"] != "docs/contract/products-seed.json" {
		t.Fatalf("first dispatched = %+v, want read_file docs/contract/products-seed.json", *calls)
	}
	if len(seen) != 0 {
		t.Errorf("batched extras were dispatched: %+v", seen)
	}
	second := string(srv.ModelCalls()[1])
	for _, want := range []string{"Only ONE tool call runs per turn", "NOT run", "list_dir, search"} {
		if !strings.Contains(second, want) {
			t.Errorf("next request does not tell the model %q", want)
		}
	}
	if rep.RailFires[string(RailExtraCalls)] != 1 {
		t.Errorf("rail_fires = %v, want extra-calls=1", rep.RailFires)
	}
}
