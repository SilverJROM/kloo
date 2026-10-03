package tools

import (
	"context"
	"errors"
	"os"
	"reflect"
	"strings"
	"testing"

	"github.com/lokalhub/kloo/internal/llm"
)

// The two exact replies glm-5.3-flash sent on the live planner run (content,
// finish_reason=stop, no structured tool_calls).
const (
	glmLiveReply1 = `<tool_call>list_dir<arg_key>path</arg_key><arg_value>.</arg_value></tool_call>`
	glmLiveReply2 = `<tool_call>read_file<arg_key>limit</arg_key><arg_value>120</arg_value><arg_key>offset</arg_key><arg_value>60</arg_value><arg_key>path</arg_key><arg_value>docs/contract/products-seed.json</arg_value></tool_call><tool_call>list_dir<arg_key>path</arg_key><arg_value>.</arg_value></tool_call><tool_call>search<arg_key>query</arg_key><arg_value>plan-lint|PLAN-DRAFT</arg_value></tool_call>`
)

func TestGLMLiveRepliesParse(t *testing.T) {
	cases := []struct {
		name    string
		content string
		want    []Call
	}{
		{"reply 1: list_dir", glmLiveReply1, []Call{{Name: NameListDir, Args: map[string]any{"path": "."}}}},
		{"reply 2: three batched calls, read_file args are its real ones", glmLiveReply2, []Call{
			{Name: NameReadFile, Args: map[string]any{"limit": float64(120), "offset": float64(60), "path": "docs/contract/products-seed.json"}},
			{Name: NameListDir, Args: map[string]any{"path": "."}},
			{Name: NameSearch, Args: map[string]any{"query": "plan-lint|PLAN-DRAFT"}},
		}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := NativeFCAdapter{}.ParseAll(llm.Message{Role: llm.RoleAssistant, Content: tc.content})
			if err != nil {
				t.Fatalf("ParseAll: %v", err)
			}
			if !reflect.DeepEqual(got, tc.want) {
				t.Fatalf("calls =\n  %#v\nwant\n  %#v", got, tc.want)
			}
			// Every recovered call must also be dispatchable against the real
			// registry: the names exist and the args satisfy the schemas.
			ws, root := wsAt(t)
			if err := os.MkdirAll(root+"/docs/contract", 0o755); err != nil {
				t.Fatal(err)
			}
			mustWrite(t, root+"/docs/contract/products-seed.json", "{}\n")
			reg := DefaultRegistry(ws)
			for _, c := range got {
				if _, err := reg.Dispatch(context.Background(), c); errors.Is(err, ErrUnknownTool) || errors.Is(err, ErrInvalidArgs) {
					t.Errorf("%s does not dispatch: %v", c.Name, err)
				}
			}
		})
	}
}

func TestTextCallAliases(t *testing.T) {
	cases := []struct {
		name    string
		content string
		want    []Call
	}{
		{"GLM: ls + dir → list_dir + path",
			`<tool_call>ls<arg_key>dir</arg_key><arg_value>src</arg_value></tool_call>`,
			[]Call{{Name: NameListDir, Args: map[string]any{"path": "src"}}}},
		{"GLM: cat + file_path → read_file + path",
			`<tool_call>cat<arg_key>file_path</arg_key><arg_value>a.go</arg_value></tool_call>`,
			[]Call{{Name: NameReadFile, Args: map[string]any{"path": "a.go"}}}},
		{"GLM: bash + cmd → run_command + command",
			`<tool_call>bash<arg_key>cmd</arg_key><arg_value>go test ./...</arg_value></tool_call>`,
			[]Call{{Name: NameRunCommand, Args: map[string]any{"command": "go test ./..."}}}},
		{"GLM: grep + pattern → search + query",
			`<tool_call>grep<arg_key>pattern</arg_key><arg_value>TODO</arg_value></tool_call>`,
			[]Call{{Name: NameSearch, Args: map[string]any{"query": "TODO"}}}},
		{"function dialect: create_file + contents → write_file + content",
			"<function=create_file>\n<parameter=path>x.txt</parameter>\n<parameter=contents>hi</parameter>\n</function>",
			[]Call{{Name: NameWriteFile, Args: map[string]any{"path": "x.txt", "content": "hi"}}}},
		{"a canonical arg already present is never overwritten",
			`<tool_call>read_file<arg_key>path</arg_key><arg_value>a.go</arg_value><arg_key>file</arg_key><arg_value>b.go</arg_value></tool_call>`,
			[]Call{{Name: NameReadFile, Args: map[string]any{"path": "a.go", "file": "b.go"}}}},
		{"real tool names are never remapped (read_dir stays read_dir)",
			`<tool_call>read_dir<arg_key>path</arg_key><arg_value>.</arg_value></tool_call>`,
			[]Call{{Name: NameReadDir, Args: map[string]any{"path": "."}}}},
		{"an unknown name passes through untouched, for Dispatch to refuse",
			`<tool_call>teleport<arg_key>to</arg_key><arg_value>mars</arg_value></tool_call>`,
			[]Call{{Name: "teleport", Args: map[string]any{"to": "mars"}}}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := NativeFCAdapter{}.ParseAll(llm.Message{Role: llm.RoleAssistant, Content: tc.content})
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(got, tc.want) {
				t.Errorf("calls =\n  %#v\nwant\n  %#v", got, tc.want)
			}
		})
	}
}

// Structured tool_calls are generated against the schema: aliasing them would
// rewrite what the model actually asked for, so they are left exactly as sent.
func TestStructuredCallsAreNotAliased(t *testing.T) {
	msg := llm.Message{Role: llm.RoleAssistant, ToolCalls: []llm.ToolCall{{Function: llm.FunctionCall{Name: "ls", Arguments: `{"dir":"src"}`}}}}
	got, err := NativeFCAdapter{}.ParseAll(msg)
	if err != nil || len(got) != 1 || got[0].Name != "ls" || got[0].Args["dir"] != "src" {
		t.Errorf("structured call rewritten: %#v %v", got, err)
	}
}

func TestXMLAdapterAliases(t *testing.T) {
	got, err := XMLAdapter{}.ParseAll(llm.Message{Role: llm.RoleAssistant, Content: `<tool name="cat"><arg name="file_path">a.go</arg></tool>`})
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 1 || got[0].Name != NameReadFile || got[0].Args["path"] != "a.go" {
		t.Errorf("xml call not aliased: %#v", got)
	}
}

func TestUnknownToolNamesTheVocabulary(t *testing.T) {
	ws, _ := wsAt(t)
	_, err := DefaultRegistry(ws).Dispatch(context.Background(), Call{Name: "teleport", Args: map[string]any{}})
	if !errors.Is(err, ErrUnknownTool) {
		t.Fatalf("err = %v, want ErrUnknownTool", err)
	}
	for _, want := range []string{`"teleport"`, NameReadFile, NameListDir, NameWriteFile} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("error %q does not name %s", err, want)
		}
	}
}
