package tools

import (
	"errors"
	"reflect"
	"testing"

	"github.com/lokalhub/kloo/internal/llm"
)

func TestNativeFCRecoversGLMDialect(t *testing.T) {
	fileBody := "package main\n\nimport \"fmt\"\n\nfunc main() {\n\tif 1 < 2 && 3 > 2 {\n\t\tfmt.Println(\"it's <ok> & \\\"quoted\\\"\")\n\t}\n}\n"
	cases := []struct {
		name    string
		content string
		want    []Call
	}{
		{
			name:    "the exact line from the live run",
			content: `<tool_call>list_dir<arg_key>path</arg_key><arg_value>.</arg_value></tool_call>`,
			want:    []Call{{Name: "list_dir", Args: map[string]any{"path": "."}}},
		},
		{
			name: "multi-line write_file content with <, > and quotes survives verbatim",
			content: "I'll create the file.\n<tool_call>write_file\n<arg_key>path</arg_key>\n<arg_value>main.go</arg_value>\n" +
				"<arg_key>content</arg_key>\n<arg_value>\n" + fileBody + "\n</arg_value>\n</tool_call>",
			want: []Call{{Name: "write_file", Args: map[string]any{"path": "main.go", "content": fileBody}}},
		},
		{
			name: "two calls in one message",
			content: "<tool_call>list_dir<arg_key>path</arg_key><arg_value>src</arg_value></tool_call>\n" +
				"<tool_call>read_file<arg_key>path</arg_key><arg_value>src/a.go</arg_value><arg_key>offset</arg_key><arg_value>10</arg_value></tool_call>",
			want: []Call{
				{Name: "list_dir", Args: map[string]any{"path": "src"}},
				{Name: "read_file", Args: map[string]any{"path": "src/a.go", "offset": float64(10)}},
			},
		},
		{
			// Also pins the ordering: the JSON-in-text extractor must not claim the
			// object (it has a "name" key) as a call to a tool called "app".
			name:    "JSON object value decodes; a non-JSON value stays a string",
			content: `<tool_call>write_file<arg_key>path</arg_key><arg_value>package.json</arg_value><arg_key>content</arg_key><arg_value>{"name":"app","private":true}</arg_value></tool_call>`,
			want:    []Call{{Name: "write_file", Args: map[string]any{"path": "package.json", "content": map[string]any{"name": "app", "private": true}}}},
		},
		{
			name:    "JSON bool and array values",
			content: `<tool_call>todo_write<arg_key>merge</arg_key><arg_value>true</arg_value><arg_key>todos</arg_key><arg_value>["a","b"]</arg_value></tool_call>`,
			want:    []Call{{Name: "todo_write", Args: map[string]any{"merge": true, "todos": []any{"a", "b"}}}},
		},
		{
			name:    "unclosed final </tool_call> is accepted when the args are complete",
			content: "<tool_call>run_command<arg_key>command</arg_key><arg_value>go test ./...</arg_value>\n",
			want:    []Call{{Name: "run_command", Args: map[string]any{"command": "go test ./..."}}},
		},
		{
			name:    "no-argument call",
			content: `<tool_call>finish</tool_call>`,
			want:    []Call{{Name: "finish", Args: map[string]any{}}},
		},
		{
			name:    "only one leading/trailing newline is stripped; interior whitespace kept",
			content: "<tool_call>write_file<arg_key>path</arg_key><arg_value>a.txt</arg_value><arg_key>content</arg_key><arg_value>\n\n  indented\n\n</arg_value></tool_call>",
			want:    []Call{{Name: "write_file", Args: map[string]any{"path": "a.txt", "content": "\n  indented\n"}}},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := NativeFCAdapter{}.ParseAll(llm.Message{Role: llm.RoleAssistant, Content: tc.content})
			if err != nil {
				t.Fatalf("ParseAll: %v", err)
			}
			if !reflect.DeepEqual(got, tc.want) {
				t.Errorf("calls =\n  %#v\nwant\n  %#v", got, tc.want)
			}
		})
	}
}

// A GLM call the extractor cannot fully parse must still fail LOUDLY, never parse
// to zero calls (a silent "answered") or to half a batch.
func TestNativeFCMalformedGLMIsLoud(t *testing.T) {
	cases := map[string]string{
		"unterminated value": `<tool_call>write_file<arg_key>path</arg_key><arg_value>a.txt`,
		"broken call with a JSON value is not recovered by the JSON extractor": `<tool_call>write_file<arg_key>content</arg_key><arg_value>{"name":"app"}`,
		"key without a value":         `<tool_call>list_dir<arg_key>path</arg_key></tool_call>`,
		"unterminated key":            `<tool_call>list_dir<arg_key>path`,
		"junk between elements":       `<tool_call>list_dir<arg_key>path</arg_key>oops<arg_value>.</arg_value></tool_call>`,
		"prose where the name goes":   `<tool_call>I will now list the dir</tool_call>`,
		"second call of batch broken": `<tool_call>list_dir<arg_key>path</arg_key><arg_value>.</arg_value></tool_call><tool_call>read_file<arg_key>path</arg_key>`,
	}
	for name, content := range cases {
		t.Run(name, func(t *testing.T) {
			calls, err := NativeFCAdapter{}.ParseAll(llm.Message{Role: llm.RoleAssistant, Content: content})
			if !errors.Is(err, ErrMalformedToolCall) {
				t.Errorf("err = %v (calls %v), want ErrMalformedToolCall", err, calls)
			}
		})
	}
}

// The new extractor must not change what the earlier dialects that share the
// <tool_call> wrapper produce.
func TestGLMExtractorLeavesFunctionDialectAlone(t *testing.T) {
	content := "<tool_call>\n<function=read_file>\n<parameter=path>a.go</parameter>\n</function>\n</tool_call>"
	if got := extractGLMToolCalls(content); got != nil {
		t.Errorf("GLM extractor claimed a <function=…> call: %#v", got)
	}
	calls, err := NativeFCAdapter{}.ParseAll(llm.Message{Role: llm.RoleAssistant, Content: content})
	if err != nil || len(calls) != 1 || calls[0].Name != "read_file" {
		t.Errorf("function dialect regressed: %v %#v", err, calls)
	}
}
