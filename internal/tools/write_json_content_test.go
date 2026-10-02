package tools

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// Models routinely send a JSON file's content as a JSON value rather than a
// string. Before this was handled, argString returned "" and write_file wrote a
// 0-byte package.json while reporting success — the worst kind of tool failure.
func TestWriteFileNonStringContent(t *testing.T) {
	cases := []struct {
		name    string
		content any
		want    string // expected file content; "" means the call must fail
		wantErr string
	}{
		{"object is written as formatted JSON", map[string]any{"name": "app", "private": true}, "{\n  \"name\": \"app\",\n  \"private\": true\n}\n", ""},
		{"array is written as formatted JSON", []any{"a", float64(1)}, "[\n  \"a\",\n  1\n]\n", ""},
		{"string still written verbatim", "plain\n", "plain\n", ""},
		{"missing content is an error, not an empty file", nil, "", `argument "content"`},
		{"number is rejected", float64(3), "", "content must be a string"},
		{"bool is rejected", true, "", "content must be a string"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			ws, root := wsAt(t)
			args := map[string]any{"path": "out.json"}
			if tc.content != nil {
				args["content"] = tc.content
			}
			_, err := DefaultRegistry(ws).Dispatch(context.Background(), Call{Name: NameWriteFile, Args: args})
			p := filepath.Join(root, "out.json")
			if tc.wantErr != "" {
				if err == nil || !strings.Contains(err.Error(), tc.wantErr) {
					t.Fatalf("err = %v, want containing %q", err, tc.wantErr)
				}
				if _, statErr := os.Stat(p); statErr == nil {
					t.Errorf("a rejected call must not create the file")
				}
				return
			}
			if err != nil {
				t.Fatalf("Dispatch: %v", err)
			}
			got, _ := os.ReadFile(p)
			if string(got) != tc.want {
				t.Errorf("content = %q, want %q", got, tc.want)
			}
		})
	}
}
