package tools

import (
	"encoding/json"
	"strings"
)

// A fifth text tool-call dialect: GLM's own chat-template format. glm-5.3-flash
// behind an OpenAI-compatible gateway (toolFormat native) usually returns
// structured tool_calls, but some turns come back with NO tool_calls and the
// template's raw markup in content instead:
//
//	<tool_call>list_dir<arg_key>path</arg_key><arg_value>.</arg_value></tool_call>
//
// The name is bare text straight after <tool_call>; each argument is a key/value
// element pair. None of the other four extractors match it, so looksLikeToolCall
// fired and the run died on "no usable tool call after re-prompt" — both GLM
// planner runs on the lab ended that way at step 13.
//
// Values follow GLM's template convention: a non-string argument is written as
// JSON, a string argument raw. So a value that parses as a JSON number, bool,
// object or array becomes that value; anything else stays the raw string, with
// interior whitespace untouched (it is often a whole file).

const (
	glmCallOpen   = "<tool_call>"
	glmCallClose  = "</tool_call>"
	glmKeyOpen    = "<arg_key>"
	glmKeyClose   = "</arg_key>"
	glmValueOpen  = "<arg_value>"
	glmValueClose = "</arg_value>"
)

// extractGLMToolCalls recovers every call in content written in the GLM dialect.
//
// It is all-or-nothing: if any <tool_call> in the message is malformed (an
// unterminated key or value, junk between elements, a name that is not an
// identifier) it returns nil, and the caller's looksLikeToolCall turns that into
// the loud corrective error. Executing the half of a garbled batch that happened
// to parse is worse than asking once more.
func extractGLMToolCalls(content string) []Call {
	// The <tool_call><function=…> and <tool_call><invoke_name> dialects share the
	// wrapper, but their body starts with '<', which is never a bare name — so they
	// fail isToolIdent below and are not claimed here (and they are tried first).
	if !strings.Contains(content, glmCallOpen) {
		return nil
	}
	var out []Call
	s := content
	for {
		i := strings.Index(s, glmCallOpen)
		if i < 0 {
			break
		}
		call, rest, ok := parseGLMCall(s[i+len(glmCallOpen):])
		if !ok {
			return nil
		}
		out = append(out, call)
		s = rest
	}
	return out
}

// parseGLMCall parses one call body (the text after <tool_call>) and returns the
// remainder after its </tool_call>. A final call whose </tool_call> is missing is
// accepted when the message ends right after its last complete argument.
func parseGLMCall(s string) (Call, string, bool) {
	nameEnd := len(s)
	for _, m := range []string{glmKeyOpen, glmCallClose, "<"} {
		if j := strings.Index(s, m); j >= 0 && j < nameEnd {
			nameEnd = j
		}
	}
	name := strings.TrimSpace(s[:nameEnd])
	if !isToolIdent(name) {
		return Call{}, "", false
	}
	s = s[nameEnd:]
	args := map[string]any{}
	for {
		s = strings.TrimLeft(s, " \t\r\n")
		switch {
		case strings.HasPrefix(s, glmCallClose):
			return Call{Name: name, Args: args}, s[len(glmCallClose):], true
		case s == "":
			return Call{Name: name, Args: args}, "", true // unclosed final call, args complete
		case strings.HasPrefix(s, glmKeyOpen):
			s = s[len(glmKeyOpen):]
			k := strings.Index(s, glmKeyClose)
			if k < 0 {
				return Call{}, "", false
			}
			key := strings.TrimSpace(s[:k])
			s = strings.TrimLeft(s[k+len(glmKeyClose):], " \t\r\n")
			if key == "" || !strings.HasPrefix(s, glmValueOpen) {
				return Call{}, "", false
			}
			s = s[len(glmValueOpen):]
			v := strings.Index(s, glmValueClose)
			if v < 0 {
				return Call{}, "", false
			}
			args[key] = glmValue(s[:v])
			s = s[v+len(glmValueClose):]
		default:
			return Call{}, "", false
		}
	}
}

// glmValue decodes one <arg_value>. Exactly one leading and one trailing newline
// are dropped — the template's own line breaks around a multi-line value — and
// nothing else, so a file's indentation and trailing blank lines survive.
func glmValue(raw string) any {
	v := strings.TrimPrefix(raw, "\r\n")
	if len(v) == len(raw) {
		v = strings.TrimPrefix(v, "\n")
	}
	if t := strings.TrimSuffix(v, "\r\n"); len(t) != len(v) {
		v = t
	} else {
		v = strings.TrimSuffix(v, "\n")
	}
	if t := strings.TrimSpace(v); t != "" {
		var j any
		if err := json.Unmarshal([]byte(t), &j); err == nil {
			switch j.(type) {
			case float64, bool, map[string]any, []any:
				return j
			}
		}
	}
	return v
}

// isToolIdent reports whether name looks like a tool name. Unknown-but-valid
// names are left for Dispatch to reject (ErrUnknownTool), exactly as with the
// other dialects; this only refuses text that cannot be a name at all, such as
// prose that happened to follow a stray <tool_call>.
func isToolIdent(name string) bool {
	if name == "" || len(name) > 128 {
		return false
	}
	for _, r := range name {
		if !(r == '_' || r == '-' || r == '.' || r >= '0' && r <= '9' || r >= 'a' && r <= 'z' || r >= 'A' && r <= 'Z') {
			return false
		}
	}
	return true
}
