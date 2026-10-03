package tools

import "strings"

// Tool and argument aliases for calls recovered from TEXT.
//
// A model writing a call as text is writing from its training, not from the
// schema kloo sent: it reaches for the names its own agent harness used (ls,
// cat, bash, grep) and the argument names that went with them (file_path, cmd,
// pattern). Structured tool_calls are generated against the schema and do not
// need this; text calls do, and every miss costs a corrective round-trip on a
// small model that may not recover from it.
//
// Deliberately small and unambiguous. A name only maps when the target means
// the same thing with the same arguments; anything that would need its
// arguments rewritten (str_replace, apply_patch) is NOT aliased, because a
// mistranslated edit is worse than a refused one. A name that is not here and
// not a real tool still fails at Dispatch with ErrUnknownTool.
var toolNameAliases = map[string]string{
	// directory listing
	"ls": NameListDir, "list_files": NameListDir, "list_directory": NameListDir,
	"listdir": NameListDir, "list_dir_contents": NameListDir, "dir": NameListDir,
	// reading
	"cat": NameReadFile, "open_file": NameReadFile, "view_file": NameReadFile,
	"read": NameReadFile, "readfile": NameReadFile, "get_file_contents": NameReadFile,
	// searching
	"grep": NameSearch, "rg": NameSearch, "ripgrep": NameSearch, "search_files": NameSearch,
	"grep_search": NameSearch, "search_code": NameSearch, "find_in_files": NameSearch,
	// shell
	"bash": NameRunCommand, "shell": NameRunCommand, "sh": NameRunCommand, "exec": NameRunCommand,
	"execute": NameRunCommand, "execute_command": NameRunCommand, "run_shell_command": NameRunCommand,
	"run_terminal_cmd": NameRunCommand, "terminal": NameRunCommand,
	// whole-file writes
	"create_file": NameWriteFile, "write": NameWriteFile, "writefile": NameWriteFile, "save_file": NameWriteFile,
}

// argAliases maps, per CANONICAL tool, an alias argument name to kloo's name.
// Applied only when the canonical argument is absent, so a call that already
// uses kloo's names is never rewritten.
var argAliases = map[string]map[string]string{
	NameReadFile: {
		"file_path": "path", "filepath": "path", "file": "path", "filename": "path", "target_file": "path",
		"start_line": "offset", "line_offset": "offset",
		"max_lines": "limit", "num_lines": "limit", "line_count": "limit",
	},
	NameListDir: {
		"dir": "path", "directory": "path", "dir_path": "path", "folder": "path", "relative_workspace_path": "path",
	},
	NameReadDir: {
		"dir": "path", "directory": "path", "dir_path": "path", "folder": "path",
	},
	NameWriteFile: {
		"file_path": "path", "filepath": "path", "file": "path", "filename": "path", "target_file": "path",
		"contents": "content", "text": "content", "file_content": "content", "data": "content", "body": "content",
	},
	NameEditFile: {
		"file_path": "path", "filepath": "path", "file": "path", "filename": "path", "target_file": "path",
	},
	NameSearch: {
		"pattern": "query", "regex": "query", "q": "query", "search": "query", "search_term": "query",
		"directory": "path", "dir": "path",
	},
	NameRunCommand: {
		"cmd": "command", "shell_command": "command", "script": "command",
	},
}

// normalizeTextCalls rewrites alias tool and argument names to kloo's own, in
// place. Names are matched case-insensitively; a name that is already a kloo
// tool is never remapped (list_dir and read_dir are both real tools here).
func normalizeTextCalls(calls []Call) []Call {
	for i := range calls {
		name := calls[i].Name
		if canon, ok := toolNameAliases[strings.ToLower(name)]; ok && !isCanonicalToolName(name) {
			calls[i].Name = canon
		}
		aliases := argAliases[calls[i].Name]
		if len(aliases) == 0 || calls[i].Args == nil {
			continue
		}
		for alias, canon := range aliases {
			v, ok := calls[i].Args[alias]
			if !ok {
				continue
			}
			if _, taken := calls[i].Args[canon]; taken {
				continue
			}
			calls[i].Args[canon] = v
			delete(calls[i].Args, alias)
		}
	}
	return calls
}

// canonicalToolNames are kloo's built-in tool names; an alias never shadows one.
var canonicalToolNames = map[string]bool{
	NameReadFile: true, NameListDir: true, NameReadDir: true, NameWriteFile: true, NameEditFile: true,
	NameSearch: true, NameRunCommand: true, NameCommandOutput: true, NameFinish: true,
	NameTodoWrite: true, NameViewImage: true, "search_replace": true,
}

func isCanonicalToolName(name string) bool { return canonicalToolNames[name] }
