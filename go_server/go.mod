module github.com/spitmaan/aneurologic-phase5

go 1.22.5

// No external dependencies — pure stdlib Go server.
// gollama.cpp (dianlight/gollama.cpp) is used via the cloned repo at
// /go/src/github.com/dianlight/gollama.cpp for native in-process bindings.
// The default server path uses Ollama's REST API (http://localhost:11434)
// which provides the same interface without CGO requirements.
