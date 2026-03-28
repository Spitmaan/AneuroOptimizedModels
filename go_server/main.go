// ANeurologic Phase 5 — Stage 4: Go-Native Inference Server
// ===========================================================
// High-performance, concurrent LLM inference API server using Go.
//
// Architecture:
//   - Integrates with llama.cpp server (running separately) via HTTP
//   - OR uses Ollama's Go API package for direct Go-native calls
//   - Concurrent request handling with goroutines + worker pool
//   - Benchmarks multi-client throughput vs Python vLLM baseline
//
// Approach:
//   gollama.cpp (github.com/dianlight/gollama.cpp) provides direct in-process
//   llama.cpp bindings via purego (no CGO). For production serving, we also
//   provide an Ollama-compatible client path.
//
// Endpoints:
//   POST /v1/completions      — OpenAI-compatible text completion
//   POST /v1/chat/completions — OpenAI-compatible chat
//   GET  /health              — Health check
//   GET  /metrics             — Prometheus-format throughput metrics
//
// Usage (inside container):
//   cd /workspace/go_server && go mod tidy && go run main.go --model /workspace/models/gguf/lfm.gguf
//   OR with Ollama backend:
//   cd /workspace/go_server && go run main.go --backend ollama --model lfm2.5-1.2b

package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"math"
	"net/http"
	"os"
	"sync"
	"sync/atomic"
	"time"
)

// ── Config ──────────────────────────────────────────────────────────────────

type Config struct {
	Host        string
	Port        int
	Backend     string // "ollama" | "llamacpp_server"
	ModelPath   string // GGUF path (llamacpp_server) or model name (ollama)
	BackendURL  string // Ollama or llama-server URL
	MaxWorkers  int    // concurrent inference threads
	MaxTokens   int
}

// ── Metrics ─────────────────────────────────────────────────────────────────

type Metrics struct {
	TotalRequests  atomic.Int64
	TotalTokens    atomic.Int64
	TotalErrors    atomic.Int64
	InflightReqs   atomic.Int64
	TotalLatencyMs atomic.Int64 // sum, divide by TotalRequests for avg
	StartTime      time.Time
}

var metrics = &Metrics{StartTime: time.Now()}

func (m *Metrics) RecordRequest(tokens int64, latencyMs int64, isErr bool) {
	m.TotalRequests.Add(1)
	if isErr {
		m.TotalErrors.Add(1)
		return
	}
	m.TotalTokens.Add(tokens)
	m.TotalLatencyMs.Add(latencyMs)
}

// ── OpenAI-compatible types ──────────────────────────────────────────────────

type CompletionRequest struct {
	Model       string    `json:"model"`
	Prompt      string    `json:"prompt,omitempty"`
	Messages    []Message `json:"messages,omitempty"`
	MaxTokens   int       `json:"max_tokens"`
	Temperature float32   `json:"temperature"`
	Stream      bool      `json:"stream"`
}

type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type CompletionChoice struct {
	Text         string  `json:"text,omitempty"`
	Index        int     `json:"index"`
	FinishReason string  `json:"finish_reason"`
	Message      *Message `json:"message,omitempty"`
}

type CompletionResponse struct {
	ID      string             `json:"id"`
	Object  string             `json:"object"`
	Created int64              `json:"created"`
	Model   string             `json:"model"`
	Choices []CompletionChoice `json:"choices"`
	Usage   UsageStats         `json:"usage"`
}

type UsageStats struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// ── Backend interface ────────────────────────────────────────────────────────

type Backend interface {
	Generate(ctx context.Context, prompt string, maxTokens int) (string, int, error)
	Name() string
}

// ── Ollama backend ───────────────────────────────────────────────────────────

type OllamaBackend struct {
	BaseURL   string
	ModelName string
	Client    *http.Client
}

func NewOllamaBackend(url, model string) *OllamaBackend {
	return &OllamaBackend{
		BaseURL:   url,
		ModelName: model,
		Client:    &http.Client{Timeout: 120 * time.Second},
	}
}

func (b *OllamaBackend) Name() string { return fmt.Sprintf("Ollama(%s)", b.ModelName) }

func (b *OllamaBackend) Generate(ctx context.Context, prompt string, maxTokens int) (string, int, error) {
	reqBody, _ := json.Marshal(map[string]interface{}{
		"model":  b.ModelName,
		"prompt": prompt,
		"stream": false,
		"options": map[string]interface{}{
			"num_predict": maxTokens,
			"temperature": 0.7,
		},
	})

	req, err := http.NewRequestWithContext(ctx, "POST",
		b.BaseURL+"/api/generate",
		jsonReader(reqBody))
	if err != nil {
		return "", 0, err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := b.Client.Do(req)
	if err != nil {
		return "", 0, fmt.Errorf("ollama request failed: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", 0, err
	}

	var result map[string]interface{}
	if err := json.Unmarshal(body, &result); err != nil {
		return "", 0, fmt.Errorf("parse error: %w", err)
	}
	if errMsg, ok := result["error"].(string); ok {
		return "", 0, fmt.Errorf("ollama: %s", errMsg)
	}
	text, _ := result["response"].(string)
	// Ollama returns eval_count (tokens generated)
	evalCount := 0
	if ec, ok := result["eval_count"].(float64); ok {
		evalCount = int(ec)
	}
	return text, evalCount, nil
}

// ── llama.cpp server backend ─────────────────────────────────────────────────

type LlamaCppBackend struct {
	BaseURL string
	Client  *http.Client
}

func NewLlamaCppBackend(url string) *LlamaCppBackend {
	return &LlamaCppBackend{
		BaseURL: url,
		Client:  &http.Client{Timeout: 120 * time.Second},
	}
}

func (b *LlamaCppBackend) Name() string { return "llama-server" }

func (b *LlamaCppBackend) Generate(ctx context.Context, prompt string, maxTokens int) (string, int, error) {
	reqBody, _ := json.Marshal(map[string]interface{}{
		"prompt":      prompt,
		"n_predict":   maxTokens,
		"temperature": 0.7,
	})

	req, err := http.NewRequestWithContext(ctx, "POST",
		b.BaseURL+"/completion",
		jsonReader(reqBody))
	if err != nil {
		return "", 0, err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := b.Client.Do(req)
	if err != nil {
		return "", 0, fmt.Errorf("llama-server request failed: %w", err)
	}
	defer resp.Body.Close()

	body, _ := io.ReadAll(resp.Body)
	var result map[string]interface{}
	json.Unmarshal(body, &result)

	text, _ := result["content"].(string)
	tokens := 0
	if ts, ok := result["tokens_predicted"].(float64); ok {
		tokens = int(ts)
	}
	return text, tokens, nil
}

// ── Worker pool ──────────────────────────────────────────────────────────────

type WorkerPool struct {
	backend    Backend
	semaphore  chan struct{}
	maxWorkers int
}

func NewWorkerPool(backend Backend, maxWorkers int) *WorkerPool {
	return &WorkerPool{
		backend:    backend,
		semaphore:  make(chan struct{}, maxWorkers),
		maxWorkers: maxWorkers,
	}
}

func (wp *WorkerPool) Submit(ctx context.Context, prompt string, maxTokens int) (string, int, time.Duration, error) {
	// Acquire slot
	select {
	case wp.semaphore <- struct{}{}:
		defer func() { <-wp.semaphore }()
	case <-ctx.Done():
		return "", 0, 0, ctx.Err()
	}

	metrics.InflightReqs.Add(1)
	defer metrics.InflightReqs.Add(-1)

	t0 := time.Now()
	text, tokens, err := wp.backend.Generate(ctx, prompt, maxTokens)
	latency := time.Since(t0)

	if err != nil {
		metrics.RecordRequest(0, latency.Milliseconds(), true)
	} else {
		metrics.RecordRequest(int64(tokens), latency.Milliseconds(), false)
	}
	return text, tokens, latency, err
}

// ── HTTP handlers ────────────────────────────────────────────────────────────

type Server struct {
	pool   *WorkerPool
	config *Config
	mux    *http.ServeMux
}

func NewServer(cfg *Config, pool *WorkerPool) *Server {
	s := &Server{pool: pool, config: cfg, mux: http.NewServeMux()}
	s.mux.HandleFunc("/health", s.healthHandler)
	s.mux.HandleFunc("/metrics", s.metricsHandler)
	s.mux.HandleFunc("/v1/completions", s.completionHandler)
	s.mux.HandleFunc("/v1/chat/completions", s.chatHandler)
	return s
}

func (s *Server) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	s.mux.ServeHTTP(w, r)
}

func (s *Server) healthHandler(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]string{
		"status":  "ok",
		"backend": s.pool.backend.Name(),
		"model":   s.config.ModelPath,
	})
}

func (s *Server) metricsHandler(w http.ResponseWriter, r *http.Request) {
	elapsed := time.Since(metrics.StartTime).Seconds()
	totalReqs := metrics.TotalRequests.Load()
	totalToks := metrics.TotalTokens.Load()
	totalErrs := metrics.TotalErrors.Load()
	totalLatMs := metrics.TotalLatencyMs.Load()

	avgLatMs := float64(0)
	if successReqs := totalReqs - totalErrs; successReqs > 0 {
		avgLatMs = float64(totalLatMs) / float64(successReqs)
	}
	tps := float64(totalToks) / math.Max(elapsed, 1)

	// Prometheus format
	fmt.Fprintf(w,
		"# HELP aneurologic_requests_total Total requests\n"+
			"aneurologic_requests_total %d\n"+
			"# HELP aneurologic_tokens_total Total tokens generated\n"+
			"aneurologic_tokens_total %d\n"+
			"# HELP aneurologic_errors_total Total errors\n"+
			"aneurologic_errors_total %d\n"+
			"# HELP aneurologic_tokens_per_sec Tokens per second\n"+
			"aneurologic_tokens_per_sec %.2f\n"+
			"# HELP aneurologic_avg_latency_ms Average latency ms\n"+
			"aneurologic_avg_latency_ms %.2f\n"+
			"# HELP aneurologic_inflight Inflight requests\n"+
			"aneurologic_inflight %d\n",
		totalReqs, totalToks, totalErrs, tps, avgLatMs,
		metrics.InflightReqs.Load(),
	)
}

func (s *Server) completionHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "POST only", http.StatusMethodNotAllowed)
		return
	}
	var req CompletionRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "bad request", http.StatusBadRequest)
		return
	}
	if req.MaxTokens <= 0 {
		req.MaxTokens = s.config.MaxTokens
	}

	text, tokens, latency, err := s.pool.Submit(r.Context(), req.Prompt, req.MaxTokens)
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": err.Error()})
		return
	}

	_ = latency
	writeJSON(w, http.StatusOK, CompletionResponse{
		ID:      fmt.Sprintf("cmpl-%d", time.Now().UnixNano()),
		Object:  "text_completion",
		Created: time.Now().Unix(),
		Model:   s.config.ModelPath,
		Choices: []CompletionChoice{{Text: text, Index: 0, FinishReason: "length"}},
		Usage:   UsageStats{CompletionTokens: tokens, TotalTokens: tokens},
	})
}

func (s *Server) chatHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "POST only", http.StatusMethodNotAllowed)
		return
	}
	var req CompletionRequest
	json.NewDecoder(r.Body).Decode(&req)

	// Build prompt from messages
	prompt := ""
	for _, m := range req.Messages {
		prompt += fmt.Sprintf("%s: %s\n", m.Role, m.Content)
	}
	prompt += "assistant:"

	if req.MaxTokens <= 0 {
		req.MaxTokens = s.config.MaxTokens
	}
	text, tokens, _, err := s.pool.Submit(r.Context(), prompt, req.MaxTokens)
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": err.Error()})
		return
	}
	writeJSON(w, http.StatusOK, CompletionResponse{
		ID:      fmt.Sprintf("chat-%d", time.Now().UnixNano()),
		Object:  "chat.completion",
		Created: time.Now().Unix(),
		Model:   s.config.ModelPath,
		Choices: []CompletionChoice{{Index: 0, FinishReason: "stop",
			Message: &Message{Role: "assistant", Content: text}}},
		Usage: UsageStats{CompletionTokens: tokens, TotalTokens: tokens},
	})
}

// ── Concurrent load test ─────────────────────────────────────────────────────

type BenchResult struct {
	Concurrency int
	NumRequests int
	TotalTimeMs float64
	TotalTokens int64
	Errors      int
	TPS         float64 // tokens per second (aggregate)
	RQPS        float64 // requests per second
	AvgLatMs    float64
	P50LatMs    float64
	P95LatMs    float64
}

func RunLoadTest(pool *WorkerPool, concurrency, numRequests, maxTokens int) BenchResult {
	prompts := []string{
		"Explain the key principles of edge AI optimization in three sentences.",
		"What are the benefits of neural network quantization for embedded systems?",
		"Describe how knowledge distillation can improve small model performance.",
		"List five advantages of using Go for high-performance API servers.",
		"How does TensorRT accelerate neural network inference on NVIDIA hardware?",
	}

	var (
		wg          sync.WaitGroup
		mu          sync.Mutex
		totalToks   int64
		errorCount  int
		latencies   []float64
	)

	t0 := time.Now()
	sem := make(chan struct{}, concurrency)

	for i := 0; i < numRequests; i++ {
		wg.Add(1)
		prompt := prompts[i%len(prompts)]
		sem <- struct{}{}
		go func(p string) {
			defer wg.Done()
			defer func() { <-sem }()

			ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
			defer cancel()

			_, tokens, latency, err := pool.Submit(ctx, p, maxTokens)
			mu.Lock()
			if err != nil {
				errorCount++
			} else {
				totalToks += int64(tokens)
				latencies = append(latencies, float64(latency.Milliseconds()))
			}
			mu.Unlock()
		}(prompt)
	}
	wg.Wait()

	elapsed := time.Since(t0)

	// Compute percentiles
	p50, p95 := percentiles(latencies)
	avgLat := average(latencies)
	successCount := numRequests - errorCount

	return BenchResult{
		Concurrency: concurrency,
		NumRequests: numRequests,
		TotalTimeMs: float64(elapsed.Milliseconds()),
		TotalTokens: totalToks,
		Errors:      errorCount,
		TPS:         float64(totalToks) / elapsed.Seconds(),
		RQPS:        float64(successCount) / elapsed.Seconds(),
		AvgLatMs:    avgLat,
		P50LatMs:    p50,
		P95LatMs:    p95,
	}
}

// ── Main ─────────────────────────────────────────────────────────────────────

func main() {
	cfg := &Config{}
	flag.StringVar(&cfg.Host, "host", "0.0.0.0", "Listen host")
	flag.IntVar(&cfg.Port, "port", 8080, "Listen port")
	flag.StringVar(&cfg.Backend, "backend", "ollama",
		"Backend: 'ollama' or 'llamacpp_server'")
	flag.StringVar(&cfg.ModelPath, "model", "lfm2.5",
		"Model path (GGUF) or Ollama model name")
	flag.StringVar(&cfg.BackendURL, "backend-url", "http://localhost:11434",
		"Backend URL (Ollama: :11434, llama-server: :8081)")
	flag.IntVar(&cfg.MaxWorkers, "workers", 4,
		"Max concurrent inference workers")
	flag.IntVar(&cfg.MaxTokens, "max-tokens", 128, "Default max tokens")

	benchMode := flag.Bool("bench", false,
		"Run concurrent load test instead of serving")
	benchConcurrency := flag.Int("bench-concurrency", 4, "Load test concurrency")
	benchRequests    := flag.Int("bench-requests", 20, "Load test total requests")

	flag.Parse()

	// Select backend
	var backend Backend
	switch cfg.Backend {
	case "ollama":
		backend = NewOllamaBackend(cfg.BackendURL, cfg.ModelPath)
		log.Printf("[phase5] Backend: Ollama @ %s  model=%s", cfg.BackendURL, cfg.ModelPath)
	case "llamacpp_server":
		backend = NewLlamaCppBackend(cfg.BackendURL)
		log.Printf("[phase5] Backend: llama-server @ %s", cfg.BackendURL)
	default:
		log.Fatalf("Unknown backend: %s (use 'ollama' or 'llamacpp_server')", cfg.Backend)
	}

	pool := NewWorkerPool(backend, cfg.MaxWorkers)

	if *benchMode {
		log.Printf("[phase5] Running load test: concurrency=%d requests=%d maxTokens=%d",
			*benchConcurrency, *benchRequests, cfg.MaxTokens)

		result := RunLoadTest(pool, *benchConcurrency, *benchRequests, cfg.MaxTokens)

		fmt.Printf("\n"+
			"┌─────────────────────────────────────────────┐\n"+
			"│   ANeurologic Go Server — Load Test Results │\n"+
			"├─────────────────────────────────────────────┤\n"+
			"│  Backend      : %-29s│\n"+
			"│  Concurrency  : %-29d│\n"+
			"│  Requests     : %-29d│\n"+
			"│  Errors       : %-29d│\n"+
			"│  Total tokens : %-29d│\n"+
			"│  Total time   : %-28.1f s│\n"+
			"│  Throughput   : %-27.1f t/s│\n"+
			"│  Req/sec      : %-28.2f │\n"+
			"│  Avg latency  : %-28.1f ms│\n"+
			"│  P50 latency  : %-28.1f ms│\n"+
			"│  P95 latency  : %-28.1f ms│\n"+
			"└─────────────────────────────────────────────┘\n",
			backend.Name(), result.Concurrency, result.NumRequests,
			result.Errors, result.TotalTokens,
			result.TotalTimeMs/1000, result.TPS, result.RQPS,
			result.AvgLatMs, result.P50LatMs, result.P95LatMs,
		)

		// Save JSON result
		outPath := "/workspace/outputs/logs/stage4_go_bench.json"
		os.MkdirAll("/workspace/outputs/logs", 0755)
		data, _ := json.MarshalIndent(map[string]interface{}{
			"stage":   "Stage 4 - Go Inference",
			"backend": backend.Name(),
			"result":  result,
		}, "", "  ")
		os.WriteFile(outPath, data, 0644)
		fmt.Printf("  Results saved → %s\n", outPath)
		return
	}

	// Serve mode
	srv := NewServer(cfg, pool)
	addr := fmt.Sprintf("%s:%d", cfg.Host, cfg.Port)
	log.Printf("[phase5] Go inference server starting on %s  workers=%d",
		addr, cfg.MaxWorkers)
	log.Printf("[phase5] Endpoints: POST /v1/completions  POST /v1/chat/completions")
	log.Printf("[phase5] Metrics:   GET /metrics  GET /health")

	if err := http.ListenAndServe(addr, srv); err != nil {
		log.Fatalf("Server error: %v", err)
	}
}

// ── Helpers ──────────────────────────────────────────────────────────────────

func writeJSON(w http.ResponseWriter, status int, v interface{}) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(v)
}

func jsonReader(data []byte) io.Reader {
	return &byteReader{data: data}
}

type byteReader struct {
	data   []byte
	offset int
}

func (r *byteReader) Read(p []byte) (n int, err error) {
	if r.offset >= len(r.data) {
		return 0, io.EOF
	}
	n = copy(p, r.data[r.offset:])
	r.offset += n
	return n, nil
}

func percentiles(data []float64) (p50, p95 float64) {
	if len(data) == 0 {
		return 0, 0
	}
	sorted := make([]float64, len(data))
	copy(sorted, data)
	// Simple insertion sort (small data sets)
	for i := 1; i < len(sorted); i++ {
		for j := i; j > 0 && sorted[j-1] > sorted[j]; j-- {
			sorted[j-1], sorted[j] = sorted[j], sorted[j-1]
		}
	}
	p50 = sorted[int(0.50*float64(len(sorted)))]
	p95 = sorted[int(0.95*float64(len(sorted)))]
	return
}

func average(data []float64) float64 {
	if len(data) == 0 {
		return 0
	}
	sum := 0.0
	for _, v := range data {
		sum += v
	}
	return sum / float64(len(data))
}
