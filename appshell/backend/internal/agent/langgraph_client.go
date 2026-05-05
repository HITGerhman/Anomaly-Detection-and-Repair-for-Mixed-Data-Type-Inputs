package agent

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

type planCaller interface {
	Plan(ctx context.Context, req LangGraphPlanRequest) (LangGraphPlanResponse, error)
}

type explainCaller interface {
	Explain(ctx context.Context, req LangGraphExplainRequest) (LangGraphExplainResponse, error)
}

type healthChecker interface {
	Health(ctx context.Context) (LangGraphHealth, error)
}

type LangGraphClient struct {
	baseURL string
	client  *http.Client
}

func NewLangGraphClient(baseURL string, timeout time.Duration) *LangGraphClient {
	if timeout <= 0 {
		timeout = defaultLangGraphRequestTimeout
	}
	return &LangGraphClient{
		baseURL: strings.TrimRight(strings.TrimSpace(baseURL), "/"),
		client: &http.Client{
			Timeout: timeout,
		},
	}
}

func (c *LangGraphClient) Health(ctx context.Context) (LangGraphHealth, error) {
	if c == nil {
		return LangGraphHealth{}, fmt.Errorf("langgraph client is nil")
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, c.baseURL+"/health", nil)
	if err != nil {
		return LangGraphHealth{}, err
	}
	resp, err := c.client.Do(req)
	if err != nil {
		return LangGraphHealth{}, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return LangGraphHealth{}, fmt.Errorf("langgraph health returned status %d", resp.StatusCode)
	}
	var payload LangGraphHealth
	if err := json.NewDecoder(resp.Body).Decode(&payload); err != nil {
		return LangGraphHealth{}, fmt.Errorf("decode langgraph health failed: %w", err)
	}
	if strings.TrimSpace(payload.Status) != "ok" || strings.TrimSpace(payload.Service) != "langgraph-sidecar" || !payload.Ready {
		return LangGraphHealth{}, fmt.Errorf("langgraph sidecar returned invalid health payload")
	}
	return payload, nil
}

func (c *LangGraphClient) Plan(ctx context.Context, reqPayload LangGraphPlanRequest) (LangGraphPlanResponse, error) {
	if c == nil {
		return LangGraphPlanResponse{}, fmt.Errorf("langgraph client is nil")
	}
	body, err := json.Marshal(reqPayload)
	if err != nil {
		return LangGraphPlanResponse{}, fmt.Errorf("marshal langgraph plan request failed: %w", err)
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+"/v1/plan", bytes.NewReader(body))
	if err != nil {
		return LangGraphPlanResponse{}, err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := c.client.Do(req)
	if err != nil {
		return LangGraphPlanResponse{}, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		raw, _ := io.ReadAll(resp.Body)
		return LangGraphPlanResponse{}, fmt.Errorf("langgraph plan returned status %d: %s", resp.StatusCode, strings.TrimSpace(string(raw)))
	}
	var payload LangGraphPlanResponse
	if err := json.NewDecoder(resp.Body).Decode(&payload); err != nil {
		return LangGraphPlanResponse{}, fmt.Errorf("decode langgraph plan response failed: %w", err)
	}
	if strings.TrimSpace(payload.StrategyLabel) == "" {
		return LangGraphPlanResponse{}, fmt.Errorf("langgraph plan response missing strategy_label")
	}
	if strings.TrimSpace(payload.SelectedCandidateID) == "" {
		return LangGraphPlanResponse{}, fmt.Errorf("langgraph plan response missing selected_candidate_id")
	}
	if len(payload.ShortBullets) > 3 {
		payload.ShortBullets = payload.ShortBullets[:3]
	}
	return payload, nil
}

func (c *LangGraphClient) Explain(ctx context.Context, reqPayload LangGraphExplainRequest) (LangGraphExplainResponse, error) {
	if c == nil {
		return LangGraphExplainResponse{}, fmt.Errorf("langgraph client is nil")
	}
	body, err := json.Marshal(reqPayload)
	if err != nil {
		return LangGraphExplainResponse{}, fmt.Errorf("marshal langgraph explain request failed: %w", err)
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+"/v1/explain", bytes.NewReader(body))
	if err != nil {
		return LangGraphExplainResponse{}, err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := c.client.Do(req)
	if err != nil {
		return LangGraphExplainResponse{}, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		raw, _ := io.ReadAll(resp.Body)
		return LangGraphExplainResponse{}, fmt.Errorf("langgraph explain returned status %d: %s", resp.StatusCode, strings.TrimSpace(string(raw)))
	}
	var payload LangGraphExplainResponse
	if err := json.NewDecoder(resp.Body).Decode(&payload); err != nil {
		return LangGraphExplainResponse{}, fmt.Errorf("decode langgraph explain response failed: %w", err)
	}
	if strings.TrimSpace(payload.Summary) == "" && strings.TrimSpace(payload.FinalMessage) == "" {
		return LangGraphExplainResponse{}, fmt.Errorf("langgraph explain response missing summary and final_message")
	}
	if len(payload.ShortBullets) > 3 {
		payload.ShortBullets = payload.ShortBullets[:3]
	}
	return payload, nil
}
