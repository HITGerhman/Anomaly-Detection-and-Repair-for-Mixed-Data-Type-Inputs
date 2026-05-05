package agent

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

func TestLangGraphClientHealthParsesResponse(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/health" {
			http.NotFound(w, r)
			return
		}
		_, _ = w.Write([]byte(`{"status":"ok","service":"langgraph-sidecar","planner_mode":"llm","llm_mode":"configured","model":"gpt-test","ready":true,"graph_id":"phase_c_cognition_graph","version":"phase_c"}`))
	}))
	defer server.Close()

	client := NewLangGraphClient(server.URL, time.Second)
	health, err := client.Health(context.Background())
	if err != nil {
		t.Fatalf("Health failed: %v", err)
	}
	if health.GraphID != "phase_c_cognition_graph" {
		t.Fatalf("unexpected graph id: %s", health.GraphID)
	}
	if health.LLMMode != "configured" || health.Model != "gpt-test" {
		t.Fatalf("unexpected llm health payload: %+v", health)
	}
}

func TestLangGraphClientPlanParsesResponse(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/plan" {
			http.NotFound(w, r)
			return
		}
		_, _ = w.Write([]byte(`{"strategy_label":"neighbor_similarity","selected_candidate_id":"candidate-rule","reason_codes":["selected_rule"],"risk_note":"validation first","intent_label":"auto_repair","one_sentence_summary":"Selected rule.","short_bullets":["one","two"],"approval_needed":false}`))
	}))
	defer server.Close()

	client := NewLangGraphClient(server.URL, time.Second)
	resp, err := client.Plan(context.Background(), LangGraphPlanRequest{})
	if err != nil {
		t.Fatalf("Plan failed: %v", err)
	}
	if resp.SelectedCandidateID != "candidate-rule" {
		t.Fatalf("unexpected selected candidate: %s", resp.SelectedCandidateID)
	}
	if resp.IntentLabel != "auto_repair" || resp.RiskNote != "validation first" {
		t.Fatalf("unexpected plan response: %+v", resp)
	}
}

func TestLangGraphClientExplainParsesResponse(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/explain" {
			http.NotFound(w, r)
			return
		}
		_, _ = w.Write([]byte(`{"summary":"Short summary.","final_message":"Short final message.","short_bullets":["one","two"],"reason_codes":["selected_rule"],"risk_note":"validation first"}`))
	}))
	defer server.Close()

	client := NewLangGraphClient(server.URL, time.Second)
	resp, err := client.Explain(context.Background(), LangGraphExplainRequest{})
	if err != nil {
		t.Fatalf("Explain failed: %v", err)
	}
	if resp.FinalMessage != "Short final message." || resp.RiskNote != "validation first" {
		t.Fatalf("unexpected explain response: %+v", resp)
	}
}

func TestLangGraphClientReturnsErrorsForBadResponses(t *testing.T) {
	tests := []struct {
		name       string
		path       string
		statusCode int
		body       string
		call       func(*LangGraphClient) error
	}{
		{
			name:       "health_non_200",
			path:       "/health",
			statusCode: http.StatusBadGateway,
			body:       `bad gateway`,
			call: func(client *LangGraphClient) error {
				_, err := client.Health(context.Background())
				return err
			},
		},
		{
			name:       "plan_invalid_json",
			path:       "/v1/plan",
			statusCode: http.StatusOK,
			body:       `not-json`,
			call: func(client *LangGraphClient) error {
				_, err := client.Plan(context.Background(), LangGraphPlanRequest{})
				return err
			},
		},
		{
			name:       "plan_timeout",
			path:       "/v1/plan",
			statusCode: http.StatusOK,
			body:       `{"strategy_label":"late"}`,
			call: func(client *LangGraphClient) error {
				_, err := client.Plan(context.Background(), LangGraphPlanRequest{})
				return err
			},
		},
		{
			name:       "explain_invalid_json",
			path:       "/v1/explain",
			statusCode: http.StatusOK,
			body:       `not-json`,
			call: func(client *LangGraphClient) error {
				_, err := client.Explain(context.Background(), LangGraphExplainRequest{})
				return err
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.URL.Path != tc.path {
					http.NotFound(w, r)
					return
				}
				if tc.name == "plan_timeout" {
					time.Sleep(150 * time.Millisecond)
				}
				w.WriteHeader(tc.statusCode)
				_, _ = w.Write([]byte(tc.body))
			}))
			defer server.Close()

			timeout := 50 * time.Millisecond
			if tc.name != "plan_timeout" {
				timeout = time.Second
			}
			client := NewLangGraphClient(server.URL, timeout)
			if err := tc.call(client); err == nil {
				t.Fatalf("expected error for %s", tc.name)
			}
		})
	}
}

func TestLangGraphClientRejectsInvalidHealthPayload(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"status":"ok","service":"other","planner_mode":"fallback","ready":true}`))
	}))
	defer server.Close()

	client := NewLangGraphClient(server.URL, time.Second)
	if _, err := client.Health(context.Background()); err == nil {
		t.Fatalf("expected invalid payload error")
	}
}

func TestLangGraphClientRejectsPlanWithoutStrategyLabel(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"selected_candidate_id":"candidate-rule"}`))
	}))
	defer server.Close()

	client := NewLangGraphClient(server.URL, time.Second)
	if _, err := client.Plan(context.Background(), LangGraphPlanRequest{}); err == nil {
		t.Fatalf("expected missing strategy label error")
	}
}

func TestLangGraphClientRejectsPlanWithoutSelectedCandidateID(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"strategy_label":"deterministic_rule"}`))
	}))
	defer server.Close()

	client := NewLangGraphClient(server.URL, time.Second)
	if _, err := client.Plan(context.Background(), LangGraphPlanRequest{}); err == nil {
		t.Fatalf("expected missing selected candidate id error")
	}
}

func TestLangGraphClientRoundTripsStructuredRequest(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/plan" {
			http.NotFound(w, r)
			return
		}
		if r.Method != http.MethodPost {
			t.Fatalf("unexpected method: %s", r.Method)
		}
		_, _ = w.Write([]byte(fmt.Sprintf(`{"strategy_label":"hybrid_balanced","selected_candidate_id":"candidate-hybrid","reason_codes":["phase_c_llm"],"risk_note":"safe","intent_label":"auto_repair","one_sentence_summary":"ok","short_bullets":["%s"],"approval_needed":false}`, r.Header.Get("Content-Type"))))
	}))
	defer server.Close()

	client := NewLangGraphClient(server.URL, time.Second)
	resp, err := client.Plan(context.Background(), LangGraphPlanRequest{})
	if err != nil {
		t.Fatalf("Plan failed: %v", err)
	}
	if len(resp.ShortBullets) != 1 || resp.ShortBullets[0] != "application/json" {
		t.Fatalf("unexpected bullets: %#v", resp.ShortBullets)
	}
}
