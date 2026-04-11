package agent

import (
	"context"

	"appshell/backend/internal/engine"
)

type previewToolSpec struct {
	ToolID      string
	CallSummary string
	DoneSummary string
	Payload     map[string]any
}

type previewToolOutcome struct {
	Spec previewToolSpec
	Resp engine.Response
	Err  error
}

type indexedPreviewOutcome struct {
	Index   int
	Outcome previewToolOutcome
}

func (r *RuntimeRunner) runPreviewTools(ctx context.Context, parentTaskID string, specs []previewToolSpec, retrieveMode string) []previewToolOutcome {
	outcomes := make([]previewToolOutcome, len(specs))
	if !retrieveModeRunsInParallel(retrieveMode) {
		for idx, spec := range specs {
			resp, err := r.callTool(ctx, parentTaskID, spec.ToolID, spec.Payload)
			outcomes[idx] = previewToolOutcome{
				Spec: spec,
				Resp: resp,
				Err:  err,
			}
		}
		return outcomes
	}

	resultCh := make(chan indexedPreviewOutcome, len(specs))
	for idx, spec := range specs {
		go func(index int, item previewToolSpec) {
			resp, err := r.callTool(ctx, parentTaskID, item.ToolID, item.Payload)
			resultCh <- indexedPreviewOutcome{
				Index: index,
				Outcome: previewToolOutcome{
					Spec: item,
					Resp: resp,
					Err:  err,
				},
			}
		}(idx, spec)
	}

	for range specs {
		result := <-resultCh
		outcomes[result.Index] = result.Outcome
	}
	return outcomes
}
