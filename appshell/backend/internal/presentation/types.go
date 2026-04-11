package presentation

type Bundle struct {
	Version    string         `json:"version"`
	Kind       string         `json:"kind"`
	Headline   string         `json:"headline"`
	Summary    string         `json:"summary"`
	Verdict    string         `json:"verdict"`
	Highlights []Highlight    `json:"highlights"`
	Sections   []Section      `json:"sections"`
	Charts     []ChartSpec    `json:"charts"`
	Artifacts  map[string]any `json:"artifacts,omitempty"`
}

type Highlight struct {
	ID    string `json:"id"`
	Label string `json:"label"`
	Value string `json:"value"`
	Tone  string `json:"tone,omitempty"`
	Hint  string `json:"hint,omitempty"`
}

type Section struct {
	ID           string   `json:"id"`
	Title        string   `json:"title"`
	Body         string   `json:"body"`
	Bullets      []string `json:"bullets,omitempty"`
	EvidenceRefs []string `json:"evidence_refs,omitempty"`
}

type ChartSpec struct {
	ID         string         `json:"id"`
	Kind       string         `json:"kind"`
	Title      string         `json:"title"`
	Subtitle   string         `json:"subtitle,omitempty"`
	EmptyState string         `json:"empty_state,omitempty"`
	Data       map[string]any `json:"data"`
}
