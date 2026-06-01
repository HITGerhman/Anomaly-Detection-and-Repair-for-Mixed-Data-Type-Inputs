const pptxgen = require("pptxgenjs");
const path = require("path");

const pptx = new pptxgen();
pptx.layout = "LAYOUT_WIDE";
pptx.author = "Anomaly Detection and Repair Project";
pptx.subject = "Large-scale stability validation update";
pptx.title = "Large-scale Stability Validation Update";
pptx.company = "Graduation Project";
pptx.lang = "en-US";
pptx.theme = {
  headFontFace: "Aptos Display",
  bodyFontFace: "Aptos",
  lang: "en-US",
};
pptx.defineLayout({ name: "LAYOUT_WIDE", width: 13.333, height: 7.5 });
pptx.layout = "LAYOUT_WIDE";

const C = {
  ink: "17202A",
  muted: "52606D",
  pale: "F4F7FA",
  line: "D7DEE8",
  blue: "2458A6",
  teal: "168A78",
  amber: "B36B00",
  red: "B42318",
  white: "FFFFFF",
};

function title(slide, text, kicker) {
  if (kicker) {
    slide.addText(kicker, { x: 0.55, y: 0.35, w: 5.5, h: 0.25, fontFace: "Aptos", fontSize: 9, color: C.blue, bold: true, margin: 0 });
  }
  slide.addText(text, { x: 0.55, y: 0.62, w: 8.5, h: 0.5, fontFace: "Aptos Display", fontSize: 25, color: C.ink, bold: true, margin: 0 });
  slide.addShape(pptx.ShapeType.line, { x: 0.55, y: 1.2, w: 12.2, h: 0, line: { color: C.line, width: 1 } });
}

function footer(slide, n) {
  slide.addText(`2026-05-17 stability update | ${n}`, { x: 0.55, y: 7.08, w: 4.8, h: 0.2, fontSize: 8, color: C.muted, margin: 0 });
  slide.addText("Source: outputs/stability_reprobe_20260517/reprobe_summary.json", { x: 6.1, y: 7.08, w: 6.6, h: 0.2, fontSize: 8, color: C.muted, align: "right", margin: 0 });
}

function metric(slide, value, label, x, y, color) {
  slide.addText(value, { x, y, w: 2.15, h: 0.42, fontSize: 24, color, bold: true, margin: 0, fit: "shrink" });
  slide.addText(label, { x, y: y + 0.48, w: 2.2, h: 0.42, fontSize: 9.5, color: C.muted, margin: 0, fit: "shrink" });
}

function addSmallNote(slide, text, x, y, w) {
  slide.addText(text, { x, y, w, h: 0.55, fontSize: 10.5, color: C.muted, breakLine: false, fit: "shrink", margin: 0.02 });
}

function addTable(slide, rows, x, y, w, h, colW) {
  slide.addTable(rows, {
    x, y, w, h,
    colW,
    border: { type: "solid", color: C.line, width: 0.5 },
    margin: 0.05,
    fontFace: "Aptos",
    fontSize: 8.5,
    color: C.ink,
    valign: "mid",
    fit: "shrink",
  });
}

// Slide 1
{
  const slide = pptx.addSlide();
  slide.background = { color: C.white };
  slide.addShape(pptx.ShapeType.rect, { x: 0, y: 0, w: 13.333, h: 0.16, fill: { color: C.teal }, line: { color: C.teal } });
  slide.addText("Large-scale Stability Validation", { x: 0.72, y: 0.78, w: 8.8, h: 0.82, fontFace: "Aptos Display", fontSize: 34, bold: true, color: C.ink, margin: 0 });
  slide.addText("500k / 1M / 10M mixed-type CSV runs with streaming repair, Validation Gate, and rollback metadata", { x: 0.74, y: 1.72, w: 8.6, h: 0.5, fontSize: 16, color: C.muted, margin: 0 });
  metric(slide, "10,000,024", "rows in largest auto run", 0.78, 3.02, C.blue);
  metric(slide, "1.19 GB", "repaired CSV output", 3.24, 3.02, C.teal);
  metric(slide, "584.805 s", "10M end-to-end time", 5.68, 3.02, C.amber);
  metric(slide, "rollback", "manifest preserved", 8.1, 3.02, C.red);
  slide.addShape(pptx.ShapeType.rect, { x: 0.74, y: 4.65, w: 11.9, h: 0.02, fill: { color: C.line }, line: { color: C.line } });
  addSmallNote(slide, "Claim boundary: this proves the tested local pipeline can complete a 10M-row run; it is not a production-readiness or unlimited-scale claim.", 0.78, 5.05, 10.9);
  footer(slide, 1);
}

// Slide 2
{
  const slide = pptx.addSlide();
  title(slide, "End-to-end auto-session evidence", "REAL WRITES, NOT PLAN-ONLY");
  const rows = [
    [
      { text: "Run", options: { bold: true, fill: { color: C.pale } } },
      { text: "Rows", options: { bold: true, fill: { color: C.pale } } },
      { text: "Output", options: { bold: true, fill: { color: C.pale } } },
      { text: "Total", options: { bold: true, fill: { color: C.pale } } },
      { text: "Write", options: { bold: true, fill: { color: C.pale } } },
      { text: "Post validation", options: { bold: true, fill: { color: C.pale } } },
      { text: "Verdict", options: { bold: true, fill: { color: C.pale } } },
    ],
    ["500k", "500,024", "59.25 MB", "42.761 s", "pandas_full", "scoped + full", "warn"],
    ["1M", "1,000,024", "118.50 MB", "62.255 s", "streaming", "scoped + full", "warn"],
    ["10M", "10,000,024", "1.19 GB", "584.805 s", "streaming", "affected columns", "warn"],
  ];
  addTable(slide, rows, 0.65, 1.62, 12.0, 2.25, [1.25, 1.55, 1.45, 1.25, 1.35, 2.25, 1.0]);
  slide.addText("How to describe it", { x: 0.7, y: 4.3, w: 3.1, h: 0.26, fontSize: 13, bold: true, color: C.ink, margin: 0 });
  slide.addText("The 10M run wrote a real repaired CSV, used streaming output, and kept rollback metadata. The post validation was explicitly incremental, not a second full scan.", { x: 0.7, y: 4.68, w: 11.2, h: 0.72, fontSize: 15, color: C.ink, margin: 0.02, fit: "shrink" });
  footer(slide, 2);
}

// Slide 3
{
  const slide = pptx.addSlide();
  title(slide, "Preview path got materially faster", "ENGINE PROBES");
  const rows = [
    [
      { text: "10M operation", options: { bold: true, fill: { color: C.pale } } },
      { text: "Before", options: { bold: true, fill: { color: C.pale } } },
      { text: "After", options: { bold: true, fill: { color: C.pale } } },
      { text: "Speedup", options: { bold: true, fill: { color: C.pale } } },
      { text: "Main reason", options: { bold: true, fill: { color: C.pale } } },
    ],
    ["repair_batch plan-only", "146.192 s", "27.674 s", "5.28x", "precomputed issues + lightweight comparison"],
    ["Gower single issue", "93.140 s", "31.180 s", "2.99x", "bucket prefilter + 512 sample cap"],
    ["MissForest single issue", "189.083 s", "29.720 s", "6.36x", "21 encoded features + compact frame"],
  ];
  addTable(slide, rows, 0.65, 1.6, 12.0, 2.35, [2.4, 1.35, 1.35, 1.05, 4.0]);
  metric(slide, "21", "MissForest encoded features after policy", 0.75, 4.55, C.blue);
  metric(slide, "5,012", "10M MissForest working rows", 3.35, 4.55, C.teal);
  metric(slide, "512", "Gower candidate sample cap", 5.95, 4.55, C.amber);
  metric(slide, "833k", "10M Gower prefiltered pool", 8.55, 4.55, C.red);
  footer(slide, 3);
}

// Slide 4
{
  const slide = pptx.addSlide();
  title(slide, "Incremental validation is guarded", "SAFETY UPDATE");
  slide.addText("Large output path", { x: 0.72, y: 1.55, w: 2.4, h: 0.26, fontSize: 13, bold: true, color: C.ink, margin: 0 });
  slide.addText("For outputs above 512 MiB, the runtime may run affected-column post validation instead of a second full-table scan.", { x: 0.72, y: 1.9, w: 5.2, h: 0.8, fontSize: 14, color: C.ink, margin: 0.03, fit: "shrink" });
  slide.addText("New reject rule", { x: 6.55, y: 1.55, w: 2.4, h: 0.26, fontSize: 13, bold: true, color: C.ink, margin: 0 });
  slide.addText("If any repaired column has a higher issue count after repair than in the baseline scan, Validation Gate rejects and rolls back.", { x: 6.55, y: 1.9, w: 5.35, h: 0.8, fontSize: 14, color: C.ink, margin: 0.03, fit: "shrink" });
  const ruleRows = [
    [
      { text: "Signal", options: { bold: true, fill: { color: C.pale } } },
      { text: "Meaning", options: { bold: true, fill: { color: C.pale } } },
    ],
    ["post_scan_incremental_estimate", "Validation is affected-column scoped, not a full post scan."],
    ["affected_column_issue_count_increased", "Touched column got worse; output is unsafe."],
    ["rollback manifest", "Written outputs can be restored after rejection."],
  ];
  addTable(slide, ruleRows, 0.72, 3.35, 11.5, 1.8, [3.5, 6.6]);
  addSmallNote(slide, "Thesis wording: the system does not hide the validation boundary; it records the scope and rejects local side effects.", 0.76, 5.65, 10.9);
  footer(slide, 4);
}

// Slide 5
{
  const slide = pptx.addSlide();
  title(slide, "What the thesis can safely claim", "WRITING GUIDE");
  const safe = [
    "Completed real 500k, 1M, and 10M AppShell auto runs on the tested Windows machine.",
    "Wrote a 1.19 GB repaired CSV for the 10M case and preserved rollback metadata.",
    "Improved large-file previews through schema cache, lightweight comparison, Gower prefiltering, and compact MissForest.",
  ];
  const avoid = [
    "Do not claim production readiness or arbitrary billion-row scalability.",
    "Do not equate affected-column incremental validation with a full post scan.",
    "Do not claim every anomaly type is safe for automatic repair.",
  ];
  slide.addText("Safe claims", { x: 0.78, y: 1.55, w: 3.0, h: 0.28, fontSize: 14, bold: true, color: C.teal, margin: 0 });
  slide.addText(safe.map((text) => ({ text, options: { bullet: { type: "ul" }, breakLine: true } })), { x: 0.85, y: 1.95, w: 5.4, h: 2.3, fontSize: 13.5, color: C.ink, breakLine: false, fit: "shrink", margin: 0.02, paraSpaceAfterPt: 8 });
  slide.addText("Avoid", { x: 6.85, y: 1.55, w: 3.0, h: 0.28, fontSize: 14, bold: true, color: C.red, margin: 0 });
  slide.addText(avoid.map((text) => ({ text, options: { bullet: { type: "ul" }, breakLine: true } })), { x: 6.92, y: 1.95, w: 5.3, h: 2.3, fontSize: 13.5, color: C.ink, breakLine: false, fit: "shrink", margin: 0.02, paraSpaceAfterPt: 8 });
  slide.addShape(pptx.ShapeType.line, { x: 0.75, y: 4.75, w: 11.8, h: 0, line: { color: C.line, width: 1 } });
  addSmallNote(slide, "Detailed source for paper writing: docs/large_scale_stability_20260517.md", 0.8, 5.25, 8.0);
  footer(slide, 5);
}

const out = path.join(__dirname, "Scale_Validation_Update_2026-05-17.pptx");
pptx.writeFile({ fileName: out }).then(() => {
  console.log(out);
});
