const fs = require("fs");
const path = require("path");
const { spawnSync } = require("child_process");

const repo = path.resolve(__dirname, "..");
const workspace = path.join(repo, "outputs", "manual-large-labeled-deck", "presentations", "large-labeled");
const slidesDir = path.join(workspace, "slides");
const previewDir = path.join(workspace, "preview");
const layoutDir = path.join(workspace, "layout");
const out = path.join(__dirname, "Large_Labeled_Validation_Update_2026-05-31.pptx");

function findPresentationBuilder() {
  const home = process.env.HOME || process.env.USERPROFILE;
  if (!home) throw new Error("HOME or USERPROFILE is required to locate the Presentations runtime.");
  const root = path.join(home, ".codex", "plugins", "cache", "openai-primary-runtime", "presentations");
  const versions = fs.readdirSync(root).sort().reverse();
  for (const version of versions) {
    const candidate = path.join(root, version, "skills", "presentations", "scripts", "build_artifact_deck.mjs");
    if (fs.existsSync(candidate)) return candidate;
  }
  throw new Error(`Could not find build_artifact_deck.mjs under ${root}`);
}

function write(name, text) {
  fs.mkdirSync(slidesDir, { recursive: true });
  fs.writeFileSync(path.join(slidesDir, name), `${text.trim()}\n`, "utf8");
}

write("theme.mjs", String.raw`
export const C = { ink:"#17202A", muted:"#52606D", pale:"#F4F7FA", line:"#D7DEE8", blue:"#2458A6", teal:"#168A78", amber:"#B36B00", red:"#B42318", white:"#FFFFFF" };
export function base(slide, ctx) { ctx.addShape(slide,{x:0,y:0,w:1280,h:720,fill:C.white}); ctx.addShape(slide,{x:0,y:0,w:1280,h:14,fill:C.teal}); }
export function title(slide, ctx, text, kicker) { if (kicker) ctx.addText(slide,{text:kicker,x:54,y:32,w:640,h:24,fontSize:14,color:C.blue,bold:true}); ctx.addText(slide,{text,x:54,y:62,w:940,h:42,fontSize:32,color:C.ink,bold:true,typeface:ctx.fonts.title}); ctx.addShape(slide,{x:54,y:122,w:1170,h:2,fill:C.line}); }
export function footer(slide, ctx, n) { ctx.addText(slide,{text:"2026-05-31 labeled scale update | "+n,x:54,y:682,w:460,h:20,fontSize:11,color:C.muted}); ctx.addText(slide,{text:"Source: artifacts/experiments/large_labeled_validation",x:710,y:682,w:510,h:20,fontSize:11,color:C.muted,align:"right"}); }
export function metric(slide, ctx, value, label, x, y, color) { ctx.addText(slide,{text:value,x,y,w:220,h:44,fontSize:30,color,bold:true}); ctx.addText(slide,{text:label,x,y:y+48,w:230,h:44,fontSize:14,color:C.muted}); }
export function table(slide, ctx, rows, x, y, colW, rowH=34) { let top=y; rows.forEach((row,r)=>{ let left=x; row.forEach((cell,c)=>{ ctx.addShape(slide,{x:left,y:top,w:colW[c],h:rowH,fill:r===0?C.pale:C.white,line:{fill:C.line,width:1}}); ctx.addText(slide,{text:String(cell),x:left+6,y:top+7,w:colW[c]-12,h:rowH-10,fontSize:12,color:C.ink,bold:r===0}); left+=colW[c]; }); top+=rowH; }); }
export function bullets(slide, ctx, items, x, y, w, color=C.ink) { items.forEach((item,i)=>{ ctx.addText(slide,{text:"•",x,y:y+i*42,w:18,h:24,fontSize:18,color}); ctx.addText(slide,{text:item,x:x+24,y:y+i*42,w,h:34,fontSize:16,color}); }); }
`);

write("slide-01.mjs", String.raw`
import { C, base, footer, metric } from "./theme.mjs";
export async function slide01(presentation, ctx) {
  const slide = presentation.slides.add(); base(slide, ctx);
  ctx.addText(slide,{text:"Large-scale Labeled Validation",x:70,y:76,w:900,h:64,fontSize:42,color:C.ink,bold:true,typeface:ctx.fonts.title});
  ctx.addText(slide,{text:"Ground-truth injection bridge between controlled accuracy and 10M stability evidence",x:72,y:158,w:930,h:42,fontSize:20,color:C.muted});
  metric(slide,ctx,"100","injected anomalies per scale",74,292,C.blue);
  metric(slide,ctx,"1M + 10M","labeled generated datasets",304,292,C.teal);
  metric(slide,ctx,"100/100","injected anomalies recalled",590,292,C.amber);
  metric(slide,ctx,"72","1M repairable GT cells",860,292,C.red);
  ctx.addShape(slide,{x:74,y:468,w:1140,h:2,fill:C.line});
  ctx.addText(slide,{text:"Scope boundary: 10M is labeled detection-only. Repair accuracy at 10M scale remains future work.",x:78,y:508,w:1060,h:42,fontSize:17,color:C.ink});
  footer(slide,ctx,1); return slide;
}
`);

write("slide-02.mjs", String.raw`
import { C, base, title, footer, table, bullets } from "./theme.mjs";
export async function slide02(presentation, ctx) {
  const slide = presentation.slides.add(); base(slide, ctx); title(slide, ctx, "Detection metrics at scale", "LABELED SCAN RESULTS");
  table(slide,ctx,[["Dataset","Rows","GT","Pred","TP","FP","Precision","Recall","F1"],["1M labeled","1,000,012","100","7,680","100","7,580","0.013021","1.000000","0.025707"],["10M labeled","10,000,012","100","75,864","100","75,764","0.001318","1.000000","0.002633"]],54,160,[150,130,58,78,58,86,110,100,90],38);
  ctx.addText(slide,{text:"Interpretation",x:72,y:356,w:260,h:26,fontSize:17,bold:true,color:C.ink});
  bullets(slide,ctx,["All injected anomalies were recalled in both labeled scale runs.","Missing, rare-category, duplicate, and cross-column findings were exact in this generated setup.","Numeric outlier precision collapsed because naturally high total_amount values are also flagged."],86,402,1030);
  footer(slide,ctx,2); return slide;
}
`);

write("slide-03.mjs", String.raw`
import { C, base, title, footer, table, metric } from "./theme.mjs";
export async function slide03(presentation, ctx) {
  const slide = presentation.slides.add(); base(slide, ctx); title(slide, ctx, "1M repair accuracy and side effects", "CONTROLLED REPAIR EVALUATION");
  table(slide,ctx,[["Type","GT","Changed","Exact","Improved/Exact","Non-GT modified"],["missing","30","30","5","5","0"],["numeric","24","24","0","24","7,580"],["rare category","18","18","2","2","0"],["overall","72","72","7","31","7,580"]],64,160,[190,80,100,90,160,170],38);
  metric(slide,ctx,"0.430556","overall improved-or-exact rate",84,458,C.teal);
  metric(slide,ctx,"7,652","total cells modified",342,458,C.blue);
  metric(slide,ctx,"7,580","non-GT numeric side effects",590,458,C.red);
  metric(slide,ctx,"yes","rollback manifest generated",854,458,C.amber);
  footer(slide,ctx,3); return slide;
}
`);

write("slide-04.mjs", String.raw`
import { C, base, title, footer, table } from "./theme.mjs";
export async function slide04(presentation, ctx) {
  const slide = presentation.slides.add(); base(slide, ctx); title(slide, ctx, "Runtime and memory evidence", "RESOURCE OBSERVATION");
  table(slide,ctx,[["Dataset","Stage","Runtime","Peak working set","Peak private"],["1M","scan + GT","11.481 s","494.695 MB","966.125 MB"],["1M","repair + GT","17.074 s","523.504 MB","1007.016 MB"],["10M","generate","213.854 s","203.512 MB","661.512 MB"],["10M","scan + GT","147.100 s","4931.457 MB","5598.250 MB"]],64,160,[100,230,120,190,180],40);
  ctx.addText(slide,{text:"Memory note",x:76,y:424,w:200,h:28,fontSize:17,color:C.ink,bold:true});
  ctx.addText(slide,{text:"Peak memory is current-process working set/private memory sampled during each stage. It is an engineering observation, not a formal complexity proof.",x:76,y:464,w:1050,h:52,fontSize:17,color:C.ink});
  footer(slide,ctx,4); return slide;
}
`);

write("slide-05.mjs", String.raw`
import { C, base, title, footer, bullets } from "./theme.mjs";
export async function slide05(presentation, ctx) {
  const slide = presentation.slides.add(); base(slide, ctx); title(slide, ctx, "How to defend the result", "CLAIM BOUNDARY");
  ctx.addText(slide,{text:"Safe to say",x:76,y:158,w:240,h:28,fontSize:18,color:C.teal,bold:true});
  bullets(slide,ctx,["The large-scale story now has ground-truth evidence, not only throughput evidence.","The scanner recalled all injected anomalies at 1M and 10M scale on generated orders data.","The 1M run completed controlled repair evaluation and preserved rollback metadata."],90,206,500);
  ctx.addText(slide,{text:"Do not overclaim",x:684,y:158,w:280,h:28,fontSize:18,color:C.red,bold:true});
  bullets(slide,ctx,["Do not claim high overall precision; numeric thresholds produced many false positives.","Do not claim 10M repair accuracy; that run is detection-only.","Do not present numeric side-effect repairs as successes."],698,206,500);
  ctx.addShape(slide,{x:74,y:498,w:1130,h:2,fill:C.line});
  ctx.addText(slide,{text:"Suggested line: This experiment strengthens the evidence chain, while also making numeric threshold tuning a clear limitation.",x:82,y:536,w:1060,h:42,fontSize:17,color:C.ink});
  footer(slide,ctx,5); return slide;
}
`);

if (!process.env.HOME && process.env.USERPROFILE) process.env.HOME = process.env.USERPROFILE;
const builder = findPresentationBuilder();
const result = spawnSync(process.execPath, [
  builder,
  "--slides-dir", slidesDir,
  "--out", out,
  "--preview-dir", previewDir,
  "--layout-dir", layoutDir,
  "--slide-count", "5",
], { cwd: repo, stdio: "inherit", env: process.env });

if (result.status !== 0) process.exit(result.status || 1);
console.log(out);
