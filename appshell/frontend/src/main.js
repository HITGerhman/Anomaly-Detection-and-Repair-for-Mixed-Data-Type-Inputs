const statusBox = document.getElementById("status-box");
const resultBox = document.getElementById("result-box");
const form = document.getElementById("train-form");
const runBtn = document.getElementById("run-btn");

function setStatus(msg) {
  statusBox.textContent = msg;
}

function setResult(obj) {
  resultBox.textContent = JSON.stringify(obj, null, 2);
}

async function runTrainTask(payload) {
  // In real Wails runtime, this binding should be provided by Go.
  if (window?.go?.main?.App?.RunTrainTask) {
    return window.go.main.App.RunTrainTask(payload);
  }

  // Static fallback so template can be previewed without Wails runtime.
  await new Promise((resolve) => setTimeout(resolve, 600));
  return {
    id: "mock-task-1",
    status: "succeeded",
    response: {
      task_id: "mock-task-1",
      status: "ok",
      result: {
        note: "Wails binding not detected. This is a mock response.",
        request_payload: payload,
      },
    },
  };
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  const payload = {
    csv_path: document.getElementById("csv-path").value.trim(),
    target_col: document.getElementById("target-col").value.trim(),
    output_dir: document.getElementById("output-dir").value.trim(),
  };

  runBtn.disabled = true;
  setStatus("running");
  setResult({});

  try {
    const resp = await runTrainTask(payload);
    setStatus(resp.status || "done");
    setResult(resp);
  } catch (err) {
    setStatus("failed");
    setResult({ error: String(err) });
  } finally {
    runBtn.disabled = false;
  }
});
