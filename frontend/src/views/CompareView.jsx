import { useState } from "react";

import { compareModes } from "../api.js";
import AnswerText from "../components/AnswerText.jsx";
import TracePanel from "../components/TracePanel.jsx";

const EXAMPLES = [
  "What topics does the EE2200 quiz cover overall?",
  "Summarise what the Aspen problem sets involve.",
  "Across both courses, what kinds of calculations am I expected to do?",
  "What is the full-load slip of the induction motor?",
];

function Side({ title, note, result }) {
  return (
    <div className="panel">
      <div className="col-head">
        <h3>{title}</h3>
        <span className="pill">{note}</span>
      </div>
      <AnswerText text={result.answer} />
      <div style={{ marginTop: 12 }}>
        <TracePanel trace={result.trace} />
      </div>
      <p className="note" style={{ marginTop: 10 }}>
        retrieved: {result.trace.retrieved_ids.join(", ")}
      </p>
    </div>
  );
}

export default function CompareView() {
  const [question, setQuestion] = useState("");
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");

  async function run(text) {
    const asked = (text === undefined ? question : text).trim();
    if (asked === "" || busy) {
      return;
    }
    setQuestion(asked);
    setBusy(true);
    setError("");
    setResult(null);
    try {
      setResult(await compareModes({ question: asked }));
    } catch (problem) {
      setError(problem.message);
    }
    setBusy(false);
  }

  return (
    <div>
      <div className="panel">
        <p className="panel-title">Same question, both retrievers</p>
        <form
          className="asker"
          onSubmit={function (event) {
            event.preventDefault();
            run();
          }}
        >
          <input
            id="compare-question"
            type="text"
            placeholder="Ask something broad, then something specific…"
            value={question}
            onChange={function (event) {
              setQuestion(event.target.value);
            }}
          />
          <button className="primary" type="submit" disabled={busy}>
            {busy ? "running…" : "Compare"}
          </button>
        </form>
        <div className="examples">
          {EXAMPLES.map(function (example) {
            return (
              <button
                key={example}
                onClick={function () {
                  run(example);
                }}
              >
                {example}
              </button>
            );
          })}
        </div>
        <p className="note" style={{ marginTop: 12 }}>
          Broad questions are where the tree should help: a summary node already spans many
          chunks. On narrow factual questions the two usually agree.
        </p>
      </div>

      {error !== "" && (
        <div className="panel">
          <p className="error">{error}</p>
        </div>
      )}

      {result && (
        <div>
          <div className="two-col">
            <Side
              title="Flat baseline"
              note="chunks only"
              result={result.flat}
            />
            <Side
              title="RAPTOR tree"
              note="chunks + summaries"
              result={result.tree}
            />
          </div>
          <div className="panel">
            <p className="note">
              {result.shared_count} of {result.tree.trace.top_k} retrieved nodes were the same in
              both runs
              {result.shared_count > 0 && ": " + result.shared_nodes.join(", ")}. The tree run
              used {result.tree.trace.summary_hits} summary node(s).
            </p>
          </div>
        </div>
      )}
    </div>
  );
}
