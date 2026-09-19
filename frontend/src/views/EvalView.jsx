import { useEffect, useState } from "react";

import { getEvalResults } from "../api.js";

const METRICS = [
  { key: "specific_hit_rate", label: "Specific hit@k", hint: "evidence found for a narrow factual question" },
  { key: "specific_mrr", label: "Specific MRR", hint: "how near the top the evidence was" },
  { key: "broad_coverage", label: "Broad coverage", hint: "share of required evidence for a broad question" },
  { key: "refusal_rate", label: "Refusal rate", hint: "declined to answer when the corpus cannot" },
];

export default function EvalView({ demoCorpus }) {
  const [payload, setPayload] = useState(null);
  const [error, setError] = useState("");

  useEffect(function () {
    getEvalResults()
      .then(setPayload)
      .catch(function (problem) {
        setError(problem.message);
      });
  }, []);

  if (error !== "") {
    return (
      <div className="panel">
        <p className="error">{error}</p>
      </div>
    );
  }

  if (payload === null) {
    return (
      <div className="panel">
        <p className="note">loading…</p>
      </div>
    );
  }

  if (!payload.exists) {
    return (
      <div className="panel">
        <p className="panel-title">No results yet</p>
        <p className="note">
          Run <code>python eval/run_eval.py</code> and reload this page.
        </p>
      </div>
    );
  }

  const results = payload.results;
  const counts = results.runs[0].metrics.counts;
  // On a public deployment the evaluation corpus (private course material)
  // is not the demo that is loaded, so say so rather than imply it is.
  const differentCorpus =
    demoCorpus && demoCorpus.stats && results.index.stats &&
    demoCorpus.stats.nodes !== results.index.stats.nodes;

  return (
    <div>
      {differentCorpus && (
        <div className="panel notice">
          <p className="note">
            These results come from the evaluation corpus described below, which is
            private course material and is <strong>not</strong> the demo loaded on
            this server. They are shown as recorded.
          </p>
        </div>
      )}

      <div className="panel">
        <p className="panel-title">How this was measured</p>
        <p className="note">
          {counts.specific} narrow factual questions, {counts.broad} broad questions and{" "}
          {counts.unanswerable} unanswerable ones, all written by hand against{" "}
          {results.index.documents} documents ({results.index.leaf_chunks} chunks,{" "}
          {results.index.stats.nodes} tree nodes). A question counts as a hit when a retrieved
          node actually contains the evidence quote. Answers were produced by the{" "}
          <code>{results.provider}</code> provider; summaries in the index were written by{" "}
          <code>{results.index.summary_provider}</code>.
        </p>
      </div>

      <div className="panel">
        <p className="panel-title">Results</p>
        <table className="results">
          <thead>
            <tr>
              <th>Run</th>
              {METRICS.map(function (metric) {
                return <th key={metric.key}>{metric.label}</th>;
              })}
              <th>Summaries / query</th>
              <th>s / query</th>
            </tr>
          </thead>
          <tbody>
            {results.runs.map(function (run) {
              return (
                <tr key={run.name}>
                  <td className="name">{run.name}</td>
                  {METRICS.map(function (metric) {
                    const value = run.metrics[metric.key];
                    if (value === null || value === undefined) {
                      return (
                        <td key={metric.key} title="not measured in this run">
                          &mdash;
                        </td>
                      );
                    }
                    return (
                      <td key={metric.key}>
                        {value.toFixed(3)}
                        <div className="bar">
                          <span style={{ width: Math.round(value * 100) + "%" }} />
                        </div>
                      </td>
                    );
                  })}
                  <td>{run.metrics.avg_summary_hits.toFixed(2)}</td>
                  <td>{run.seconds_per_question.toFixed(2)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <div className="panel">
        <p className="panel-title">What each metric means</p>
        <ul className="note">
          {METRICS.map(function (metric) {
            return (
              <li key={metric.key}>
                <strong>{metric.label}</strong> — {metric.hint}
              </li>
            );
          })}
        </ul>
      </div>
    </div>
  );
}
