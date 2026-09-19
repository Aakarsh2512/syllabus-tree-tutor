import { useEffect, useRef, useState } from "react";

import { compareStream } from "../api.js";
import AnswerText from "../components/AnswerText.jsx";

// The point of the project: one question, both retrievers, side by side.
// Retrieval takes milliseconds, so the passages show at once; the two answers
// stream in afterwards.

const EXAMPLES = [
  "What does this document cover overall?",
  "Summarise the main sections.",
  "What are the key numbers or results?",
  "What should I revise first?",
];

const EMPTY_SIDE = { hits: [], trace: null, answer: "", citations: null, streaming: false };

function Passage({ hit }) {
  const isSummary = hit.kind === "summary";
  return (
    <li className="passage">
      <span className="rank">{hit.rank}</span>
      <span className={"badge " + (isSummary ? "summary" : "leaf")}>
        {isSummary ? "summary of " + hit.member_count : "chunk"}
      </span>
      <span className="passage-where">
        {hit.source} p.{hit.pages.slice(0, 3).join(",")}
      </span>
      <span className="score">{hit.score.toFixed(3)}</span>
      <span className="passage-text" title={hit.text}>
        {hit.text.slice(0, 150)}…
      </span>
    </li>
  );
}

function Side({ title, note, side, highlight }) {
  return (
    <div className="panel" data-winner={highlight}>
      <div className="col-head">
        <h3>{title}</h3>
        <span className="pill">{note}</span>
        {side.trace && (
          <span className="pill">
            {side.trace.summary_hits} summaries · {side.trace.leaf_hits} chunks
          </span>
        )}
      </div>

      {side.hits.length > 0 && (
        <ul className="passages">
          {side.hits.map(function (hit) {
            return <Passage key={hit.id} hit={hit} />;
          })}
        </ul>
      )}

      {(side.answer !== "" || side.streaming) && (
        <div className="answer-box">
          <AnswerText text={side.answer} streaming={side.streaming} />
        </div>
      )}

      {side.citations && (
        <p className="note">
          cited: {side.citations.cited.length > 0 ? side.citations.cited.join(", ") : "none"}
          {side.citations.all_valid ? " · all valid" : " · invalid: " + side.citations.invalid.join(", ")}
        </p>
      )}
    </div>
  );
}

export default function CompareView({ corpusId, corpusName, llm, initialQuestion, initialFast }) {
  const [question, setQuestion] = useState(initialQuestion || "");
  const [topK, setTopK] = useState(6);
  const [fastAnswers, setFastAnswers] = useState(initialFast === true);
  const [busy, setBusy] = useState(false);
  const [flat, setFlat] = useState(EMPTY_SIDE);
  const [tree, setTree] = useState(EMPTY_SIDE);
  const [shared, setShared] = useState(null);
  const [error, setError] = useState("");

  function setSide(sideName, updater) {
    const setter = sideName === "flat" ? setFlat : setTree;
    setter(updater);
  }

  async function run(text, fastOverride) {
    const useFast = fastOverride === undefined ? fastAnswers : fastOverride;
    const asked = (text === undefined ? question : text).trim();
    if (asked === "" || busy) {
      return;
    }
    setQuestion(asked);
    setBusy(true);
    setError("");
    setShared(null);
    setFlat({ ...EMPTY_SIDE });
    setTree({ ...EMPTY_SIDE });

    try {
      await compareStream(
        {
          question: asked,
          corpus_id: corpusId,
          top_k: topK,
          fast_answers: useFast,
        },
        {
          onSideMeta: function (sideName, payload) {
            setSide(sideName, function (current) {
              return { ...current, hits: payload.hits, trace: payload.trace };
            });
          },
          onRetrievalDone: function (payload) {
            setShared(payload);
            setFlat(function (current) {
              return { ...current, streaming: true };
            });
            setTree(function (current) {
              return { ...current, streaming: true };
            });
          },
          onSideText: function (sideName, piece) {
            setSide(sideName, function (current) {
              return { ...current, answer: current.answer + piece };
            });
          },
          onSideDone: function (sideName, payload) {
            setSide(sideName, function (current) {
              return { ...current, citations: payload.citations, streaming: false };
            });
          },
        }
      );
    } catch (problem) {
      setError(problem.message);
    }
    setFlat(function (current) {
      return { ...current, streaming: false };
    });
    setTree(function (current) {
      return { ...current, streaming: false };
    });
    setBusy(false);
  }

  // A question in the URL runs once, so a link can show the comparison directly.
  const autoRan = useRef(false);
  useEffect(function () {
    if (!autoRan.current && initialQuestion && initialQuestion !== "") {
      autoRan.current = true;
      run(initialQuestion, initialFast === true);
    }
  }, [initialQuestion]);

  const slow = !fastAnswers && llm && llm.provider === "ollama";

  return (
    <div>
      <div className="panel">
        <p className="panel-title">
          Same question, both retrievers{corpusName ? " · " + corpusName : ""}
        </p>
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

        <div className="controls">
          <label className="check" htmlFor="topk">
            passages each&nbsp;
            <input
              id="topk"
              type="range"
              min="2"
              max="10"
              value={topK}
              onChange={function (event) {
                setTopK(Number(event.target.value));
              }}
            />
            <strong>{topK}</strong>
          </label>

          {llm && llm.provider !== "offline" && (
          <label className="check">
            <input
              type="checkbox"
              checked={fastAnswers}
              onChange={function (event) {
                setFastAnswers(event.target.checked);
              }}
            />
            fast answers (no model)
          </label>
          )}

          <span className="note">
            {slow
              ? "A local model writes both answers, which takes a couple of minutes each on CPU. The passages appear immediately."
              : "Passages appear immediately; answers follow."}
          </span>
        </div>

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
      </div>

      {error !== "" && (
        <div className="panel">
          <p className="error">{error}</p>
        </div>
      )}

      {shared && (
        <div className="panel verdict">
          <p className="note">
            <strong>{shared.shared_count}</strong> of {topK} retrieved passages were the same
            in both. The tree used <strong>{tree.trace ? tree.trace.summary_hits : 0}</strong>{" "}
            summary node{tree.trace && tree.trace.summary_hits === 1 ? "" : "s"}, each standing
            in for many chunks the baseline had to spend slots on.
          </p>
        </div>
      )}

      <div className="two-col">
        <Side title="Flat baseline" note="chunks only" side={flat} highlight={false} />
        <Side title="RAPTOR tree" note="chunks + summaries" side={tree} highlight={true} />
      </div>
    </div>
  );
}
