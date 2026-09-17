import { useEffect, useRef, useState } from "react";

import { askStream } from "../api.js";
import AnswerText from "../components/AnswerText.jsx";
import SourceCard from "../components/SourceCard.jsx";
import TracePanel from "../components/TracePanel.jsx";

const EXAMPLES = [
  "What topics does the EE2200 quiz cover overall?",
  "How is armature copper loss calculated?",
  "What does Shift+F5 do in Aspen Plus?",
  "Which property methods suit polar mixtures with azeotropes?",
  "What should I revise about induction motor slip?",
];

export default function AskView({ onRetrieved, llm, initialQuestion }) {
  const [question, setQuestion] = useState(initialQuestion || "");
  const [mode, setMode] = useState("tree");
  const [hybrid, setHybrid] = useState(false);
  const [busy, setBusy] = useState(false);
  const [answer, setAnswer] = useState("");
  const [hits, setHits] = useState([]);
  const [trace, setTrace] = useState(null);
  const [provider, setProvider] = useState("");
  const [citations, setCitations] = useState(null);
  const [error, setError] = useState("");
  const [flashRank, setFlashRank] = useState(0);

  async function ask(text) {
    const asked = (text === undefined ? question : text).trim();
    if (asked === "" || busy) {
      return;
    }
    setQuestion(asked);
    setBusy(true);
    setError("");
    setAnswer("");
    setHits([]);
    setTrace(null);
    setCitations(null);

    try {
      await askStream(
        { question: asked, mode: mode, hybrid: hybrid },
        {
          onMeta: function (payload) {
            setHits(payload.hits);
            setTrace(payload.trace);
            setProvider(payload.provider);
            if (onRetrieved) {
              onRetrieved(payload.trace.retrieved_ids);
            }
          },
          onText: function (piece) {
            setAnswer(function (current) {
              return current + piece;
            });
          },
          onDone: function (payload) {
            setCitations(payload.citations);
          },
        }
      );
    } catch (problem) {
      setError(problem.message);
    }
    setBusy(false);
  }

  // A question carried in the URL is asked once, on first render.
  const autoAsked = useRef(false);
  useEffect(function () {
    if (!autoAsked.current && initialQuestion && initialQuestion !== "") {
      autoAsked.current = true;
      ask(initialQuestion);
    }
  }, [initialQuestion]);

  function jumpToSource(rank) {
    const element = document.getElementById("source-" + rank);
    if (element) {
      element.scrollIntoView({ behavior: "smooth", block: "center" });
      setFlashRank(rank);
    }
  }

  return (
    <div>
      <div className="panel">
        <form
          className="asker"
          onSubmit={function (event) {
            event.preventDefault();
            ask();
          }}
        >
          <input
            id="question"
            type="text"
            placeholder="Ask anything about your course material…"
            value={question}
            onChange={function (event) {
              setQuestion(event.target.value);
            }}
          />
          <button className="primary" type="submit" disabled={busy}>
            {busy ? "thinking…" : "Ask"}
          </button>
        </form>

        <div className="controls">
          <div className="seg">
            <button
              data-active={mode === "tree"}
              onClick={function () {
                setMode("tree");
              }}
            >
              RAPTOR tree
            </button>
            <button
              data-active={mode === "flat"}
              onClick={function () {
                setMode("flat");
              }}
            >
              Flat baseline
            </button>
          </div>

          <label className="check">
            <input
              type="checkbox"
              checked={hybrid}
              onChange={function (event) {
                setHybrid(event.target.checked);
              }}
            />
            hybrid BM25
          </label>

          <span className="note">
            {mode === "tree"
              ? "searches chunks and summaries together"
              : "searches only the original chunks"}
          </span>
        </div>

        <div className="examples">
          {EXAMPLES.map(function (example) {
            return (
              <button
                key={example}
                onClick={function () {
                  ask(example);
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

      {(answer !== "" || busy) && (
        <div className="panel">
          <p className="panel-title">Answer</p>
          <AnswerText text={answer} onCiteClick={jumpToSource} streaming={busy} />
          {citations && (
            <p className="note" style={{ marginTop: 10 }}>
              cited passages: {citations.cited.length > 0 ? citations.cited.join(", ") : "none"}
              {citations.all_valid ? " · all citations valid" : " · invalid: " + citations.invalid.join(", ")}
              {llm && llm.provider === "offline" && " · extractive mode, no language model"}
            </p>
          )}
        </div>
      )}

      {trace && (
        <div className="panel">
          <p className="panel-title">Behind the answer</p>
          <TracePanel trace={trace} provider={provider} />
        </div>
      )}

      {hits.length > 0 && (
        <div className="panel">
          <p className="panel-title">Retrieved passages</p>
          {hits.map(function (hit) {
            return <SourceCard key={hit.id} hit={hit} flash={hit.rank === flashRank} />;
          })}
        </div>
      )}
    </div>
  );
}
