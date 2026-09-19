import { useEffect, useState } from "react";

import { getCorpora, getHealth } from "./api.js";
import AskView from "./views/AskView.jsx";
import TreeView from "./views/TreeView.jsx";
import CompareView from "./views/CompareView.jsx";
import EvalView from "./views/EvalView.jsx";
import UploadView from "./views/UploadView.jsx";

const TABS = [
  { key: "upload", label: "Upload a PDF" },
  { key: "compare", label: "Compare" },
  { key: "ask", label: "Ask" },
  { key: "tree", label: "Tree" },
  { key: "eval", label: "Evaluation" },
];

// The URL hash names the view, and may carry a question:
//   #tree        #compare        #ask?q=what%20is%20slip
function readHash() {
  const raw = window.location.hash.replace(/^#/, "");
  const [name, query] = raw.split("?");
  const params = new URLSearchParams(query || "");
  const known = TABS.map(function (item) {
    return item.key;
  });
  return {
    tab: known.includes(name) ? name : "compare",
    question: params.get("q") || "",
    fast: params.get("fast") === "1",
  };
}

export default function App() {
  const initial = readHash();
  const [tab, setTab] = useState(initial.tab);
  const [initialQuestion] = useState(initial.question);
  const [initialFast] = useState(initial.fast);
  const [health, setHealth] = useState(null);
  const [healthError, setHealthError] = useState("");
  const [corpora, setCorpora] = useState([]);
  const [corpusId, setCorpusId] = useState("demo");
  // The nodes the last question retrieved, so the Tree view can light them up.
  const [highlightIds, setHighlightIds] = useState([]);

  function refreshCorpora() {
    getCorpora()
      .then(function (payload) {
        setCorpora(payload.corpora);
      })
      .catch(function () {
        setCorpora([]);
      });
  }

  useEffect(function () {
    function onHashChange() {
      setTab(readHash().tab);
    }
    window.addEventListener("hashchange", onHashChange);
    return function () {
      window.removeEventListener("hashchange", onHashChange);
    };
  }, []);

  useEffect(function () {
    getHealth()
      .then(function (payload) {
        setHealth(payload);
        setCorpora(payload.corpora || []);
      })
      .catch(function (error) {
        setHealthError(error.message);
      });
  }, []);

  function goTo(key) {
    setTab(key);
    window.location.hash = key;
  }

  function onCorpusReady(newCorpusId) {
    refreshCorpora();
    setCorpusId(newCorpusId);
    setHighlightIds([]);
    goTo("compare");
  }

  const llm = health ? health.llm : null;
  let current = null;
  for (const corpus of corpora) {
    if (corpus.corpus_id === corpusId) {
      current = corpus;
    }
  }

  return (
    <div className="shell">
      <header className="topbar">
        <div className="brand">
          <h1>Syllabus Tree Tutor</h1>
          <span className="paper">
            RAPTOR ·{" "}
            <a href="https://arxiv.org/abs/2401.18059" target="_blank" rel="noreferrer">
              arXiv 2401.18059
            </a>
          </span>
        </div>

        <div className="health">
          {llm && (
            <span>
              <span className={"dot " + (llm.ready ? "ok" : "warn")} />
              llm: {llm.provider}
            </span>
          )}
          {corpora.length > 0 && (
            <select
              className="corpus-select"
              value={corpusId}
              onChange={function (event) {
                setCorpusId(event.target.value);
                setHighlightIds([]);
              }}
            >
              {corpora.map(function (corpus) {
                return (
                  <option key={corpus.corpus_id} value={corpus.corpus_id}>
                    {corpus.name}
                    {corpus.stats ? " (" + corpus.stats.nodes + " nodes)" : ""}
                  </option>
                );
              })}
            </select>
          )}
          {healthError !== "" && <span className="error">backend offline</span>}
        </div>

        <nav className="tabs">
          {TABS.map(function (item) {
            return (
              <button
                key={item.key}
                data-active={tab === item.key}
                onClick={function () {
                  goTo(item.key);
                }}
              >
                {item.label}
              </button>
            );
          })}
        </nav>
      </header>

      <main>
        {healthError !== "" && (
          <div className="panel">
            <p className="panel-title">Backend not reachable</p>
            <p className="note">
              Start it with <code>uvicorn app.api:app --reload --port 8000</code> from the{" "}
              <code>backend/</code> folder, then reload this page.
            </p>
          </div>
        )}

        {tab === "upload" && <UploadView onReady={onCorpusReady} llm={llm} />}
        {tab === "compare" && (
          <CompareView
            corpusId={corpusId}
            corpusName={current ? current.name : ""}
            llm={llm}
            initialQuestion={initialQuestion}
            initialFast={initialFast}
          />
        )}
        {tab === "ask" && (
          <AskView
            onRetrieved={setHighlightIds}
            llm={llm}
            initialQuestion={initialQuestion}
            corpusId={corpusId}
          />
        )}
        {tab === "tree" && <TreeView highlightIds={highlightIds} corpusId={corpusId} />}
        {tab === "eval" && <EvalView />}
      </main>
    </div>
  );
}
