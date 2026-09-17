import { useEffect, useState } from "react";

import { getHealth } from "./api.js";
import AskView from "./views/AskView.jsx";
import TreeView from "./views/TreeView.jsx";
import CompareView from "./views/CompareView.jsx";
import EvalView from "./views/EvalView.jsx";

const TABS = [
  { key: "ask", label: "Ask" },
  { key: "tree", label: "Tree" },
  { key: "compare", label: "Compare" },
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
    tab: known.includes(name) ? name : "ask",
    question: params.get("q") || "",
  };
}

export default function App() {
  const initial = readHash();
  const [tab, setTab] = useState(initial.tab);
  const [initialQuestion] = useState(initial.question);
  const [health, setHealth] = useState(null);
  const [healthError, setHealthError] = useState("");
  // The nodes the last question retrieved, so the Tree view can light them up.
  const [highlightIds, setHighlightIds] = useState([]);

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
      .then(setHealth)
      .catch(function (error) {
        setHealthError(error.message);
      });
  }, []);

  const stats = health && health.index && health.index.stats ? health.index.stats : null;
  const llm = health ? health.llm : null;

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
          {stats && (
            <span>
              {stats.documents} docs · {stats.nodes} nodes · {stats.levels} levels
            </span>
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
                  setTab(item.key);
                  window.location.hash = item.key;
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

        {tab === "ask" && (
          <AskView
            onRetrieved={setHighlightIds}
            llm={llm}
            initialQuestion={initialQuestion}
          />
        )}
        {tab === "tree" && <TreeView highlightIds={highlightIds} />}
        {tab === "compare" && <CompareView />}
        {tab === "eval" && <EvalView />}
      </main>
    </div>
  );
}
