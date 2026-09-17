import { useEffect, useState } from "react";

import { getNode, getTree } from "../api.js";
import TreeMap from "../components/TreeMap.jsx";

export default function TreeView({ highlightIds }) {
  const [nodes, setNodes] = useState([]);
  const [stats, setStats] = useState(null);
  const [selectedId, setSelectedId] = useState("");
  const [detail, setDetail] = useState(null);
  const [error, setError] = useState("");

  useEffect(function () {
    getTree()
      .then(function (payload) {
        setNodes(payload.nodes);
        setStats(payload.stats);
      })
      .catch(function (problem) {
        setError(problem.message);
      });
  }, []);

  function select(nodeId) {
    setSelectedId(nodeId);
    setDetail(null);
    getNode(nodeId)
      .then(setDetail)
      .catch(function (problem) {
        setError(problem.message);
      });
  }

  return (
    <div>
      <div className="panel">
        <p className="panel-title">The tree</p>
        <p className="note">
          Every chunk of your PDFs is a tick on the bottom row. Each row above is a level of
          LLM-written summaries, clustered by meaning, ending in a single root that covers
          everything. Click any node to read it.
          {highlightIds.length > 0 && " Nodes from your last question are highlighted."}
        </p>

        <div className="legend">
          <span>
            <span className="swatch" style={{ background: "#2f5c55", border: "1px solid #4fb6a8" }} />
            chunk (original text)
          </span>
          <span>
            <span className="swatch" style={{ background: "#5c4a22", border: "1px solid #e0a33e" }} />
            summary node
          </span>
          <span>
            <span className="swatch" style={{ background: "#7c9cf5" }} />
            retrieved for the last question
          </span>
          {stats && (
            <span style={{ marginLeft: "auto", fontFamily: "var(--mono)" }}>
              {JSON.stringify(stats.nodes_per_level)}
            </span>
          )}
        </div>

        {error !== "" && <p className="error">{error}</p>}
        <TreeMap
          nodes={nodes}
          highlightIds={highlightIds}
          selectedId={selectedId}
          onSelect={select}
        />
      </div>

      {selectedId !== "" && (
        <div className="panel node-detail">
          <p className="panel-title">{selectedId}</p>
          {detail === null && <p className="note">loading…</p>}
          {detail && (
            <div>
              <div className="source-head">
                <span className={"badge " + detail.node.kind}>
                  {detail.node.kind === "summary" ? "summary L" + detail.node.level : "original text"}
                </span>
                <span>{detail.node.source}</span>
                <span>p. {detail.node.pages.slice(0, 10).join(", ")}</span>
                <span>· covers {detail.node.member_count} chunk(s)</span>
              </div>

              <div className="body">{detail.node.text}</div>

              {detail.children.length > 0 && (
                <div>
                  <p className="panel-title" style={{ marginTop: 16 }}>
                    built from {detail.children.length} nodes
                  </p>
                  <div className="child-list">
                    {detail.children.map(function (child) {
                      return (
                        <div
                          key={child.id}
                          className="child"
                          onClick={function () {
                            select(child.id);
                          }}
                          style={{ cursor: "pointer" }}
                        >
                          <div className="id">
                            {child.id} · p. {child.pages.slice(0, 4).join(", ")}
                          </div>
                          {child.text.slice(0, 150)}…
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
