// The "behind the answer" strip: what was searched and what came back.

export default function TracePanel({ trace, provider }) {
  if (!trace) {
    return null;
  }

  const levels = Object.keys(trace.levels_used)
    .sort()
    .map(function (level) {
      return "L" + level + "×" + trace.levels_used[level];
    })
    .join("  ");

  const cells = [
    { k: "mode", v: trace.mode },
    { k: "nodes searched", v: trace.searched_nodes + " / " + trace.total_nodes },
    { k: "levels used", v: levels === "" ? "—" : levels },
    { k: "summaries", v: trace.summary_hits },
    { k: "leaves", v: trace.leaf_hits },
    { k: "top cosine", v: trace.top_score },
    { k: "duplicates skipped", v: trace.duplicates_skipped },
    { k: "hybrid bm25", v: trace.hybrid ? "on" : "off" },
  ];

  if (provider) {
    cells.push({ k: "answer by", v: provider });
  }

  return (
    <div className="trace">
      {cells.map(function (cell) {
        return (
          <div key={cell.k}>
            <div className="k">{cell.k}</div>
            <div className="v">{String(cell.v)}</div>
          </div>
        );
      })}
    </div>
  );
}
