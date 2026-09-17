import { useState } from "react";

export default function SourceCard({ hit, flash }) {
  const [open, setOpen] = useState(false);
  const isSummary = hit.kind === "summary";
  const pages = hit.pages.slice(0, 8).join(", ");

  return (
    <div className="source" id={"source-" + hit.rank} data-flash={flash === true}>
      <div className="source-head">
        <span className="rank">{hit.rank}</span>
        <span className={"badge " + (isSummary ? "summary" : "leaf")}>
          {isSummary ? "summary L" + hit.level : "original text"}
        </span>
        <span>{hit.source}</span>
        <span>p. {pages}</span>
        {isSummary && <span>· covers {hit.member_count} chunks</span>}
        <span className="score">cos {hit.score.toFixed(3)}</span>
      </div>

      <div className="source-text" data-open={open}>
        {hit.text}
      </div>

      <button
        className="more"
        onClick={function () {
          setOpen(!open);
        }}
      >
        {open ? "show less" : "show full passage"}
      </button>
    </div>
  );
}
