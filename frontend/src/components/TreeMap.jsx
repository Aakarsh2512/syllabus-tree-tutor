import { useMemo } from "react";

// Draws the whole RAPTOR tree as one SVG: the root at the top, summary
// levels below it, and every leaf chunk as a tick along the bottom.
// Nodes retrieved by the last question are drawn in the accent colour.

const MARGIN_X = 46;
const ROW_HEIGHT = 96;
const TOP_PADDING = 34;

export default function TreeMap({ nodes, highlightIds, selectedId, onSelect }) {
  const layout = useMemo(
    function () {
      if (nodes.length === 0) {
        return { positions: {}, width: 1100, height: 300, levels: [] };
      }

      const byLevel = {};
      let maxLevel = 0;
      for (const node of nodes) {
        if (!byLevel[node.level]) {
          byLevel[node.level] = [];
        }
        byLevel[node.level].push(node);
        if (node.level > maxLevel) {
          maxLevel = node.level;
        }
      }

      const leafCount = byLevel[0] ? byLevel[0].length : 1;
      const width = Math.max(1100, leafCount * 7 + MARGIN_X * 2);
      const height = TOP_PADDING + (maxLevel + 1) * ROW_HEIGHT;

      const positions = {};
      const levels = [];
      for (let level = maxLevel; level >= 0; level = level - 1) {
        const row = byLevel[level] || [];
        // Level maxLevel sits on the first row, level 0 on the last.
        const y = TOP_PADDING + (maxLevel - level) * ROW_HEIGHT;
        const span = width - MARGIN_X * 2;
        for (let index = 0; index < row.length; index = index + 1) {
          const x = MARGIN_X + ((index + 0.5) * span) / row.length;
          positions[row[index].id] = { x: x, y: y, level: level };
        }
        levels.push({ level: level, y: y, count: row.length });
      }

      return { positions: positions, width: width, height: height, levels: levels };
    },
    [nodes]
  );

  const highlighted = new Set(highlightIds);

  const edges = [];
  for (const node of nodes) {
    if (node.children.length === 0) {
      continue;
    }
    const from = layout.positions[node.id];
    if (!from) {
      continue;
    }
    for (const childId of node.children) {
      const to = layout.positions[childId];
      if (!to) {
        continue;
      }
      const lit = highlighted.has(node.id) || highlighted.has(childId);
      edges.push(
        <path
          key={node.id + "-" + childId}
          d={
            "M " + from.x + " " + (from.y + 13) +
            " C " + from.x + " " + (from.y + 50) + ", " +
            to.x + " " + (to.y - 40) + ", " +
            to.x + " " + (to.y - 9)
          }
          fill="none"
          stroke={lit ? "#7c9cf5" : "#272d3a"}
          strokeWidth={lit ? 1.4 : 0.6}
          opacity={lit ? 0.9 : 0.5}
        />
      );
    }
  }

  return (
    <div className="tree-wrap">
      <svg
        viewBox={"0 0 " + layout.width + " " + layout.height}
        width={layout.width}
        height={layout.height}
        style={{ maxWidth: "100%", height: "auto", display: "block" }}
        preserveAspectRatio="xMidYMid meet"
        role="img"
        aria-label="RAPTOR tree"
      >
        {layout.levels.map(function (row) {
          return (
            <text
              key={"label-" + row.level}
              x={8}
              y={row.y + 4}
              fill="#6f7a90"
              fontSize="10"
              fontFamily="Consolas, monospace"
            >
              {row.level === 0 ? "chunks" : "L" + row.level}
            </text>
          );
        })}

        {edges}

        {nodes.map(function (node) {
          const position = layout.positions[node.id];
          if (!position) {
            return null;
          }
          const isLeaf = node.kind === "leaf";
          const lit = highlighted.has(node.id);
          const isSelected = node.id === selectedId;

          let fill = isLeaf ? "#2f5c55" : "#5c4a22";
          if (lit) {
            fill = "#7c9cf5";
          }
          const stroke = isSelected ? "#e6e8ee" : lit ? "#a9bdf8" : isLeaf ? "#4fb6a8" : "#e0a33e";

          const width = isLeaf ? 5 : Math.max(34, Math.min(120, node.member_count * 2.2));
          const height = isLeaf ? 17 : 22;

          return (
            <g
              key={node.id}
              transform={"translate(" + (position.x - width / 2) + "," + (position.y - height / 2) + ")"}
              onClick={function () {
                onSelect(node.id);
              }}
              style={{ cursor: "pointer" }}
            >
              <title>
                {node.id + " · " + node.kind + " · " + node.source +
                  " · covers " + node.member_count + " chunk(s)"}
              </title>
              <rect
                width={width}
                height={height}
                rx={isLeaf ? 1.5 : 4}
                fill={fill}
                stroke={stroke}
                strokeWidth={isSelected ? 1.8 : 0.9}
              />
              {!isLeaf && width >= 44 && (
                <text
                  x={width / 2}
                  y={height / 2 + 4}
                  textAnchor="middle"
                  fontSize="10"
                  fontFamily="Consolas, monospace"
                  fill={lit ? "#0d1220" : "#e6e8ee"}
                  pointerEvents="none"
                >
                  {node.id}
                </text>
              )}
            </g>
          );
        })}
      </svg>
    </div>
  );
}
