// Renders an answer, turning every [n] marker into a button that jumps to
// the matching source card.

export default function AnswerText({ text, onCiteClick, streaming }) {
  const pieces = [];
  const pattern = /\[(\d+)\]/g;
  let lastIndex = 0;
  let match = pattern.exec(text);
  let key = 0;

  while (match !== null) {
    if (match.index > lastIndex) {
      pieces.push(<span key={key}>{text.slice(lastIndex, match.index)}</span>);
      key = key + 1;
    }
    const rank = Number(match[1]);
    pieces.push(
      <button
        key={key}
        className="cite"
        title={"Jump to source " + rank}
        onClick={function () {
          if (onCiteClick) {
            onCiteClick(rank);
          }
        }}
      >
        {rank}
      </button>
    );
    key = key + 1;
    lastIndex = match.index + match[0].length;
    match = pattern.exec(text);
  }

  if (lastIndex < text.length) {
    pieces.push(<span key={key}>{text.slice(lastIndex)}</span>);
  }

  return (
    <div className="answer">
      {pieces}
      {streaming && <span className="cursor-blink">&nbsp;</span>}
    </div>
  );
}
