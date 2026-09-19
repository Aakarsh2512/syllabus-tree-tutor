import { useEffect, useRef, useState } from "react";

import { getJob, uploadCorpus } from "../api.js";

// Drop a PDF, watch its tree get built, then go and ask it questions.

export default function UploadView({ onReady, llm, limits }) {
  const noModel = !llm || llm.provider === "offline";
  const [files, setFiles] = useState([]);
  const [name, setName] = useState("");
  const [fast, setFast] = useState(true);
  const [dragging, setDragging] = useState(false);
  const [job, setJob] = useState(null);
  const [corpusId, setCorpusId] = useState("");
  const [error, setError] = useState("");
  const pollTimer = useRef(null);

  useEffect(function () {
    return function () {
      if (pollTimer.current) {
        window.clearTimeout(pollTimer.current);
      }
    };
  }, []);

  function chooseFiles(list) {
    const picked = [];
    for (const file of list) {
      if (file.name.toLowerCase().endsWith(".pdf")) {
        picked.push(file);
      }
    }
    if (picked.length === 0) {
      setError("Please choose PDF files.");
      return;
    }
    setError("");
    setFiles(picked);
    if (name === "") {
      setName(picked[0].name.replace(/\.pdf$/i, ""));
    }
  }

  function poll(jobId) {
    getJob(jobId)
      .then(function (current) {
        setJob(current);
        if (current.state === "queued" || current.state === "running") {
          pollTimer.current = window.setTimeout(function () {
            poll(jobId);
          }, 1200);
        }
      })
      .catch(function (problem) {
        setError(problem.message);
      });
  }

  async function start() {
    if (files.length === 0) {
      setError("Choose a PDF first.");
      return;
    }
    setError("");
    setJob({ state: "queued", messages: ["uploading…"], seconds: 0 });
    try {
      const started = await uploadCorpus(files, name, fast);
      setCorpusId(started.corpus_id);
      poll(started.job_id);
    } catch (problem) {
      setError(problem.message);
      setJob(null);
    }
  }

  const building = job !== null && (job.state === "queued" || job.state === "running");
  const done = job !== null && job.state === "done";

  return (
    <div>
      <div className="panel">
        <p className="panel-title">Try it with your own PDF</p>
        <p className="note">
          Upload anything with selectable text — lecture notes, a paper, a manual. The
          chunks get clustered and summarised into a tree, then you can ask the same
          question of the tree and of ordinary chunk search, side by side.
        </p>

        <div
          className="dropzone"
          data-dragging={dragging}
          onDragOver={function (event) {
            event.preventDefault();
            setDragging(true);
          }}
          onDragLeave={function () {
            setDragging(false);
          }}
          onDrop={function (event) {
            event.preventDefault();
            setDragging(false);
            chooseFiles(event.dataTransfer.files);
          }}
        >
          <input
            id="pdf-input"
            type="file"
            accept="application/pdf"
            multiple
            style={{ display: "none" }}
            onChange={function (event) {
              chooseFiles(event.target.files);
            }}
          />
          <p className="drop-big">Drop PDFs here</p>
          <p className="note">
            or{" "}
            <button
              className="linklike"
              onClick={function () {
                document.getElementById("pdf-input").click();
              }}
            >
              choose files
            </button>
          </p>
          {files.length > 0 && (
            <p className="note">
              {files.length} file{files.length > 1 ? "s" : ""}:{" "}
              {files.map(function (file) {
                return file.name;
              }).join(", ")}
            </p>
          )}
        </div>

        <div className="controls">
          <input
            id="corpus-name"
            type="text"
            className="name-input"
            placeholder="Name this document set"
            value={name}
            onChange={function (event) {
              setName(event.target.value);
            }}
          />

          <div className="seg">
            <button
              data-active={fast === true}
              onClick={function () {
                setFast(true);
              }}
            >
              Fast summaries
            </button>
            <button
              data-active={fast === false}
              disabled={noModel}
              title={noModel ? "No language model is configured on this server" : ""}
              onClick={function () {
                setFast(false);
              }}
            >
              Model summaries
            </button>
          </div>

          <button className="primary" onClick={start} disabled={building}>
            {building ? "building…" : "Build the tree"}
          </button>
        </div>

        <p className="note">
          {fast
            ? "Fast: extractive summaries, about 20 seconds for a 20-page PDF."
            : "Model: written summaries via " +
              (llm ? llm.provider : "the configured provider") +
              ". Better, but minutes per summary on a CPU."}
        </p>

        {limits && (
          <p className="note">
            Up to {limits.max_upload_files} PDFs, {limits.max_upload_mb} MB in total. Uploads are
            not private: anyone using this server can select them, and they are cleared when
            it restarts.
          </p>
        )}

        {error !== "" && <p className="error">{error}</p>}
      </div>

      {job !== null && (
        <div className="panel">
          <p className="panel-title">
            Build {job.state} {job.seconds ? "· " + job.seconds + "s" : ""}
          </p>
          <ol className="buildlog">
            {job.messages.map(function (message, index) {
              return <li key={index}>{message}</li>;
            })}
          </ol>

          {building && (
            <div className="bar indeterminate">
              <span />
            </div>
          )}

          {job.state === "failed" && <p className="error">{job.error}</p>}

          {done && (
            <div className="controls">
              <button
                className="primary"
                onClick={function () {
                  onReady(corpusId, name);
                }}
              >
                Ask this document a question
              </button>
              <span className="note">
                {job.stats.nodes} nodes · {job.stats.nodes_per_level["0"]} chunks ·{" "}
                {job.stats.levels} levels
              </span>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
