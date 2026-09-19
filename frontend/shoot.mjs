// Screenshot the compare view once BOTH answers have finished streaming.
// Edge's headless --screenshot cannot wait for a three-minute fetch; this can.
//
//   node shoot.mjs "<url>" "<out.png>"

import { chromium } from "playwright";

const url = process.argv[2];
const out = process.argv[3];

const browser = await chromium.launch({ channel: "msedge" });
const page = await browser.newPage({ viewport: { width: 1500, height: 1200 } });

page.on("console", function (message) {
  if (message.type() === "error") {
    console.log("page error:", message.text());
  }
});

console.log("loading", url);
await page.goto(url, { waitUntil: "load" });

// Each side prints a "cited: …" line when its answer is done.
await page.waitForFunction(
  function () {
    const notes = Array.from(document.querySelectorAll("p.note"));
    const done = notes.filter(function (node) {
      return node.textContent.trim().startsWith("cited:");
    });
    return done.length >= 2;
  },
  null,
  { timeout: 900000, polling: 1000 }
);

console.log("both answers finished");
await page.screenshot({ path: out, fullPage: true });
console.log("wrote", out);
await browser.close();
