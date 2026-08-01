import { createRequire } from "module";
import { pathToFileURL } from "url";
import path from "path";

const require = createRequire(import.meta.url);
const { chromium } = require("C:/Users/吕佩哲/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright");

const htmlPath = path.resolve(process.argv[2]);
const pdfPath = path.resolve(process.argv[3]);
const browser = await chromium.launch({
  headless: true,
  executablePath: "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe",
});

try {
  const page = await browser.newPage({ viewport: { width: 1240, height: 1754 } });
  await page.goto(pathToFileURL(htmlPath).href, { waitUntil: "networkidle" });
  await page.emulateMedia({ media: "print" });
  await page.pdf({
    path: pdfPath,
    format: "A4",
    printBackground: true,
    preferCSSPageSize: true,
    displayHeaderFooter: true,
    headerTemplate: `<div style="width:100%;padding:0 18mm;font-family:Arial,'Microsoft YaHei',sans-serif;font-size:7px;color:#64748b;display:flex;justify-content:space-between;"><span>DEEPBLOCK · QUANTUM + OPTIMIZATION</span><span>SEMIFINAL PAPER</span></div>`,
    footerTemplate: `<div style="width:100%;padding:0 18mm;font-family:Arial,'Microsoft YaHei',sans-serif;font-size:7px;color:#64748b;display:flex;justify-content:space-between;"><span>Baihua hardware-in-the-loop CVRP refinement</span><span><span class="pageNumber"></span> / <span class="totalPages"></span></span></div>`,
    margin: { top: "7mm", right: "0", bottom: "8mm", left: "0" },
  });
} finally {
  await browser.close();
}
