/*
 * Visionneuse de page documentaire partagée — PDF à gauche, texte extrait à droite.
 *
 * Mutualise ce qui était dupliqué entre la modale « Recherche » des espaces
 * (space_detail.html) et la modale « Monitoring des chunks » de la bibliothèque
 * (library.html) : chargement paresseux de pdf.js, rendu d'une page en canvas,
 * surlignage des occurrences, et rendu markdown minimal.
 *
 * Volontairement SANS dépendance externe (ni marked, ni DOMPurify) : la
 * bibliothèque ne les charge pas, et le rendu markdown ici est délibérément
 * réduit — on échappe d'abord, puis on ne retransforme que les constructions
 * markdown reconnues. Aucune balise issue d'un document ne peut donc s'exécuter.
 *
 * Chaque appelant garde son orchestration (source des données, libellés, badges)
 * et délègue la mécanique à ce module.
 */
(function (global) {
  "use strict";

  const PDFJS_VERSION = "4.0.379";
  let pdfjsLibPromise = null;

  /* ---------------------------------------------------------------- pdf.js */

  /** Charge pdf.js une seule fois (import dynamique mémoïsé). */
  function loadPdfJs() {
    if (pdfjsLibPromise) return pdfjsLibPromise;
    pdfjsLibPromise = new Promise((resolve, reject) => {
      const script = document.createElement("script");
      script.type = "module";
      script.textContent = `
        import * as pdfjs from 'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/${PDFJS_VERSION}/pdf.min.mjs';
        pdfjs.GlobalWorkerOptions.workerSrc = 'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/${PDFJS_VERSION}/pdf.worker.min.mjs';
        window.__sharedPdfjsLib = pdfjs;
        window.dispatchEvent(new Event('shared-pdfjs-ready'));
      `;
      window.addEventListener(
        "shared-pdfjs-ready",
        () => resolve(global.__sharedPdfjsLib),
        { once: true },
      );
      script.onerror = () => reject(new Error("pdf.js indisponible"));
      document.head.appendChild(script);
      setTimeout(() => {
        if (!global.__sharedPdfjsLib) reject(new Error("pdf.js : délai dépassé"));
      }, 20000);
    });
    return pdfjsLibPromise;
  }

  /** Ouvre le PDF source d'un document de la bibliothèque. */
  async function openDocumentPdf(documentId) {
    const pdfjs = await loadPdfJs();
    const response = await fetch(`/api/library/documents/${documentId}/file`, {
      credentials: "include",
    });
    if (!response.ok) throw new Error("PDF indisponible");
    const data = await response.arrayBuffer();
    return pdfjs.getDocument({ data }).promise;
  }

  /** Rend une page du PDF dans un conteneur, à la largeur disponible. */
  async function renderPdfPage(container, pdfDoc, pageNum) {
    if (!container || !pdfDoc) return;
    const safePage = Math.min(Math.max(1, pageNum || 1), pdfDoc.numPages);
    const page = await pdfDoc.getPage(safePage);
    const width = Math.max(container.clientWidth - 24, 280);
    const base = page.getViewport({ scale: 1 });
    const viewport = page.getViewport({ scale: width / base.width });

    container.innerHTML = "";
    const wrapper = document.createElement("div");
    wrapper.className = "relative mx-auto";
    wrapper.style.width = "fit-content";
    const canvas = document.createElement("canvas");
    canvas.width = viewport.width;
    canvas.height = viewport.height;
    canvas.className = "rounded-lg";
    canvas.style.boxShadow = "0 2px 10px rgba(0,0,0,0.2)";
    wrapper.appendChild(canvas);
    container.appendChild(wrapper);
    await page.render({ canvasContext: canvas.getContext("2d"), viewport }).promise;
  }

  /* ------------------------------------------------------------ surlignage */

  function escapeRegex(value) {
    return String(value).replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  }

  /**
   * Surligne les occurrences d'un ou plusieurs termes dans un conteneur déjà
   * rendu, sans jamais réinjecter de HTML (manipulation de nœuds texte).
   */
  function highlightTerms(container, term) {
    if (!container || !term) return;
    const needles = String(term)
      .split(/\s+/)
      .map((t) => t.trim())
      .filter((t) => t.length >= 2);
    if (!needles.length) return;

    const pattern = needles
      .slice()
      .sort((a, b) => b.length - a.length)
      .map(escapeRegex)
      .join("|");
    const re = new RegExp(pattern, "gi");

    const walker = document.createTreeWalker(container, NodeFilter.SHOW_TEXT, {
      acceptNode(node) {
        if (!node.nodeValue) return NodeFilter.FILTER_REJECT;
        re.lastIndex = 0;
        if (!re.test(node.nodeValue)) return NodeFilter.FILTER_REJECT;
        if (node.parentElement && node.parentElement.closest("mark")) {
          return NodeFilter.FILTER_REJECT;
        }
        return NodeFilter.FILTER_ACCEPT;
      },
    });
    const targets = [];
    while (walker.nextNode()) targets.push(walker.currentNode);

    targets.forEach((node) => {
      const text = node.nodeValue;
      const frag = document.createDocumentFragment();
      let idx = 0;
      let match;
      re.lastIndex = 0;
      while ((match = re.exec(text)) !== null) {
        if (match.index > idx) {
          frag.appendChild(document.createTextNode(text.slice(idx, match.index)));
        }
        const mark = document.createElement("mark");
        mark.className =
          "bg-yellow-300 dark:bg-yellow-500/60 text-gray-900 rounded px-0.5";
        mark.textContent = match[0];
        frag.appendChild(mark);
        idx = match.index + match[0].length;
        if (match[0].length === 0) re.lastIndex++;
      }
      if (idx < text.length) {
        frag.appendChild(document.createTextNode(text.slice(idx)));
      }
      node.parentNode.replaceChild(frag, node);
    });
  }

  /* --------------------------------------------------------- rendu markdown */

  function escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text ?? "";
    return div.innerHTML;
  }

  function renderInline(escaped) {
    return escaped
      .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
      .replace(/(^|[\s(])\*([^*\n]+)\*(?=[\s.,;:)]|$)/g, "$1<em>$2</em>")
      .replace(/(^|[\s(])_([^_\n]+)_(?=[\s.,;:)]|$)/g, "$1<em>$2</em>")
      .replace(
        /`([^`\n]+)`/g,
        '<code class="px-1 py-0.5 rounded bg-gray-200 dark:bg-gray-700 text-[0.85em]">$1</code>',
      );
  }

  const isTableRow = (line) => /^\s*\|.*\|\s*$/.test(line);
  const isTableSeparator = (line) => /^\s*\|?[\s\-:|]+\|[\s\-:|]*$/.test(line);

  function renderTable(lines) {
    const rows = lines
      .filter((l) => !isTableSeparator(l))
      .map((l) =>
        l.trim().replace(/^\|/, "").replace(/\|$/, "").split("|").map((c) => c.trim()),
      );
    if (!rows.length) return "";
    const [head, ...body] = rows;
    const th = head
      .map(
        (c) =>
          `<th class="px-2 py-1 text-left font-semibold border border-gray-300 dark:border-gray-600">${renderInline(escapeHtml(c))}</th>`,
      )
      .join("");
    const trs = body
      .map(
        (r) =>
          `<tr>${r.map((c) => `<td class="px-2 py-1 align-top border border-gray-300 dark:border-gray-600">${renderInline(escapeHtml(c))}</td>`).join("")}</tr>`,
      )
      .join("");
    return `<div class="overflow-x-auto my-2"><table class="text-xs border-collapse"><thead class="bg-gray-100 dark:bg-gray-800"><tr>${th}</tr></thead><tbody>${trs}</tbody></table></div>`;
  }

  /**
   * Rendu markdown minimal et sûr : titres (tous niveaux), tableaux, listes,
   * étapes numérotées, gras/italique/code. Le contenu est échappé AVANT toute
   * transformation.
   */
  function renderMarkdown(raw) {
    const lines = String(raw || "").split("\n");
    const out = [];
    let paragraph = [];
    let list = [];

    const flushParagraph = () => {
      if (!paragraph.length) return;
      out.push(`<p class="my-1.5">${renderInline(escapeHtml(paragraph.join(" ")))}</p>`);
      paragraph = [];
    };
    const flushList = () => {
      if (!list.length) return;
      out.push(
        `<ul class="my-1.5 ml-4 list-disc space-y-0.5">${list
          .map((li) => `<li>${renderInline(escapeHtml(li))}</li>`)
          .join("")}</ul>`,
      );
      list = [];
    };
    const flushAll = () => {
      flushParagraph();
      flushList();
    };

    for (let i = 0; i < lines.length; i++) {
      const line = lines[i];
      if (!line.trim()) {
        flushAll();
        continue;
      }

      if (isTableRow(line)) {
        flushAll();
        const block = [];
        while (i < lines.length && isTableRow(lines[i])) block.push(lines[i++]);
        i--;
        out.push(renderTable(block));
        continue;
      }

      const heading = line.match(/^(#{1,6})\s+(.+)$/);
      if (heading) {
        flushAll();
        // Les niveaux de titre des PDF sont arbitraires (un DTA titre ses sections
        // en ######) : on les ramène à deux tailles lisibles.
        const size = heading[1].length <= 2 ? "text-base" : "text-sm";
        const text = heading[2].replace(/^\*\*(.+)\*\*$/, "$1");
        out.push(
          `<h4 class="${size} font-semibold mt-3 mb-1 text-gray-900 dark:text-gray-50">${renderInline(escapeHtml(text))}</h4>`,
        );
        continue;
      }

      const bullet = line.match(/^\s*[-*+]\s+(.+)$/);
      if (bullet) {
        flushParagraph();
        list.push(bullet[1]);
        continue;
      }

      const numbered = line.match(/^\s*(\d{1,2})[.)]\s+(.+)$/);
      if (numbered) {
        flushAll();
        out.push(
          `<p class="my-1.5"><span class="inline-flex items-center justify-center w-5 h-5 mr-1.5 rounded bg-indigo-100 dark:bg-indigo-900/40 text-indigo-700 dark:text-indigo-300 text-xs font-semibold">${escapeHtml(numbered[1])}</span>${renderInline(escapeHtml(numbered[2]))}</p>`,
        );
        continue;
      }

      flushList();
      paragraph.push(line.trim());
    }
    flushAll();
    return out.join("");
  }

  global.PageViewer = {
    loadPdfJs,
    openDocumentPdf,
    renderPdfPage,
    highlightTerms,
    renderMarkdown,
    escapeHtml,
  };
})(window);
