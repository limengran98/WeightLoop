# Project page

This is a buildless, standalone project page. Serve this directory as the web root.
It uses system fonts and local assets, with no CDN, analytics or external runtime dependencies.

## Updating content

- `index.html`: project title, author order/affiliations, text, reported means, citation and commands.
- `styles.css`: layout, responsive rules, colors and motion preferences.
- `script.js`: illustrative feedback-cycle state, pause control and clipboard buttons.
- `assets/WeightLoop.pdf`: public signed paper; synchronize with the repository's canonical public PDF.
- `assets/figure1.png`: browser-safe framework preview; the paper retains the vector figure. The original LibreOffice presentation SVG relies on script-driven slide visibility and is not used as an HTML image.
- `assets/figure2.png`: integrated discovery, matched-Adam and native-update preview.
- `assets/favicon.svg`: original small project mark.

Keep the public PDF and site results synchronized when the paper changes. The three training seeds are model-training seeds, not independent LLM constructions. The animated feedback cycle is a workflow illustration; it does not animate benchmark measurements.

## Browser checks

- Load at desktop and 375px mobile widths; no horizontal document overflow.
- `#animation-toggle` pauses/resumes the highlighted feedback phase and connector motion.
- Reduced-motion preference starts animation paused.
- `[data-copy="quickstart"]` copies four commands; `[data-copy="bibtex"]` copies the citation.
- Paper and figure links must return real files. Both scientific figure images are lazy-loaded.
- Every author, affiliation and correspondence line is plain HTML and remains visible without JavaScript.

For a local preview: `python -m http.server 8000 --directory homepage` from the repository root.
