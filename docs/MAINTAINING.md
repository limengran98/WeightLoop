# Updating the public project

## Manuscript

Keep the editable manuscript in the authors' private source directory. Export
the finished, named author manuscript as PDF, then synchronize the two public
download locations. The author/arXiv manuscript and anonymous four-page submission have the same scientific content and remain separate PDF files:

```bash
python scripts/update_public_assets.py --source /path/to/WeightLoop.pdf --submission-source /path/to/anonymous.pdf
python scripts/update_public_assets.py --check
```

The canonical repository copy is `assets/WeightLoop.pdf`; the website serves
`homepage/assets/WeightLoop.pdf`. The public repository contains no editable
manuscript files. Keep the named public manuscript separate from the anonymous
submission files. Update `metadata/authors.json`,
the manuscript author block, `CITATION.cff`, README and homepage together when
author information changes.

## Project page and README

Edit `homepage/index.html`, `homepage/styles.css` and `homepage/script.js`.
The website is static and has no package installation or build step:

```bash
python -m http.server 8000 --directory homepage
```

Open `http://localhost:8000`. Update `README.md` and `README.zh-CN.md` for the
repository presentation. Website results should remain traceable to the source
tables and use the same statistical definitions as the paper.

## GitHub Pages

The repository includes an official GitHub Pages deployment workflow. In the
repository **Settings → Pages**, choose **GitHub Actions** as the source. Once
enabled, pushes to `main` update the site. Its configured project URL is
`https://limengran98.github.io/WeightLoop/`.

If the repository is private, its plan must support Pages; make the repository
public when ready for a public code release. SSH Git push access does not itself
grant the REST administration permission needed to change that setting.

## Release checks

```bash
python -m unittest discover -s tests -v
python scripts/reproduce_tables.py
python scripts/check_release.py --links
python scripts/update_public_assets.py --check
```

Keep every row needed for each displayed comparison and update the paper, README and homepage from the
same result version. Do not add API keys, service credentials, raw datasets or
third-party model weights to Git.
