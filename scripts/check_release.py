"""Check the portable publication payload without inspecting credentials or env files."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {'.git', '.venv', 'venv', '__pycache__', '.pytest_cache', 'outputs', 'build', 'dist', '.cache'}
MANUSCRIPT_SOURCE_SUFFIXES = {'.tex', '.latex', '.bib', '.cls', '.bst', '.sty', '.ppt', '.pptx', '.doc', '.docx', '.rtf'}


def payload_files(root):
    return sorted(p for p in root.rglob('*') if p.is_file()
                  and not any(part in SKIP_DIRS or part.endswith('.egg-info') for part in p.relative_to(root).parts))


def check(root, links=False):
    patterns = {
        'absolute-private-workspace': re.compile(rb'/(?:data|root|home)/[A-Za-z0-9_./-]+'),
        'nonstandard-service-endpoint': re.compile(rb'https?://(?!localhost(?=[:/])|127\.0\.0\.1(?=[:/]))[^/\s\"\']+:\d{2,5}', re.I),
        'credential-shaped-token': re.compile(rb'\bsk-(?:[a-zA-Z0-9]+-)?[a-zA-Z0-9]{24,}\b'),
    }
    findings, files = [], payload_files(root)
    local_links = 0
    for path in files:
        if path.suffix.lower() in MANUSCRIPT_SOURCE_SUFFIXES:
            findings.append({'file':str(path.relative_to(root)), 'check':'editable-manuscript-in-public-payload'})
        data = path.read_bytes()
        for name, pattern in patterns.items():
            if pattern.search(data):
                findings.append({'file':str(path.relative_to(root)), 'check':name})
        if path.name.startswith('.env'):
            findings.append({'file':str(path.relative_to(root)), 'check':'environment-file-in-payload'})
        if not links or path.suffix not in {'.md', '.html'}:
            continue
        text = data.decode('utf-8')
        candidates = (re.findall(r'!?\[[^\]]*\]\(([^\s\)]+)(?:\s+"[^"]*")?\)', text)
                      if path.suffix == '.md' else re.findall(r'(?:href|src)\s*=\s*[\'\"]([^\'\"]+)[\'\"]', text))
        for link in candidates:
            if link.startswith(('https:', 'http:', 'mailto:', 'tel:', 'data:', 'javascript:', '#')):
                continue
            value = unquote(link.split('#',1)[0].split('?',1)[0])
            if not value or value.startswith('/'):
                continue
            target = path.parent/value
            local_links += 1
            if not target.exists():
                findings.append({'file':str(path.relative_to(root)), 'check':'missing-local-link', 'target':value})
    return {'status':'PASS' if not findings else 'FAIL', 'files_checked':len(files),
            'local_links_checked':local_links, 'findings':findings,
            'credentials_or_environment_inspected':False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--links', action='store_true', help='Also check local Markdown/HTML asset links')
    args = parser.parse_args()
    result = check(args.root.resolve(), args.links)
    print(json.dumps(result, indent=2))
    return 0 if result['status']=='PASS' else 1


if __name__=='__main__':
    raise SystemExit(main())
