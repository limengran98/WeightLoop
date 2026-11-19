"""Publish a finished PDF and synchronize the project-page download."""
import argparse
import hashlib
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check',action='store_true',help='Verify copies without writing')
    parser.add_argument('--source',type=Path,help='PDF downloaded from arXiv')
    args=parser.parse_args()
    source=args.source if args.source else ROOT/'assets/WeightLoop.pdf'
    if source.suffix.lower()!='.pdf' or not source.is_file() or not source.read_bytes().startswith(b'%PDF-'):
        parser.error('Provide an existing exported PDF')
    digest=hashlib.sha256(source.read_bytes()).hexdigest()
    for rel in ['assets/WeightLoop.pdf','homepage/assets/WeightLoop.pdf']:
        dest=ROOT/rel
        if args.check:
            assert dest.is_file() and hashlib.sha256(dest.read_bytes()).hexdigest()==digest,rel
        else:
            dest.parent.mkdir(parents=True,exist_ok=True)
            if source.resolve()!=dest.resolve():shutil.copy2(source,dest)
    print('PASS_PUBLIC_PDF_COPIES', digest)

if __name__=='__main__':main()
