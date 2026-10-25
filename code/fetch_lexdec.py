"""Retrieve the languageR 1.6 lexdec data and preserve provenance.

The analysis uses the bundled CSV and does not require pyreadr or network.
This optional source-retrieval script requires pyreadr==0.5.4.
"""
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen
import pyreadr

ROOT = Path(__file__).resolve().parents[1]/'data'
BASE = 'https://raw.githubusercontent.com/cran/languageR/master/'
URLS = {'lexdec.rda': BASE+'data/lexdec.rda',
        'lexdec.Rd': BASE+'man/lexdec.Rd',
        'languageR_DESCRIPTION': BASE+'DESCRIPTION',
        'GPL-2.txt': 'https://www.gnu.org/licenses/old-licenses/gpl-2.0.txt'}


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    prior = ROOT/'lexdec_provenance.json'
    expected = json.loads(prior.read_text())['source_files'] if prior.exists() else {}
    records = {}
    for name, url in URLS.items():
        raw = urlopen(url, timeout=40).read()
        sha = hashlib.sha256(raw).hexdigest()
        if name in expected and sha != expected[name]['sha256']:
            raise ValueError(f'Source changed: {name}; review before replacing data.')
        (ROOT/name).write_bytes(raw)
        records[name] = {'url': url, 'sha256': sha, 'bytes': len(raw)}
    data = pyreadr.read_r(str(ROOT/'lexdec.rda'))['lexdec']
    data.to_csv(ROOT/'lexdec.csv', index=False, float_format='%.17g')
    provenance = {'dataset': 'lexdec', 'package': 'languageR', 'version': '1.6',
                  'retrieved': '2026-10-25', 'rows': len(data), 'columns': list(data.columns),
                  'subjects': int(data.Subject.nunique()), 'words': int(data.Word.nunique()),
                  'primary_documentation': 'https://cran.r-project.org/web/packages/languageR/languageR.pdf',
                  'distribution': 'CRAN package read-only mirror; authors R. H. Baayen and Elnaz Shafaei-Bajestan',
                  'collection': 'With Jen Hay, University of Canterbury, Christchurch, New Zealand, 2004 (package documentation).',
                  'declared_package_license': 'GPL (>= 2)', 'source_files': records,
                  'csv_sha256': hashlib.sha256((ROOT/'lexdec.csv').read_bytes()).hexdigest(),
                  'conversion': 'pyreadr 0.5.4; CSV with 17 significant digits, no dropped rows.'}
    prior.write_text(json.dumps(provenance, indent=2))
    print(json.dumps({k: provenance[k] for k in ['rows', 'subjects', 'words', 'csv_sha256']}, indent=2))
    print(data[['RT', 'Frequency', 'Length', 'Trial']].describe().to_string())
    print(data[['Correct', 'NativeLanguage']].value_counts().to_string())


if __name__ == '__main__':
    main()
