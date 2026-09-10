"""Compare regenerated Phase5A artifacts byte-for-byte to primary outputs."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('regenerated_directory', type=Path)
    args = parser.parse_args()
    checked = {}
    for role in ['tables', 'figures']:
        original = Path('results') / role
        regenerated = args.regenerated_directory / role
        sources = sorted(original.glob('phase5a_*'))
        assert {p.name for p in sources} == {p.name for p in regenerated.glob('phase5a_*')}
        for path in sources:
            digest = sha(path)
            assert digest == sha(regenerated / path.name), path
            checked[str(path)] = digest
    result = dict(status='PASS', regenerated_directory=str(args.regenerated_directory),
                  byte_identical_artifacts=len(checked), sha256=checked,
                  post_run_audit_scripts={str(p): sha(p) for p in [Path(__file__),
                    Path('scripts/verify_phase5a_outputs.py'), Path('scripts/summarize_phase5a_environment.py')]})
    with Path('runs/phase5a_highdim_external_validity_v1/analysis_regeneration.json').open('x') as stream:
        json.dump(result, stream, indent=2)
    print('PASS:', len(checked), 'tables/figures identical')


if __name__ == '__main__':
    main()
