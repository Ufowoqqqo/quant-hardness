"""Descriptive host counters around benchmark cells, not hardware attribution."""
import argparse
import csv
import json
from pathlib import Path


def snapshot(path):
    lines = path.read_text().splitlines()
    load = float(lines[lines.index('/proc/loadavg') + 1].split()[0])
    counters = {line.split()[0]: [int(x) for x in line.split()[1:]]
                for line in lines if line.startswith('cpu')}
    memory = {line.split(':')[0]: int(line.split()[1])
              for line in lines if line.startswith(('MemAvailable:', 'SwapFree:'))}
    return load, counters, memory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tables', default='results/tables')
    parser.add_argument('--root', default='runs/phase5a_highdim_external_validity_v1')
    parser.add_argument('--prefix', default='phase5a')
    args = parser.parse_args()
    root = Path(args.root)
    rows = []
    for path in sorted(root.glob('benchmark_*/*_before.txt')):
        after = path.with_name(path.name.replace('_before.txt', '_after.txt'))
        lb, cb, mb = snapshot(path)
        la, ca, ma = snapshot(after)
        stem = path.name.removesuffix('_before.txt')
        meta = json.loads(path.with_name(stem + '.json').read_text())
        delta = [a - b for a, b in zip(ca['cpu'][:8], cb['cpu'][:8])]
        total = sum(delta)
        busy = []
        for cpu in range(2, 2 + meta['workers']):
            diff = [a - b for a, b in zip(ca[f'cpu{cpu}'][:8], cb[f'cpu{cpu}'][:8])]
            busy.append(1 - (diff[3] + diff[4]) / sum(diff))
        rows.append(dict(stage=path.parent.name, cell=stem, workers=meta['workers'],
            load1_before=lb, load1_after=la, host_busy_fraction=1 - (delta[3] + delta[4]) / total,
            host_iowait_fraction=delta[4] / total, worker_busy_min=min(busy), worker_busy_max=max(busy),
            available_memory_min_kib=min(mb['MemAvailable'], ma['MemAvailable']),
            swapfree_change_kib=ma['SwapFree'] - mb['SwapFree']))
    assert len(rows) == 50
    target = Path(args.tables) / (args.prefix+'_environment')
    with target.with_suffix('.json').open('x') as out:
        json.dump(rows, out, indent=2)
    with target.with_suffix('.csv').open('x', newline='') as out:
        writer = csv.DictWriter(out, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    print(json.dumps(dict(cells=len(rows), load1_min=min(min(r['load1_before'], r['load1_after']) for r in rows),
        load1_max=max(max(r['load1_before'], r['load1_after']) for r in rows),
        minimum_available_memory_kib=min(r['available_memory_min_kib'] for r in rows),
        max_host_iowait_fraction=max(r['host_iowait_fraction'] for r in rows),
        swapfree_changes=[r for r in rows if r['swapfree_change_kib']]), indent=2))


if __name__ == '__main__':
    main()
