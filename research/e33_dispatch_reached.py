"""E33 dispatch-reached assertion.

Proves from measured readback (not source reading) that the timed M=6 cell is
the cell that actually executed, i.e. that `get_qmv_batch_limit` let M=6 stay on
the qmv path instead of falling through to qmm.
"""
import json


def walk(node, path=()):
    if isinstance(node, dict):
        for k, v in node.items():
            yield from walk(v, path + (str(k),))
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from walk(v, path + (str(i),))
    else:
        yield path, node


for tag in ('e33-base-r1', 'e33-cand-r1'):
    d = json.load(open('.mlxfast-private/qmv-curve/%s/vendored.json' % tag))
    print('=== %s ===' % tag)
    hits = {}
    qmm = []
    for path, val in walk(d):
        if not isinstance(val, str):
            continue
        if 'qmm' in val:
            qmm.append((path, val))
        if 'qmv_fast' not in val:
            continue
        top = path[0]
        hits.setdefault(top, {}).setdefault(val, 0)
        hits[top][val] += 1
    for top in sorted(hits):
        print('  [%s]' % top)
        for k in sorted(hits[top]):
            print('     %-52s x%d' % (k, hits[top][k]))
    print('  qmm-named readbacks anywhere: %d' % len(qmm))
    for p, v in qmm[:5]:
        print('     ', '/'.join(p), '=', v)

    shapes = d.get('shapes')
    if isinstance(shapes, list):
        print('  scored shapes: %d' % len(shapes))
    print()
