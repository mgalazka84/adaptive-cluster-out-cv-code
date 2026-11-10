"""Training-only folds shared by the external and restart diagnostics."""
from pathlib import Path
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'code'))
sys.path.insert(0, str(ROOT / 'experiments/orchard'))
from run_lexdec import COLUMNS, load_data
from orchard_data import read_data
from acv import CompressedFold


def cases(dataset):
    if dataset == 'lexdec':
        data, design, y = load_data()
        designs = {f'M{m + 1}': design[:, cols] for m, cols in enumerate(COLUMNS)}
        subjects, words = data.Subject.to_numpy(), data.Word.to_numpy()
        targets = [('new_subject', subjects, subjects, words),
                   ('new_word', words, words, subjects)]
    elif dataset == 'orchard':
        y, row, col, designs = read_data(ROOT / 'experiments/orchard/OrchardSprays.csv')
        targets = [('row', row, row, col), ('column', col, row, col)]
    else:
        raise ValueError(dataset)
    for target, block, a, b in targets:
        groups = np.unique(block)
        for m, (name, x) in enumerate(designs.items()):
            for k, group in enumerate(groups):
                hold = block == group
                inputs = (x[~hold], y[~hold], a[~hold], b[~hold],
                          (x[hold], y[hold], a[hold], b[hold]))
                key = dict(dataset=dataset, target=target, model=name, model_index=m,
                           fold=k, group_label=str(group))
                yield key, inputs, k in (0, len(groups) // 2, len(groups) - 1)


def strict_starts(fold):
    scale = max(float(np.var(fold.y)), 1.e-5)
    return {k: scale * np.array(v) for k, v in {
        'moment': [.5, .25, .25], 'first_large': [.1, .8, .1],
        'second_large': [.1, .1, .8], 'both_zero': [1., 0., 0.]}.items()}


def branch_match(theta, reference, cert):
    if not cert['valid']:
        return False
    free = cert['free']
    same = np.array_equal(np.flatnonzero(reference == 0), cert['active'])
    distance = np.linalg.norm(np.linalg.solve(cert['S'][free], (reference-theta)[free]))
    return bool(same and distance <= cert['outer_radius'])
