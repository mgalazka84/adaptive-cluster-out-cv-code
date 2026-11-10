"""Data and fixed-effect designs for the OrchardSprays validation examples."""
import csv

import numpy as np


def read_data(path):
    with path.open(encoding='utf-8', newline='') as stream:
        records = list(csv.DictReader(stream))
    row = np.array([int(r['rowpos']) for r in records])
    col = np.array([int(r['colpos']) for r in records])
    treatment = np.array([r['treatment'] for r in records])
    volume = np.array([float(r['decrease']) for r in records])
    levels = np.array(list('ABCDEFGH'))
    if len(set(zip(row, col))) != 64 or len(records) != 64 or np.any(volume <= 0):
        raise ValueError('Expected 64 distinct positive observations in an 8 by 8 square.')
    for factor in (row, col):
        if not np.array_equal(np.unique(factor), np.arange(1, 9)):
            raise ValueError('Unexpected block labels')
        for group in np.unique(factor):
            if not np.array_equal(np.sort(treatment[factor == group]), levels):
                raise ValueError('Every block must contain one replicate of each treatment.')
    intercept = np.ones(len(row))
    dilution = np.array([list('ABCDEFG').index(t) if t != 'H' else 0 for t in treatment])
    control = (treatment == 'H').astype(float)
    designs = {'Common': intercept[:, None],
               'Dilution': np.column_stack((intercept, dilution, control)),
               'Treatment': (treatment[:, None] == levels[None, :]).astype(float)}
    return np.log(volume), row, col, designs


def write_rows(path, rows):
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
