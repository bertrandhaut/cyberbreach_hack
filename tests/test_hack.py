import pytest
from pytest import approx

import numpy as np
from pathlib import Path

from cyberhack.hack import gain, analyze_file


@pytest.fixture
def M():
    return np.array([['bd', '55', 'e9', 'bd', 'e9'],
                     ['55', '55', 'bd', '55', '55'],
                     ['1c', '55', '1c', 'bd', 'e9'],
                     ['1c', '55', 'bd', 'e9', '55'],
                     ['bd', '55', '55', 'e9', 'e9']]).T


@pytest.fixture
def T():
    return np.array([['55', '1c', 'bd'],
                     ['55', 'e9', None],
                     ['bd', '55', 'e9']]).T


def test_gain(M, T):
    assert gain([1, 2, 2, 1, 4, 4], M, T) == 6


@pytest.mark.parametrize('filename, expected_sol, expected_gain',
                         [
                             ('ref.png', 'C1, R2, C4, R4', 5),
                             ('1.png', 'C2, R2', 3),
                             ('2.png', 'C1, R2, C3', 2),
                          ])
def test_analyze_file(filename, expected_sol, expected_gain):
    full_filename = Path(__file__).parent / 'data' / filename
    x_opt_str, g = analyze_file(filename=full_filename)
    assert x_opt_str == expected_sol
    assert g == approx(expected_gain, abs=0.1)
