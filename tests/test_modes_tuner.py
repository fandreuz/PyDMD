from builtins import range
from unittest import TestCase
from pydmd.dmd import DMD
import matplotlib.pyplot as plt
import numpy as np
import os
from pydmd.modes_tuner import stabilize_modes

class FakeDMD:
    pass
class FakeOperator:
    pass

class TestModesTuner(TestCase):
    def test_stabilize_oneeig(self):
        dmd = FakeDMD()
        dmd.amplitudes = np.array([1], dtype=np.complex128)
        dmd.eigs = np.array([0.02891154], dtype=np.complex128)
        dmd._b = dmd.amplitudes

        dmd.operator = FakeOperator()
        dmd.operator._eigenvalues = dmd.eigs

        stab, cut = stabilize_modes(dmd, 1.e-3, bidirectional=True)
        assert len(stab) == 0
        assert len(cut) == 0

        stab, cut = stabilize_modes(dmd, 1.e-3, bidirectional=False)
        assert len(stab) == 0
        assert len(cut) == 0
