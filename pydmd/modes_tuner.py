from functools import partial
import numpy as np

class ModesSelectors:
    """
    A container class which defines some static methods for pre-packed
    modes selectors functions to be used in `select_modes`.

    `functools.partial` is used to provide both parametrization of the
    functions and immediate usability. For instance, to select the first
    x modes by integral contributions one would call:

    # TODO: check
    >>> from pydmd import DMDBase
    >>> dmd.select_modes(DMDBase.ModesSelectors.integral_contribution(x))
    """

    @staticmethod
    def _stable_modes(dmd, max_distance_from_unity, bidirectional):
        """
        Complete function of the modes selector `stable_modes`.

        :param float max_distance_from_unity: the maximum distance from the
            unit circle.
        :return np.ndarray: an array of bool, where each "True" index means
            that the corresponding DMD mode is selected.
        """
        arr = np.abs(dmd.eigs) - 1
        if bidirectional:
            arr = np.abs(arr)
        return arr < max_distance_from_unity

    @staticmethod
    def stable_modes(max_distance_from_unity, bidirectional=False):
        """
        Select all the modes such that the magnitude of the corresponding
        eigenvalue is in `(1-max_distance_from_unity,1+max_distance_from_unity)`,
        non inclusive.

        :param float max_distance_from_unity: the maximum distance from the
            unit circle.
        :return callable: function which can be used as the parameter
            of `select_modes` to select DMD modes according to
            the criteria of stability.
        """
        return partial(ModesSelectors._stable_modes,
            max_distance_from_unity=max_distance_from_unity,
            bidirectional=bidirectional)

    @staticmethod
    def _amplitude_threshold(dmd, threshold):
        print('Amplitude threshold: {}'.format(threshold))
        print('Bool amp cut array: {}'.format(np.abs(dmd.amplitudes) > threshold))
        return np.abs(dmd.amplitudes) > threshold

    @staticmethod
    def amplitude_threshold(threshold):
        return partial(ModesSelectors._amplitude_threshold, threshold=threshold)

    @staticmethod
    def _compute_integral_contribution(mode, dynamic):
        """
        Compute the integral contribution across time of the given DMD mode,
        given the mode and its dynamic, as shown in
        http://dx.doi.org/10.1016/j.euromechflu.2016.11.015

        :param numpy.ndarray mode: the DMD mode.
        :param numpy.ndarray dynamic: the dynamic of the given DMD mode, as
            returned by `dmd.dynamics[mode_index]`.
        :return float: the integral contribution of the given DMD mode.
        """
        return pow(np.linalg.norm(mode),2) * sum(np.abs(dynamic))

    @staticmethod
    def _integral_contribution(dmd, n):
        """
        Complete function of the modes selector `integral_contribution`.

        :param int n: the number of DMD modes to be selected.
        :return np.ndarray: an array of bool, where each "True" index means
            that the corresponding DMD mode is selected.
        """

        # temporary reset dmd_time to original_time
        temp = dmd.dmd_time
        dmd.dmd_time = dmd.original_time

        dynamics = dmd.dynamics
        modes = dmd.modes

        # reset dmd_time
        dmd.dmd_time = temp

        n_of_modes = modes.shape[1]
        integral_contributions = [ModesSelectors._compute_integral_contribution(*tp)
            for tp in zip(modes.T, dynamics)]

        indexes_first_n = np.array(integral_contributions).argsort()[-n:]

        truefalse_array = np.array([False for _ in range(n_of_modes)])
        truefalse_array[indexes_first_n] = True
        return truefalse_array

    @staticmethod
    def integral_contribution(n):
        """
        Select the
        Reference: http://dx.doi.org/10.1016/j.euromechflu.2016.11.015

        :param int n: the number of DMD modes to be selected.
        :return callable: function which can be used as the parameter
            of `select_modes` to select DMD modes according to
            the criteria of integral contribution.
        """
        return partial(ModesSelectors._integral_contribution, n=n)

def select_modes(dmd, func, recompute_amplitudes=False):
    """
    Select the DMD modes by using the given `func`.
    `func` has to be a callable function which takes as input the DMD
    object itself and return a numpy.ndarray of boolean where `False`
    indicates that the corresponding mode will be discarded.
    The class :class:`ModesSelectors` contains some pre-packed selector
    functions.

    :param callable func: the function to select the modes

    Example:

    >>> def stable_modes(dmd):
    >>>    toll = 1e-3
    >>>    return np.abs(np.abs(dmd.eigs) - 1) < toll
    >>> dmd = DMD(svd_rank=10)
    >>> dmd.fit(sample_data)
    >>> dmd.select_modes(stable_modes)
    """
    selected_indexes = func(dmd)

    dmd.operator._eigenvalues = dmd.operator._eigenvalues[selected_indexes]
    dmd.operator._Lambda = dmd.operator._Lambda[selected_indexes]

    dmd.operator._eigenvectors = dmd.operator._eigenvectors[:, selected_indexes]
    dmd.operator._modes = dmd.operator._modes[:, selected_indexes]

    dmd.operator._Atilde = np.linalg.multi_dot([
        dmd.operator._eigenvectors,
        np.diag(dmd.operator._eigenvalues),
        np.linalg.pinv(dmd.operator._eigenvectors)])

    if recompute_amplitudes:
        dmd._b = dmd._compute_amplitudes()
    else:
        dmd._b = dmd._b[selected_indexes]

    return np.where(np.logical_not(selected_indexes))[0]


def _compute_stabilized_quantities(eigs, amplitudes):
    factors = np.abs(eigs)

    eigs /= factors
    amplitudes *= eigs

    return (eigs, amplitudes)


def stabilize_modes(dmd, max_distance_from_unity, min_distance_from_unity=1.e-16, cut_above=False, bidirectional=False):
    fixable_eigs_indexes = [eig_distance.item() > min_distance_from_unity and eig_distance.item() < max_distance_from_unity
        for eig_distance in (
            np.abs(np.abs(dmd.eigs) - 1) if bidirectional
            else np.abs(dmd.eigs) - 1
    )]

    eigs, amps = _compute_stabilized_quantities(dmd.eigs[fixable_eigs_indexes],
        dmd.amplitudes[fixable_eigs_indexes])

    dmd.operator._eigenvalues[fixable_eigs_indexes] = eigs
    dmd._b[fixable_eigs_indexes] = amps

    stabilized_indexes = np.where(fixable_eigs_indexes)[0]

    if cut_above:
        cut_indexes = select_modes(dmd, ModesSelectors.stable_modes(max_distance_from_unity, bidirectional), recompute_amplitudes=True)
    else:
        cut_indexes = np.array([])

    return (stabilized_indexes, cut_indexes)
