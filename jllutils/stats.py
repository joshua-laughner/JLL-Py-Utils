"""
Functions for common statistical methods
"""
from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
from numpy import ma
import pandas as pd
import random
import re
import scipy.stats as st
from statsmodels import api as sm, tools as smtools
from statsmodels.robust.robust_linear_model import RLM
from statsmodels.robust.norms import TukeyBiweight


class BootstrapError(Exception):
    pass


class ConvergenceError(Exception):
    """
    Error to use if a process fails to converge
    """
    pass


class FittingError(Exception):
    """
    Base class for fitting errors
    """
    pass


class FittingDataError(FittingError):
    """
    Error to use if there's a problem with the data given to the fitting function
    """
    pass


class BootstrapSampler(object):
    """A convenience wrapper around `bootstrap_sample`.

    This class allows you to create an instance that stores the data to sample and a default sample size, which may be
    convenient if doing a large number of samples.
    """
    def __init__(self, data, flat=False, with_replacement=True, default_sample_size=None, sampling_fxn=None,
                 rvalues=1, rfill=np.nan, seed=None):
        """
        Parameters
        ----------
        data : array-like
            The data to sample. Data will always be sampled along the first dimension, so if `data` is N-D,
            the samples will also be N-D.

        flat : bool
            Convenience option; set to `True` to flatten the data to 1-D internally.

        with_replacement : bool
            Controls whether the data sampling will be done with or without replacement. When `True`, a data 
            point may be chosen more than once in a single sample. If `False`, a data point will be chosen 
            exactly once.

        default_sample_size : int
            If given, then calling the `sample` or `run_bootstrap` methods without a sample size specified 
            will use this size. If not given here, then it *must* be specified when calling those methods.

        sampling_fxn : callable
            A function that will be called on the samples when using `run_bootstrap`. The function must accept an 
            array as the sole required input. The return values from the function may be any form, and must be
            specified by the `rvalues` parameter.

        rvalues : int, list, or tuple
            This parameter specifies how many values the sampling function returns and whether they should be stored 
            in an array or a dictionary.

            If `rvalues` is an integer, it must indicate how many values are returned by `sampling_fxn`. In this form,
            `run_bootstrap` will store the results from the sampling as an ngroups-by-rvalues array. That is, the first
            row will be the values resulting from calling `sampling_fxn` on the first sample group, the second row for
            the second sampling group and so on.

            If `rvalues` is a list or tuple, then the return values from `sampling_fxn` will be stored in a dictionary
            and `rvalues` is taken to give dictionary keys to use. E.g. if `rvalues` is `('slope', 'int')` then the
            first outputs from `sampling_fxn` will be stored in the array under the `'slope'` key and the second under
            the `'int'` key.

        rfill
            The fill value to use when initializing the arrays to store the return values from `sampling_fxn`. This 
            implicitly sets the data type of these array. At present there is no way to specify separate types for different 
            keys when `rvalue` implied storing data as a dictionary.
        """
        self._data = data if not flat else data.flatten()
        self._replacement = with_replacement
        self._sample_size = default_sample_size

        if isinstance(rvalues, int):
            if rvalues < 1:
                raise BootstrapError('Number of return values must be >= 1')
        elif not isinstance(rvalues, (list, tuple)):
            raise BootstrapError('rvalues must be an int or list/tuple of dict keys')

        self._rvalues = rvalues
        self._rfill = rfill
        self._sampling_fxn = sampling_fxn
        self._seed = seed

    def sample(self, sample_size=None, sample_number=0):
        """Get a bootstrap sample group.

        Parameters
        ----------
        
        sample_size : int
            How large each sample should be. Doesn't need to be given if the default sample size was
            specified when creating the instance. If a value is given here, it overrides the default sample size.

        Returns
        -------
        array-like
            the bootstrap sample group.
        """
        if sample_size is None:
            if self._sample_size is None:
                raise BootstrapError('Must provide either a default sample size when creating the BootstrapSampler '
                                     'instance or a specific sample size when getting the sample')
            else:
                sample_size = self._sample_size

        seed = None if self._seed is None else self._seed + sample_number
        return bootstrap_sample(self._data, sample_size, with_replacement=self._replacement, seed=seed)

    def run_bootstrap(self, n_groups, sample_size=None):
        """Run a series of bootstrap samplings on the data

        This method calls the sampling function specified when creating this instance on each sample group created by
        the bootstrapping and returns the collection of results as either a 2D array or dictionary of 1D arrays,
        depending on the value of `rvalues` when this instance was created. `sampling_fxn` must have been given
        when the instance was created.

        Parameters
        ----------

        n_groups : int
            How many bootstrapping sample groups to create.

        sample_size : int
            How large each sample should be. Doesn't need to be given if the default sample size was
            specified when creating the instance. If a value is given here, it overrides the default sample size.

        Returns
        -------
        numpy.ndarray or dict
            the results of the bootstrap sampling as a 2-D array (ngroups-by-nreturn_vals) if `rvalues` was an
            integer when the instance was created or a dictionary of 1-D arrays if `rvalues` was a list/tuple.
        """
        if self._sampling_fxn is None:
            raise BootstrapError('Must have a sampling function to use run_bootstrap')

        if isinstance(self._rvalues, (tuple, list)):
            results = {k: np.full([n_groups], self._rfill) for k in self._rvalues}
            as_dict = True
        elif isinstance(self._rvalues, int):  # assume rvalues is an integer
            results = np.full([n_groups, self._rvalues], self._rfill)
            as_dict = False
        else:
            raise TypeError('rvalues must be a tuple, list, or integer')

        for i in range(n_groups):
            sample_data = self.sample(sample_size, sample_number=i)
            these_results = self._sampling_fxn(sample_data)
            if as_dict and isinstance(these_results, dict):
                for k in self._rvalues:
                    results[k][i] = these_results[k]
            elif as_dict:
                for k, val in zip(self._rvalues, these_results):
                    results[k][i] = val
            else:
                results[i] = these_results

        return results


def bootstrap_sample(data, sample_size, with_replacement=True, seed=None):
    """Perform bootstrap sampling on an array of data.

    Selects N points from `data` and returns them. If `data` is multidimensional, the selection happens along the
    first dimension, e.g. if `data` is 2D, then N rows are returned. To select individual points from a
    multidimensional array, flatten it before passing.

    Parameters
    ----------

    data : array-like
        Data to sample. Must support numpy-style indexing.

    sample_size : int or float
        The number of samples to generate for each bootstrap. If given as a value between [0, 1), it
        is interpreted as the fraction of the data set to sample.

    with_replacement : bool
        If `True`, then any single sample may be chosen multiple times. If `False`, a sample
        will be chosen at most once.

    seed : Optional[int]
        Used to set the random number generator that samples the data. If reproducibility
        is needed, provide any positive integer. Passing that same integer later will ensure
        that the same samples are taken.

    Returns
    -------
    array-like
        the sample array.
    """
    if 0 <= sample_size < 1:
        sample_size = int(sample_size * np.shape(data)[0])

    n_data_pts = np.shape(data)[0]
    if n_data_pts < sample_size and not with_replacement:
        raise BootstrapError('Fewer than the requested {} points are available for sampling'.format(sample_size))

    rng = np.random.default_rng(seed=seed)
    chosen_inds = rng.choice(np.arange(n_data_pts), size=sample_size, replace=with_replacement)

    return data[chosen_inds]


class PolyFitModel(object):
    """Common interface to polynomial fitting methods

    This class provides a common interface to fit 2D data with or without errors associated with the x and y values.
    Instantiating the class automatically does the fitting, and the result is stored in the new instance. Instantiation
    requires the x and y values to fit be given, and, if the model chosen requires it, error for the x and/or y values
    as well. The model can be chosen with the `model` keyword. Current options are:

        * 'york' - use :func:`york_linear_fit` (from this module).
        * 'y-resid' or 'y_resid' - do least-squares fitting of a straight line to the data, without weighting.
        * 'y-resid-n' or 'y_resid_n' - do least-squares fitting of a polynomial with degree n to the data, without
          weighting.
        * 'ols' - do ordinary least-squares fitting of the data with :mod:`statsmodels`. Unlike y-resid, this
          includes more information in the `results` property, like p-values.
        * 'ols0' - do ordinary least-squares fitting with :mod:`statsmodels`, with no intercept.
        * any function that takes a minimum of four arguments (`x`, `y`, `xerr`, and `yerr`, where `xerr` and
          `yerr` are the error/uncertainty in `x` and `y`) and returns three values:
            - the polynomial coefficients as a 1D array-like object, starting from the x**0 term
            - the errors in the coefficients, also as a 1D array-like object. If errors are not computed, they
              should be NaNs.
            - any object containing more detail about the fitting result, e.g. a dictionary or custom class

    If your chosen model function has additional keyword arguments, you can pass them as a dict with the
    `model_opts` input.

    Since this class is for polynomial fitting, the result is stored as a list of coefficients, where for:

    .. math::
        p(x) = a_0 + a_1 x + a_2 x^2 + ... + a_n x^n

    the coefficients :math:`a_0`, :math:`a_1`, etc. are stored in the `coeffs` property as a list. Only as many
    coefficients as given by the model are stored, so if the model only returns :math:`a_0` and :math:`a_1`, then
    `coeffs` will only have two elements and model.coeffs[2] will raise an `IndexError`.  If you have an
    application where you want the coefficient for a given power of :math:`x` regardless of whether that power is
    used in the model, the `get_any_coeff` method will do that.

    The :math:`a_0` and :math:`a_1` coefficients are accessible through the special properties `yint` and
    `slope`. These will always return a value, if not used in the model, 0.0 will be returned.

    Once created, instances of this class can be used to predict :math:`y` values for new :math:`x` values.
    This is done with the `predict` method, or by calling the instance itself, e.g.::

        >>> x = np.arange(10)
        >>> y = 2 * x
        >>> sigma = 0.1 * np.ones_like(x)
        >>> fit = PolyFitModel(x, y, sigma, sigma, model='york')

        >>> xprime = np.arange(100,110)
        # Both of the following are equivalent
        >>> yprime_1 = fit.predict(xprime)
        >>> yprime_2 = fit(xprime)
        >>> yprime_1 == yprime_2
        True
    """

    @property
    def coeffs(self):
        """Polynomial coefficients for the fit, starting from the x**0 term.

        Returns
        -------
        numpy.ndarray
            Coefficients, starting with the lowest power term.
        """
        return self._coeffs

    @property
    def coeff_errors(self):
        """Errors in the polynomial coefficients for the fit.

        Returns
        -------
        numpy.ndarray
            Errors in coefficients, starting with the lowest power term.
        """
        return self._coeff_errors

    @property
    def yint(self):
        """The y-intercept of the fit, i.e. the x**0 coefficient

        Returns
        -------
        numpy.float
        """
        return self.get_any_coeff(0)

    @property
    def slope(self):
        """The slope of the fit, i.e. the x**1 coefficient

        Returns
        -------
        numpy.float
        """
        return self.get_any_coeff(1)

    @property
    def results(self):
        """An object containing all the results from the fitting procedure. Will be unique to each fitting method.
        """
        return self._results

    @property
    def data(self):
        """The data used to create the fit

        Returns
        -------
        dict
            dictionary with keys 'x', 'y', 'xerr', 'yerr'
        """
        return self._data

    def __init__(self, x, y, xerr=None, yerr=None, model='york', model_opts=None, nans='error', print_errs=True):
        """
        Parameters
        ----------
        x : numpy.ndarray
            The independent variable

        y : numpy.ndarray
            The dependent variable

        xerr : numpy.ndarray
            Error in `x`. Required for the `york` model, ignored for all others.

        yerr : numpy.ndarray
            Error in `y`. Required for the `york` model, ignored for all others.

        model : str
            Which regression model to use. Options are:

            * 'york' - use :func:`york_linear_fit` (from this module).
            * 'y-resid' or 'y_resid' - do least-squares fitting of a straight line to the data, without weighting.
            * 'y-resid-n' or 'y_resid_n' - do least-squares fitting of a polynomial with degree n to the data, without
              weighting.
            * 'ols' - do ordinary least-squares fitting of the data with :mod:`statsmodels`. Unlike y-resid, this
              includes more information in the `results` property, like p-values.
            * 'ols0' - do ordinary least-squares fitting with :mod:`statsmodels`, with no intercept.
            * any function that takes a minimum of four arguments (`x`, `y`, `xerr`, and `yerr`, where `xerr` and
              `yerr` are the error/uncertainty in `x` and `y`) and returns three values:
                - the polynomial coefficients as a 1D array-like object, starting from the x**0 term
                - the errors in the coefficients, also as a 1D array-like object. If errors are not computed, they
                  should be NaNs.
                - any object containing more detail about the fitting result, e.g. a dictionary or custom class

        model_opts : dict
            Additional keyword arguments to the model function.

        nans : str
            How to treat NaNs in the data. Options are:

            * 'error' - raise an error if there are any NaNs.
            * 'ignore' - do nothing, pass `x`, `y`, `xerr`, `yerr` to the fitting function with NaNs left in place.
            * 'drop' - remove NaNs from `x`, `y`, `xerr`, and `yerr`

        print_errs : bool
            Include errors on the coefficients when representing this as a string.

        Raises
        ------
        FittingDataError
            if NaNs are found in `x`, `y`, `xerr`, or `yerr` and `nans == "error"`.
        """
        self._print_errs = print_errs

        # Coerce input data to floats. Check them at the same time
        x = self._fix_in_type(x, 'x')
        y = self._fix_in_type(y, 'y')
        xerr = self._fix_in_type(xerr, 'xerr', allow_none=True)
        yerr = self._fix_in_type(yerr, 'yerr', allow_none=True)

        # Deal with NaNs/infs
        x, y, xerr, yerr = _handle_nans(nans, x, y, xerr, yerr)

        # Store data
        self._fit_fxn = self._get_fit_fxn(model)
        self._model_opts = model_opts if model_opts is not None else dict()
        self._data = {'x': x, 'y': y, 'xerr': xerr, 'yerr': yerr}

        # Input checking
        if not isinstance(self._model_opts, dict):
            raise TypeError('Model options must be given as a dictionary')

        # Do the fit
        coeffs, coeff_errors, results = self._fit_fxn(x, y, xerr, yerr, **self._model_opts)
        self._coeffs = self._check_coeff_type(coeffs)
        self._coeff_errors = self._check_coeff_type(coeff_errors)
        self._results = results

    def __str__(self):
        """
        String-formatted representation of the fit.

        Returns
        -------
        str
            string of form "y = a_0 + a_1 x + a_2 x**2 + ..."
        """
        s = 'y = '
        power = 0
        for c, e in zip(self._coeffs, self._coeff_errors):
            if power > 0:
                s += ' + '

            if self._print_errs and e is not None and not np.isnan(e):
                s += '({:.3g} +/- {:.3g})'.format(c, e)
            else:
                s += '{:.3g}'.format(c)

            if power == 1:
                s += 'x'
            elif power > 1:
                s += 'x**{}'.format(power)

            power += 1

        return s

    def __call__(self, x):
        """
        Syntactic sugar to alias a call to this instance object to the :func:`PolyFitModel.predict` method
        """
        return self.predict(x)

    @staticmethod
    def _fix_in_type(input, name='input', allow_none=False):
        """
        Correct the type of any data input to this model.

        Any time data is give to this model, it should be run through this method before being stored or used.

        Parameters
        ----------
        input
            The input data

        name : str
            Optional, name of the input to use in error messages.

        Returns
        -------
        numpy.ndarray
            The input coerced to the appropriate type (1D numpy float array), if posible

        Raises
        ------
        TypeError
            If the input is of an invalid type
        """
        if allow_none and input is None:
            return None
        if not isinstance(input, np.ndarray) or input.ndim != 1:
            raise TypeError('{} must be a 1D numpy array'.format(name))
        return input.astype(np.float)

    def _check_coeff_type(self, coeff):
        """Check that arrays of coefficients returned from the fitting functions are the right type

        Parameters
        ----------
        coeff : numpy.ndarray
            The array of coefficients. Must be a 1D :class:`numpy.ndarray` or a value that can be converted
            to one by `numpy.array(coeff)`.

        Returns
        -------
        numpy.ndarray
            The coefficients, converted to a numpy float array if necessary and possible.

        Raises
        ------
        TypeError
            If cannot convert to numpy array or given as an array with ndim != 1.
        """
        if not isinstance(coeff, np.ndarray):
            try:
                coeff = np.array(coeff, dtype=np.float)
            except Exception:
                raise TypeError('Problem with fitting model ({}): could not convert output to numpy array'.format(
                    self._fit_fxn.__name__
                ))

        if coeff.ndim != 1:
            raise TypeError('Problem with fitting model ({}): returned coefficients given as array with ndim != 1'.format(
                self._fit_fxn.__name__
            ))

        return coeff.astype(np.float)

    @classmethod
    def _get_fit_fxn(cls, model):
        """Internal helper function that maps model names to function calls.

        Parameters
        ----------
        model : str or callable
            The model name, or a function.

        Returns
        -------
        callable
            The model fitting function to call. Will accept 4 positional arguments (x, y, xerr, yerr) and possibly
            additional keyword args, though the latter is not guaranteed. Will return two values, coefficients and their
            errors, as iterables.
        """

        # Define wrapper functions here. Each one must accept 4 inputs (x, y, xerr, yerr) and will be passed any
        # additional keywords received as the `model_opts` parameter in __init__. Each must return a vector of
        # polynomial coefficients, their errors, and a results object. That object may be of any type.
        def york_fit(x, y, xerr, yerr, **opts):
            if xerr is None or yerr is None:
                raise ValueError('xerr and yerr are required for the "york" fitting method')
            result = york_linear_fit(x, xerr, y, yerr, **opts)
            poly = np.array([result['yint'], result['slope']])
            poly_err = np.array([result['yint_err'], result['slope_err']])
            return poly, poly_err, {'coef': poly, 'coef_err': poly_err}

        def y_resid(x, y, deg):
            # Must call the convert() method to get the right coefficients - by default, the polynomial is scaled for
            # numerical stability, convert() unscales it
            poly = np.polynomial.polynomial.Polynomial.fit(x, y, deg).convert()
            coeffs = poly.coef  # these are in the proper order (x**0 first)
            coeffs_err = np.full(coeffs.shape, np.nan)
            return coeffs, coeffs_err, {'coef': coeffs, 'coef_err': coeffs_err}

        def ols(x, y, zero_int=False):
            if not zero_int:
                x = smtools.add_constant(x)

            results = sm.OLS(y, x, hasconst=not zero_int).fit()
            coeffs = results.params
            coeffs_err = np.sqrt(np.diag(results.cov_params()))
            if zero_int:
                coeffs = np.concatenate([[0], coeffs])
                coeffs_err = np.concatenate([[0], coeffs_err])
            return coeffs, coeffs_err, results

        def robust(x, y, zero_int=False):
            if not zero_int:
                x = smtools.add_constant(x)

            fit = RLM(y, x, M=TukeyBiweight()).fit()
            if zero_int:
                coeffs = np.array([0, fit.params.item()])
                coeffs_err = np.array([0, fit.cov_params().item()])
            else:
                coeffs = fit.params
                coeffs_err = np.sqrt(np.diag(fit.cov_params()))

            return coeffs, coeffs_err, fit

        model_lower = model.lower()
        if not isinstance(model, str):
            return model
        elif model_lower == 'york':
            return york_fit
        elif re.match(r'y[\-_]resid', model, re.IGNORECASE):
            # Allow the user to specify y-resid or y_resid for the typical linear fit, but also y-resid2, y-resid3, etc.
            # for higher-order fits.
            order = re.search(r'\d+$', model)
            order = 1 if order is None else int(order.group())
            return lambda x, y, xerr, yerr, deg=order: y_resid(x, y, deg)
        elif model_lower == 'ols':
            return lambda x, y, xerr, yerr, zint=False: ols(x, y, zero_int=zint)
        elif model_lower == 'ols0':
            return lambda x, y, xerr, yerr, zint=True: ols(x, y, zero_int=zint)
        elif model_lower == 'robust':
            return lambda x, y, xerr, yerr: robust(x, y, zero_int=False)
        elif model_lower == 'robust0':
            return lambda x, y, xerr, yerr: robust(x, y, zero_int=True)
        else:
            raise TypeError('Unknown model type: "{}"'.format(model))

    def predict(self, x):
        """Calculate the y values predicted by the fit for new x values.

        Parameters
        ----------
        x : numpy.ndarray
            The array of x values

        Returns
        -------
        numpy.ndarray
            Predicted y values, same shape as x.
        """
        x = self._fix_in_type(x, 'x')
        y = np.zeros_like(x)
        for p, c in enumerate(self._coeffs):
            y += c * x**p

        return y

    def get_any_coeff(self, term):
        """Get the coefficient from a term in the polynomial fit.

        Unlike the `coeffs` property, this method allows you to ask for any coefficient, even one not returned by the
        fitting function. If a coefficient isn't given by the fitting function, will return 0.0.

        Parameters
        ----------
        term : int
            The term to get the coefficient for, i.e. `term = 1` will give the coefficient for the x**1
            term in the fit.

        Returns
        -------
        numpy.float
            The coefficient
        """
        if term < 0:
            raise ValueError('power must be >= 0. If you are trying to index the list of coefficients from the end, '
                             'use the coeffs property directly, e.g. fit.coeffs[-1]')
        try:
            return self.coeffs[term]
        except IndexError:
            return 0.0

    def plot_fit(self, ax=None, freeze_xlims=True, label='{fit}', ls='--', color='gray', **style):
        """Plot the fit represented by this model on a set of axes

        Parameters
        ----------
        ax
            The axes to plot on, if not given, the current axes are used.

        freeze_xlims : bool
            If `True`, the xlimits of the plot will be reset to what they were before the fit was plotted.
            This usually also keeps the limits from changing after future plotting calls on these axes,
            which helps make the line stay long enough to stretch from one side of the plot to the other.

        label : str
            The string to use in the plot legend. It will be formatted with the bound :class:`PolyFitModel` 
            as the format keyword `fit` - so `"{fit}"` (the default) is replaced with the string representation
            of this fit.

            .. note::
               The label is *always* formatted using the `format` method, so any literal curly braces will need
               to be doubled (e.g. for Latex, you'd need to pass `"\mathrm{{VSF}}"` instead of `"\mathrm{VSF}"`)

        ls : str
            The line style to use when plotting the fit.

        color 
            A Matplotlib colorspec to use when plotting the fit.

        **style
            Additional keyword arguments to the plot function.

        Returns
        -------
        [pyplot handle]
            The handle to the line plotted
        """
        style['color'] = color
        style['ls'] = ls
        style['label'] = label.format(fit=self)

        if ax is None:
            ax = plt.gca()

        x = np.array(ax.get_xlim())
        y = self.predict(x)
        h = ax.plot(x, y, **style)
        if freeze_xlims:
            ax.set_xlim(x)
        return h


class RunningMean(object):
    """Compute a running mean.

    This is valuable when you have a large number of large arrays to average
    and do not need the individual arrays, so can save memory by doing a running
    average that keeps track of the sum and weights only.
    """
    @property
    def result(self):
        """The (possibly weighted) average accumulated to this point
        """
        return self._sum / self._weights

    def __init__(self, shape, dtype=np.float):
        """
        Parameters
        ----------
        shape : Sequence[int]
            The shape of the array needed to store the average.

        dtype
            What data type to use for the running mean. Default is 64-bit float.
        """
        self._sum = np.zeros(shape, dtype=dtype)
        self._weights = np.zeros(shape, dtype=dtype)

    @classmethod
    def from_first_values(cls, values, weights=1.0):
        """Create a RunningMean instance from the first array of values going into the mean

        Parameters
        ----------

        values : array-like
            The first array of values to go into the running mean. That is, if you query 
            `results` immediately on the result, you will get this array back.

        weights : float or array
            The weight to assign to the first value.

        Returns
        -------
        RunningMean
            The initialized `RunningMean` instance
        """
        inst = cls(values.shape, dtype=values.dtype)
        inst.update(values, weights)
        return inst
        
    def update(self, values, weights=1.0):
        """Add an array of values to the running mean
        
        Parameters
        ----------
        
        values : array-like
            The values to add to the running mean

        weights : float or array like
            The weight(s) to assign to these values. 

        Notes
        -----
            In the current version, only numpy arrays are supported
            for values. You may need to convert Pandas dataframes,
            xarray DataArrays, etc. to numpy arrays first.
        """
        self._sum += values
        self._weights += weights


class RunningStdDev(object):
    """Compute a running mean.

    This is valuable when you have a large number of large arrays to average
    and do not need the individual arrays, so can save memory by doing a running
    standard deviation that does not require the arrays to be concatenated.

    Notes
    -----
        This implements Welford's algorithm. It currently only supports sample standard
        deviation (N-1 in the denominator).
    """
    @property
    def result(self):
        """The standard deviation. If there have been fewer than 2 values added, this will be all NaNs.
        """
        if self._count < 2:
            return np.full(self._m2.shape, np.nan)
        else:
            return np.sqrt(self._m2 / (self._count - 1))

    def __init__(self, shape, dtype=np.float):
        """
        Parameters
        ----------
        shape : Sequence[int]
            The shape of the array needed to store the average.

        dtype
            What data type to use for the running mean. Default is 64-bit float.
        """
        self._count = 0
        self._mean = np.zeros(shape, dtype=dtype)
        self._m2 = np.zeros(shape, dtype=dtype)

    @classmethod
    def from_first_values(cls, values):
        """Create a RunningStdDev instance from the first array of values going into it

        Parameters
        ----------

        values : array-like
            The first array of values to go into the running mean. That is, if you query 
            `results` immediately on the result, you will get this array back.

        Returns
        -------
        RunningStdDev
            The initialized `RunningStdDev` instance
        """
        inst = cls(values.shape, dtype=values.dtype)
        inst.update(values)
        return inst
        
    def update(self, values):
        """Add an array of values to the running standard deviation
        
        Parameters
        ----------
        
        values : array-like
            The values to add to the running standard deviation

        Notes
        -----
            In the current version, only numpy arrays are supported
            for values. You may need to convert Pandas dataframes,
            xarray DataArrays, etc. to numpy arrays first.
        """
        self._count += 1
        delta = values - self._mean
        self._mean += delta / self._count
        delta2 = values - self._mean
        self._m2 += delta * delta2


class RunningMeanAndStd(object):
    """Simultaneously compute a running mean and standard deviation
    """
    @property
    def result(self):
        """Gives the current mean and standard deviation.
        """
        return self._mean.result, self._std.result

    def __init__(self, shape, dtype=np.float):
        self._mean = RunningMean(shape, dtype=dtype)
        self._std = RunningStdDev(shape, dtype=dtype)

    @classmethod
    def from_first_values(cls, values):
        inst = cls(values.shape, dtype=values.dtype)
        inst.update(values)
        return inst

    def update(self, values):
        """Add an array of values to the running mean and standard deviation
        
        Parameters
        ----------
        
        values : array-like
            The values to add to the running mean and standard deviation

        Notes
        -----
            In the current version, only numpy arrays are supported
            for values. You may need to convert Pandas dataframes,
            xarray DataArrays, etc. to numpy arrays first.
        """
        self._mean.update(values)
        self._std.update(values)
        


def _rolling_input_helper(A, window, edges, force_centered):
    """
    Internal helper that verifies that all of the inputs are of the correct type and value
    """
    if A.ndim != 1:
        raise NotImplementedError('Rolling operations not yet implemented for arrays with dimension != 1')
    elif not np.issubdtype(type(window), np.signedinteger) or window < 1:
        raise ValueError('window must be a positive integer (> 0)')

    if force_centered and window % 2 == 0:
        window += 1

    # First we pad the input array ourselves to have control over what values are used to extend the array
    extend_width = int(window/2)
    if edges == 'zeros':
        A = np.concatenate([np.zeros((extend_width,), dtype=A.dtype), A, np.zeros((extend_width,), dtype=A.dtype)])
    elif edges == 'nans':
        A = np.concatenate([np.full((extend_width,), np.nan, dtype=A.dtype), A, np.full((extend_width,), np.nan, dtype=A.dtype)])
    elif edges == 'extend':
        A = np.concatenate([np.repeat(A[0], extend_width), A, np.repeat(A[-1], extend_width)])
    else:
        raise ValueError('edges must be "zeros" or "extend"')

    return A, window


def rolling_mean2(A, window, edges='zeros', force_centered=False):
    """Compute a rolling mean over an array.

    A : numpy.ndarray
        The array to operate on. Must be 1D currently

    window : int
        The size of the averaging window to use.

    edges : str
        Optional, how to treat points at edges:

        * 'zeros' (default) prepends and appends 0s to A in order to give enough points for the averaging windows near
          the edge of the array. So for `A = [1, 2, 3, 4, 5]`, with a window of 3, the first window would average
          `[0, 1, 2]` and the last one `[4, 5, 0]`.
        * 'extend' repeats the first and last values instead of using zeros, so in the above example, the first window
          would be `[1, 1, 2]` and the last one `[4, 5, 5]`.

    force_centered : bool
        If `False`, does nothing. If `True`, forces the window to be an odd number by adding one
        if it is even. This matters because with an even number the averaging effectively acts "between" points, so for an
        array of `[1, 2, 3]`, a window of 2 would make the windows `[0, 1]`, `[1, 2]`, `[2, 3]`, and `[3, 0]`.
        As a result, the returned array has one more element than the input. Usually it is more useful to have windows
        centered on values in A, so this parameter makes that more convenient.

    Returns
    -------
    numpy.ndarray
        an array, `B`, either the same shape as A or one longer.
    """
    A, window = _rolling_input_helper(A, window, edges, force_centered)

    # The fact that we can do this with a convolution was described https://stackoverflow.com/a/22621523
    # The idea is, for arrays, convolution is basically a dot product of one array with another as one slides
    # over the other. So using a window function that is 1/N, where N is the window width, this is exactly an average.
    conv_fxn = np.ones((window,), dtype=A.dtype) / window

    # Using mode='valid' will restrict the running mean to t
    return np.convolve(A, conv_fxn, mode='valid')


def rolling_nanmean(A, window, edges='zeros', force_centered=False, window_behavior='default'):
    """Convenience wrapper around :func:`rolling_op`  that does a nanmean, after replacing masked values in A with NaN

    If A is not a masked array (so is just a regular :class:`numpy.ndarray`) it is not changed before doing the nanmean.

    Parameters
    ----------
    window_behavior : str
        Optional, changes what happens when the window size is greater than the size of A. The
        default behavior (given by the string `'default'`) is to extend the array as needed following the `edges` keyword
        and do the rolling mean as always. 'limited' instead mimics the behavior of the `runmean` function in Matlab that
        returns an array the same size as A filled with nanmean(A) if the window width exceeds the length of A.

    Notes
    -----
        See :func:`rolling_op` for documentation of the other parameters.
    """
    if isinstance(A, ma.masked_array):
        A = A.filled(np.nan)

    if window >= A.size and window_behavior == 'limited':
        # This mimics the behavior of the matlab `runmean` function used in the original OCO validation code. In that
        # function, if the window was larger than the array given, it returned an array the same size as A filled with
        # the mean of A
        return np.repeat(np.nanmean(A), A.size)

    A = rolling_op(np.nanmean, A, window, edges=edges, force_centered=force_centered)
    return ma.masked_where(np.isnan(A), A)


def rolling_nanstd(A, window, edges='zeros', force_centered=False, ddof=0):
    """Convenience wrapper around :func:`rolling_op`  that does a nanstd, after replacing masked values in A with NaN

    If A is not a masked array (so is just a regular :class:`numpy.ndarray`) it is not changed before doing the nanstd.

    See :func:`rolling_op` for documentation of the standard parameters.

    Parameters
    ----------
    ddof : int
        Degrees of freedom in the denominator. `ddof=1` gives the usual sample standard deviation.
    """
    if isinstance(A, ma.masked_array):
        A = A.filled(np.nan)

    def std(x, axis=None):
        return np.nanstd(x, ddof=ddof, axis=axis)

    A = rolling_op(std, A, window, edges=edges, force_centered=force_centered)
    return ma.masked_where(np.isnan(A), A)


def rolling_op(op, A, window, edges='zeros', force_centered=False):
    """Carry out a numpy array operation over a rolling window over an array.

    Parameters
    ----------
    op : callable
        The operation to do, function that will receive each window as a subset of a numpy array. Common examples
        are :func:`numpy.mean` and :func:`numpy.sum`

    A : numpy.ndarray
        The array to operate on. Must be 1D currently

    window : int
        The size of the averaging window to use.

    edges : str
        Optional, how to treat points at edges:

        * 'zeros' (default) prepends and appends 0s to A in order to give enough points for the averaging windows near
          the edge of the array. So for `A = [1, 2, 3, 4, 5]`, with a window of 3, the first window would average
          `[0, 1, 2]` and the last one `[4, 5, 0]`.
        * 'nans' prepends and appends NaNs to A.
        * 'extend' repeats the first and last values instead of using zeros, so in the above example, the first window
          would be `[1, 1, 2]` and the last one `[4, 5, 5]`.

    force_centered : bool
        If `False`, does nothing. If `True`, forces the window to be an odd number by adding one
        if it is even. This matters because with an even number the averaging effectively acts "between" points, so for an
        array of `[1, 2, 3]`, a window of 2 would make the windows `[0, 1]`, `[1, 2]`, `[2, 3]`, and `[3, 0]`.
        As a result, the returned array has one more element than the input. Usually it is more useful to have windows
        centered on values in A, so this paramter makes that more convenient.

    Returns
    -------
    numpy.ndarray
        An array either the same shape as A or one longer.
    """
    A, window = _rolling_input_helper(A, window, edges, force_centered)
    return op(rolling_window(A, window), axis=-1)


def rolling_window(a, window):
    """Create a view into an array that provides rolling windows in the last dimension.

    For example, given an array, `A = [0, 1, 2, 3, 4]`, then for `Awin = rolling_window(A, 3)`,
    `Awin[0,:] = [0, 1, 3]`, `Awin[1,:] = [1, 2, 3]`, etc. This works for higher dimensional arrays
    as well.

    Parameters
    ----------
    a : numpy.ndarray
        the array to create windows on.

    window : int
        the width of windows to use

    Returns
    -------
    view
        a read-only view into the strided array of windows.

    References
    ----------
        1. http://www.rigtorp.se/2011/01/01/rolling-statistics-numpy.html.

    Notes
    -----
    The numpy docs (https://docs.scipy.org/doc/numpy/reference/generated/numpy.lib.stride_tricks.as_strided.html#numpy.lib.stride_tricks.as_strided)
    advise caution using the as_strided method used here because effectively what it does is change how numpy iterates
    through at array at the memory level.
    """

    if not np.issubdtype(type(window), np.signedinteger) or window < 1:
        raise ValueError('window must be a positive integer (>0)')

    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides, writeable=False)


def york_linear_fit(x, std_x, y, std_y, max_iter=100, nboot=10, convergence_limit=1e-15, verbose=False):
    """Carries out an iterative, weighted linear regression.

    This linear fit method is from the paper: York, Evensen, Martinez and Delgado, "Unified equations for the slope,
    intercept, and standard errors of the best straight line", American Journal of Physics, 72(3), March 2004.

    The main inputs (`x`, `std_x`, `y`, `std_y`) must all be the same shape arrays, and must not contain any
    non-finite values (NaN or Inf). If given as masked arrays, non-finite values may be present so long as they are
    masked. Additionally, all values in `std_x` and `std_y` must be > 0.

    Parameters
    ----------
    x : numpy.ndarray or numpy.ma.masked_array
        The x values to fit.

    std_x : numpy.ndarray or numpy.ma.masked_array
        The uncertainty or error in the x values.
    :type std_x: :class:`numpy.ndarray` or :class:`numpy.ma.masked_array`

    y : numpy.ndarray or numpy.ma.masked_array
        The y values to fit.

    std_y : numpy.ndarray or numpy.ma.masked_array
        The uncertainty or error in the y values.

    max_iter : int
        The maximum number of iterations to go through. If this is exceeded, a :class:`ConvergenceError`
        will be raised.

    nboot : int
        Number of bootstrap samplings to do. Not implemented currently.

    convergence_limit : float
        Absolute value that the difference between successive slopes must be below to converge.

    verbose : bool
        If `True`, print information about each iteration

    Returns
    -------
    dict
        dictionary containing the keys "slope", "yint", "slope_err", and "yint_err".
    """
    # Input checking:
    #
    # Some of the calculations involved in this fitting procedure are fragile if given invalid values, especially infs
    # or NaNs, but also 0s in the case of the std's (since we put them in the denominator when calculating weights).
    # Further, I don't trust the results if given negative weights - I don't know what that would mean, or if it would
    # be handled correctly by the code as written - so I require that the std's all be > 0.
    #
    # Since we do all the operation with masked array ops, if the input is a masked array that has the offending NaN,
    # Inf, or <= 0 values masked, this should work. Note that, unlike some numpy ops, np.isfinite and np.isnan seem to
    # return a masked array if operating on a masked array.
    shp = x.shape
    inputs = [x, std_x, y, std_y]
    if any([v.shape != shp for v in inputs]):
        raise ValueError('x, std_x, y, and std_y must all have the same shape')
    elif ma.any(ma.less_equal(std_x, 0)) or ma.any(ma.less_equal(std_y, 0)):
        raise ValueError('Negative or zero values of std_x and std_y are not permitted')
    elif any([ma.any(~np.isfinite(v)) for v in inputs]):
        raise ValueError('Non-finite values (Inf or NaN) are not permitted in the input arrays. Either remove or mask '
                         'them.')

    # 1.) Choose an initial value of the slope. Could do a simple y-on-x regression, but here we just assume it starts
    # at 0.
    b_prev = 0

    # 2.) Compute weights in X and Y. As is fairly standard, these are set to be the inverse square of the
    # uncertainties of X and Y.
    w_x = std_x ** -2.0
    w_y = std_y ** -2.0

    # We're assuming no correlation between X and Y
    r = np.zeros_like(x)

    iterations = 0
    while True:
        # 3.) Calculate the overall weights for each point
        w_denom = w_x + b_prev**2 * w_y - 2 * b_prev * r * (w_x * w_y)**0.5
        w = w_x * w_y / w_denom

        # 4.) Calculate the distance of each observed point from its average, then use this to estimate beta
        xbar = ma.sum(w * x) / ma.sum(w)
        ybar = ma.sum(w * y) / ma.sum(w)
        u = x - xbar
        v = y - ybar
        beta = w * (u / w_y + b_prev * v / w_x - (b_prev * u + v) * (r / np.sqrt(w_x * w_y)))

        # 5.) Improve the estimate of the slopw
        b = ma.sum(w * beta * v)/ma.sum(w * beta * u)

        # 6.) Iterate until two successive slopes are within the convergence limit.
        if verbose:
            print('b = {:.4g}, delta b = {:.4g}'.format(b, b - b_prev))

        if np.abs(b - b_prev) < convergence_limit:
            break
        else:
            b_prev = b
            iterations += 1

        if iterations > max_iter:
            raise ConvergenceError('York fit did not converge in {} iterations'.format(max_iter))

    # 7.) Calculate the intercept from the most recent average values of x, y, and b (slope)
    a = ybar - b * xbar

    # 8.) Calculate the "adjusted" values
    x_i = xbar + beta
    y_i = xbar + beta

    # 9.) recalculate the U and V (distance from slope) values with the adjusted points
    xbar_i = ma.sum(w * x_i) / ma.sum(w)
    ybar_i = ma.sum(w * y_i) / ma.sum(w)
    u_i = x_i - xbar_i
    v_i = y_i - ybar_i

    # 10.) Calculate the uncertainty in the slope and intercept
    sigma_b = np.sqrt(1 / ma.sum(w * u_i**2))
    sigma_a = np.sqrt(1 / ma.sum(w) + xbar_i**2 * sigma_b**2)

    results = {'slope': b, 'yint': a, 'slope_err': sigma_b, 'yint_err': sigma_a}
    return results


def r(y, x, nans='error'):
    """Calculate the Pearson's correlation coefficient

    Parameters
    ----------
    y : array-like float or array
        The y-values, i.e. dependent data

    x : array-like 
        The x-values, i.e. independent data

    Returns
    -------
    float
        The r coefficient
    """
    x, y, _, _ = _handle_nans(nans, x, y)
    corr = np.corrcoef(x, y)
    return corr[0, 1]  # the off-diagonal terms of the correlation matrix give the correlation coefficient


def r2(y_data, y_pred=None, x=None, model=None):
    """Calculate an R2 value for a fit.

    Uses the standard definiton of :math:`R^2` (https://en.wikipedia.org/wiki/Coefficient_of_determination). Can be
    given either predicted y values, or x values and a model to predict the y values with, but not both.

    Parameters
    ----------
    y_data : numpy.ndarray or numpy.ma.masked_array
        The true y values.

    y_pred : numpy.ndarray or numpy.ma.masked_array
        The y values predicted by the fit for the same x-values as `y_data` are given at. Must be omitted
        if `x` and `model` are given.

    x : numpy.ndarray or numpy.ma.masked_array
        x values that `y_data` is defined on, to be used to predict the new y values if not given. Must be
        omitted if `y_pred` is given.

    model : PolyFitModel
        a fitting model to use to predict new y-values from the x values. Must be omitted if `y_pred` is
        given.

    Returns
    -------
    numpy.float
        The R2 value.
    """
    if y_pred is not None and (x is not None or model is not None):
        raise TypeError('Give either y_pred or x + model, not all of them')

    if y_pred is None:
        if x is None or model is None:
            raise TypeError('If y_pred is not given, must provide both x and model')
        elif not isinstance(model, PolyFitModel):
            raise TypeError('model must be a PolyFitModel instance, if given')

        y_pred = model.predict(x)

    # Definition of R2: 1 - SS_resid / SS_var, where:
    #   SS_resid = sum( (y_fit - y_data)**2 )
    #   SS_var = sum( (y_data - mean(y_data))**2 )
    ss_resid = ma.sum((y_pred - y_data)**2.0)
    ss_var = ma.sum((y_data - ma.mean(y_data))**2.0)
    return 1.0 - ss_resid / ss_var


def fisher_z_test(x=None, y=None, p=0.05, two_tailed=True):
    """Test if a regression is significant using Fisher z-transformation of the Pearson correlation

    Parameters
    ----------
    x : array-like
        The x-values (independent data)

    y : array-like
        The y-values (dependent data)

    p : float
        The p-value, i.e. the critical probability that the regression is not significant.

    two_tailed : bool
        Whether to treat the p-value as two-tailed, i.e. the regression may be positive or negative
        (either side of the null hypothesis) or can only be above or below.

    Returns
    -------
    bool
        `True` if the regression is significant, `False` otherwise.
    """
    def get_n(vec):
        vec = np.ma.masked_invalid(vec)
        return np.sum(~vec.mask)

    # I was not able to find a clear answer as to exactly how to test that a Z-score indicates that a slope is
    # significantly different from 0. What I typically found was things like this:
    #
    #   https://www.statisticssolutions.com/comparing-correlation-coefficients/
    #
    # that tell how to compare two correlation coefficients. As a result, I decided to take essentially a brute-force
    # approach. When you're testing that a correlation is significant, my understanding is that you're really asking
    # if the predictor, x, gives more information about the value of y than just the mean of y. If the correlation is
    # not significantly different from zero, then the correlation of y with x should be indistinguishable from the
    # correlation of y with its own mean (or really any constant).
    #
    # We test this by calculating z-scores for both the actual correlation of x and y and the correlation of y and its
    # mean
    rval = r(y, x)
    rval_null = r(np.full_like(y, np.mean(y).item()), x)

    z = np.arctanh(rval)
    znull = np.arctanh(rval_null)

    # From the website above, the formula for the difference between the z-scores is this. A characteristic of Fisher
    # z-scores is that they always have standard errors that tend towards 1/(n-3). Since both our z-scores come from
    # vectors with the same n, we can simplify the denominator slightly. from 1/(n1-3) + 1/(n2-3) to 2/(n-3).
    n = min(get_n(x), get_n(y))
    zobs = (z - znull) / np.sqrt(2*(1.0/(n-3)))

    # A Fisher z-transformation takes a r value, which is not necessarily normal, and transforms it into a normally
    # distributed quantity. The PPF i.e. percent-point function i.e. quantile function, is the inverse of the cumulative
    # distribution function. The CDF gives the total probability that a random draw from a given distribution lies
    # between negative infinity and the input value; so the PPF takes in a probability, p, and outputs a value that a
    # random draw from the distribution has probability p of being less than.
    #
    # So st.norm.ppf(0.95) = 1.64485, that means a normal distribution has a 95% chance of yielding a value <= 1.64485
    # and a 5% chance of a value > 1.64485. In the two tailed case, we actually want the value v such that a draw d has
    # the given probability of being -v <= d <= v, so we need to half the probability on the positive side.
    if two_tailed:
        p /= 2
    zcrit = st.norm.ppf(1.0 - p)

    # If our observed z value for the difference between the actual regression and the mean exceeds the critical value,
    # that means that there is a < p chance that they are the same.
    return np.abs(zobs) > zcrit


def bin_1d(data, coord, bins, **kwargs):
    """Bin 1D data and apply an arbitrary operation to the bins.

    Parameters
    ----------

    data : array-like
        The data to bin

    coord : array-like
        An array that gives the coordinates that the data will be binned by.

    bins : int or array-like
        An integer specifying the number of bins, or a vector (`v`) specifying bin edges where the edges for
        bin `i` are defined by `v[i]` and `v[i+1]`.

    kwargs :
        Additional keyword arguments to be passed to :func:`bin_nd`.

    Returns
    -------
    numpy.ndarray
        Binned data 

    array-like [optional]
        If requested, the bin edges or bin centers
    """
    retvals = bin_nd(data, [coord], [bins], **kwargs)
    if isinstance(retvals, tuple):
        # If the user requested that the bin edges/centers be returned, retvals will be a tuple. We want to extract the
        # bins out of the list of bins (since bin_nd needs to return one set of bins per dimension, but in this case
        # we only have one).
        if len(retvals) == 2:
            retvals = (retvals[0], retvals[1][0])
        elif len(retvals) > 2:
            raise NotImplementedError('expected 1 or 2 return values from bin_nd')

    return retvals


def bin_nd(data, coords, bins, op=np.size, out=None, ret_bins='no'):
    """Bin multidimensional data and apply an arbitrary operation to the bins.

    Parameters
    ----------
    data : array-like
        The data to bin

    coords : Sequence[array-like]
        A collection of arrays that define the coordinates of the data. Each array must have the same number
        of elements as data. The number of arrays will determine the dimensionality of the output array, i.e. if there
        are two coordinate arrays, the output array will be 2D.

    bins : Sequence[array-like]
        A collection specifying the bins to use for each coordinate. That is, `coords[0]` will be binned
        according to `bins[0]` and so on. Bins may either be specified as a integer, which gives the number of bins, or
        a vector, `v` such that `v[i]` and `v[i+1]` specify the edges for bin `i`.

    op : callable
        A function that takes an array as the sole input and returns a scalar value as the output. That value
        will be the binned value. The default is :func:`numpy.size`, which will cause the binned value to be the number of
        values in the bin.

    out : array-like or None
        If given, an array to place the binned values in. Its shape must be (n1 x n2 x ...) where n1,
        n2, etc. are the number of bins for each dimension. If not given, it is created and automatically.

    ret_bins : str or bool
        Specifies whether to return the bins as the second output:

        * "no", "n", or `False` will not (only one return value).
        * "yes", "y", `True`, "edge", or "edges" will return the bin edges as a collection.
        * "center" or "centers" will return the bin centers as a collection.


    Returns
    -------
    numpy.ndarray
        The binned values as an array with number of dimensions equal to the number of coordinates given, and shape
        equal to the number of bins in each dimension. 

    Sequence[array-like]
        Optionally, also return the bin edges or centers for each dimension as a collection of arrays (depends on
        `ret_bins`).
    """
    if len(coords) != len(bins):
        raise ValueError('coords and bins must be the same length')

    # We can use Pandas's groupby feature to group data from the same bin together so we need to construct a dataframe
    # that has columns containing the bin indices for each dimension.
    bins = deepcopy(bins)
    df_dict = {'data': data.flatten()}
    coord_cols = []
    for i, (coord, bin) in enumerate(zip(coords, bins)):
        if isinstance(bin, int):
            bin = np.linspace(np.min(coord), np.max(coord), bin+1)
            bin[-1] *= 1.001  # scale the last bin edge to ensure the max value gets included in the last bin
            bins[i] = bin
        inds = np.digitize(coord.flatten(), bin) - 1
        colname = 'ind{}'.format(i)
        df_dict[colname] = inds
        coord_cols.append(colname)

    df = pd.DataFrame(df_dict)
    if out is None:
        out = np.full([np.size(b)-1 for b in bins], np.nan if np.issubdtype(data.dtype, np.floating) else 0)

    for inds, grp in df.groupby(coord_cols):
        subdata = grp['data']
        if isinstance(inds, int) and inds < 0:
            continue
        elif not isinstance(inds, int) and any(i < 0 for i in inds):
            continue
        try:
            out[inds] = op(subdata)
        except IndexError:
            # inds will be too large if any of the values is outside the right edge of the last bin, so just skip those
            # since we don't want to bin them. We caught any outside the left edge of the bin checking if any are < 0
            continue

    if ret_bins in ('centers', 'center'):
        bin_centers = [0.5*(b[:-1] + b[1:]) for b in bins]
        return out, bin_centers
    elif ret_bins in ('edges', 'edge', 'yes', 'y', True):
        return out, bins
    elif ret_bins in ('no', 'n', False):
        return out
    else:
        raise ValueError('Unrecognized value for ret_bins: "{}"'.format(ret_bins))


def hist2d(x, y, xbins=10, ybins=10):
    """

    :param x: The x data. Assumed to be ungridded, i.e. every data point is specified in this array. May be any number
     of dimensions.
    :type x: :class:`numpy.ndarray`

    :param y: The y data. Same rules as ``x``.
    :type y: :class:`numpy.ndarray`

    :param xbins: If given as an integer, specifies the number of bins to use in the x-dimension. The bins will be
     linearly spaced. If given as a 1D array, then it is assumed to specify the bin edges (and thus has nbins + 1
     values).
    :type xbins: int or :class:`numpy.ndarray`

    :param ybins: Specifies the bins for the y-dimension according to the same rules as ``xbins``.
    :type ybins: int or :class:`numpy.ndarray`

    :return: a 2D array of counts, the x bin edges, and the y bin edges.
    :rtype: :class:`numpy.ndarray`, :class:`numpy.ndarray`, :class:`numpy.ndarray`.
    """
    def check_bins(coord, bins, name):
        if np.ndim(bins) == 0:
            # Expand the top bin just a little bit so we get the max point
            bins = np.linspace(np.nanmin(coord), np.nanmax(coord)*1.0001, bins+1)
        elif np.ndim(bins) != 1:
            raise TypeError('{} must be a scalar integer or a 1D vector'.format(name))
        elif np.any(np.diff(bins) < 0):
            raise ValueError('{} must be monotonically increasing'.format(name))

        return bins

    xbins = check_bins(x, xbins, 'xbins')
    ybins = check_bins(y, ybins, 'ybins')
    nx = np.size(xbins) - 1
    ny = np.size(ybins) - 1

    counts = np.zeros([nx, ny], dtype=np.int)
    for i in range(nx):
        xx = (x >= xbins[i]) & (x < xbins[i+1])
        for j in range(ny):
            yy = (y >= ybins[j]) & (y < ybins[j+1])
            counts[i, j] = np.sum(xx & yy)

    return counts, xbins, ybins


def nanabsmax(a, *args, **kwargs):
    """Calculate the signed maximum distance from zero in a dataset

    For a vector, v, returns the element vi for which |vi| is greatest.

    Parameters
    ----------
    a : numpy.ndarray
        The array to operate on.

    args
        Additional positional arguments recognized by nanmax and nanmin.
    
    kwargs
        Additional keyword arguments recognized by nanmax and nanmin. This and `args` can be use to operate
        along a specific axis for example.

    Returns
    -------
    number or array
        The value or values farthest from 0.
    """
    amax = np.nanmax(a, *args, **kwargs)
    amin = np.nanmin(a, *args, **kwargs)
    if np.ndim(amax) > 0:
        xx = np.abs(amax) < np.abs(amin)
        amax[xx] = amin[xx]
    elif np.abs(amax) < np.abs(amin):
        # scalar values - can't do item assignment
        amax = amin

    return amax


def _handle_nans(method, x, y, xerr=None, yerr=None):
    not_nans = np.isfinite(x) & np.isfinite(y)
    if xerr is not None:
        not_nans &= np.isfinite(xerr)
    if yerr is not None:
        not_nans &= np.isfinite(yerr)

    if method.lower() == 'ignore':
        pass
    elif method.lower() == 'error':
        if np.any(~not_nans):
            raise FittingDataError('NaNs or infs found in the x, y, xerr, or yerr vectors')
    elif method.lower() == 'drop':
        x = x[not_nans]
        y = y[not_nans]
        xerr = xerr[not_nans] if xerr is not None else xerr
        yerr = yerr[not_nans] if yerr is not None else yerr
    else:
        raise ValueError('Method to handle bad values must be one of "ignore", "error", or "drop"')

    return x, y, xerr, yerr


def reldiff(a, b, denom='first'):
    """Compute the fraction difference between two quantities
    
    Given two numbers, `a` and `b`, computes :math:`(b - a)/|a|`.

    Parameters
    ----------
    
    a, b : numeric
        Values to compute the relative difference between.

    denom : str
        How the denominator is computed. The default, `"first"`, sets the denominator to `a`. Alternately,
        `"mean"` will use the mean of a and b.

    Returns
    -------
    
    numeric
        The fractional difference between a and b.
    """
    if denom == 'first':
        d = np.abs(a)
    elif denom == 'mean':
        d = np.abs(0.5*(a+b))
    else:
        raise ValueError('Unrecognized value for `denom`: {}'.format(denom))

    return (b - a)/d

