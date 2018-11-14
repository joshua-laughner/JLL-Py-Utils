import numpy as np
from numpy import ma


class ConvergenceError(Exception):
    """
    Error to use if a process fails to converge
    """
    pass


class PolyFitModel(object):
    """
        Common interface to polynomial fitting methods

        This class provides a common interface to fit 2D data with or without errors associated with the x and y values.
        Instantiating the class automatically does the fitting, and the result is stored in the new instance. Instantiation
        requires the x and y values to fit be given, and, if the model chosen requires it, error for the x and/or y values
        as well. The model can be chosen with the ``model`` keyword. Current options are:

            * 'york' - use :func:`york_linear_fit` (from this module).
            * any function that takes a minimum of four arguments (``x``, ``y``, ``xerr``, and ``yerr``, where ``xerr`` and
              ``yerr`` are the error/uncertainty in ``x`` and ``y``) and returns two iterables (list, tuple, 1D numpy array,
              etc.), the polynomial coefficients and their errors, starting from the x**0 term. The values in the
              coefficient error array may be None if the fit does not compute errors in its coefficients.

        If your chosen model function has additional keyword arguments, you can pass them as a dict with the ``model_opts``
        inpurt.

        Since this class is for polynomial fitting, the result is stored as a list of coefficients, where for:

        .. math::
            p(x) = a_0 + a_1 x + a_2 x^2 + ... + a_n x^n

        the coefficients :math:`a_0`, :math:`a_1`, etc. are stored in the ``coeffs`` property as a list. Only as many
        coefficients as given by the model are stored, so if the model only returns :math:`a_0` and :math:`a_1`, then
        ``coeffs`` will only have two elements and model.coeffs[2] will raise an ``IndexError``.  If you have an application
        where you want the coefficient for a given power of :math:`x` regardless of whether that power is used in the model,
        the ``get_any_coeff`` method will do that.

        The :math:`a_0` and :math:`a_1` coefficients are accessible through the special properties ``yint`` and ``slope``.
        These will always return a value, if not used in the model, 0.0 will be returned.

        Once created, instances of this class can be used to predict :math:`y` values for new :math:`x` values. This is done
        with the ``predict`` method, or by calling the instance itself, e.g.::

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
        """
        Polynomial coefficients for the fit, starting from the x**0 term.

        :rtype: :class:`numpy.ndarray`
        """
        return self._coeffs

    @property
    def coeff_errors(self):
        """
        Errors in the polynomial coefficients for the fit.

        :rtype: :class:`numpy.ndarray`
        """
        return self._coeff_errors

    @property
    def yint(self):
        """
        The y-intercept of the fit, i.e. the x**0 coefficient

        :rtype: :class:`numpy.float`
        """
        return self.get_any_coeff(0)

    @property
    def slope(self):
        """
        The slope of the fit, i.e. the x**1 coefficient

        :rtype: :class:`numpy.float`
        """
        return self.get_any_coeff(1)

    @property
    def data(self):
        """
        The data used to create the fit

        :return: dictionary with keys 'x', 'y', 'xerr', 'yerr'
        :rtype: dict
        """
        return self._data

    def __init__(self, x, y, xerr=None, yerr=None, model='york', model_opts=None):
        # Coerce input data to floats. Check them at the same time
        x = self._fix_in_type(x, 'x')
        y = self._fix_in_type(y, 'y')
        xerr = self._fix_in_type(xerr, 'xerr')
        yerr = self._fix_in_type(yerr, 'yerr')

        # Store data
        self._fit_fxn = self._get_fit_fxn(model)
        self._model_opts = model_opts if model_opts is not None else dict()
        self._data = {'x': x, 'y': y, 'xerr': xerr, 'yerr': yerr}

        # Input checking
        if not isinstance(self._model_opts, dict):
            raise TypeError('Model options must be given as a dictionary')

        # Do the fit
        coeffs, coeff_errors = self._fit_fxn(x, y, xerr, yerr, **self._model_opts)
        self._coeffs = self._check_coeff_type(coeffs)
        self._coeff_errors = self._check_coeff_type(coeff_errors)

    def __str__(self):
        """
        String-formatted representation of the fit.

        :return: string of form "y = a_0 + a_1 x + a_2 x**2 + ..."
        """
        s = 'y = '
        power = 0
        for c, e in zip(self._coeffs, self._coeff_errors):
            if power > 0:
                s += ' + '

            if e is not None:
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
    def _fix_in_type(input, name='input'):
        """
        Correct the type of any data input to this model.

        Any time data is give to this model, it should be run through this method before being stored or used.

        :param input: the input data

        :param name: optional, name of the input to use in error messages.
        :type name: str

        :return: the input coerced to the appropriate type, if posible
        :rtype: :class:`numpy.ndarray` with dtype = :class:`numpy.float`

        :raises: TypeError if the input is of an invalid type
        """
        if not isinstance(input, np.ndarray) or input.ndim != 1:
            raise TypeError('{} must be a 1D numpy array'.format(name))
        return input.astype(np.float)

    def _check_coeff_type(self, coeff):
        """
        Check that arrays of coefficients returned from the fitting functions are the right type

        :param coeff: the array of coefficients. Must be a 1D :class:`numpy.ndarray` or a value that can be converted
         to one by ``numpy.array(coeff)``.
        :type coeff: :class:`numpy.ndarray` or compatible

        :return: the coefficients, converted to a numpy float array necessary and possible.
        :rtype: :class:`numpy.ndarray` with dtype = :class:`numpy.float`

        :raises: TypeError if cannot convert to numpy array or given as an array with ndim != 1.
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
        """
        Internal helper function that maps model names to function calls.

        :param model: the model name, or a function.

        :return: the model fitting function to call. Will accept 4 positional arguments (x, y, xerr, yerr) and possibly
         additional keyword args, though the latter is not guaranteed. Will return two values, coefficients and their
         errors, as iterables.
        :rtype: function
        """

        # Define wrapper functions here. Each one must accept 4 inputs (x, y, xerr, yerr) and will be passed any
        # additional keywords received as the ``model_opts`` parameter in __init__. Each must return a vector of
        # polynomial coefficients and their errors.
        def york_fit(x, y, xerr, yerr, **opts):
            if xerr is None or yerr is None:
                raise ValueError('xerr and yerr are required for the "york" fitting method')
            result = york_linear_fit(x, xerr, y, yerr, **opts)
            poly = np.array([result['yint'], result['slope']])
            poly_err = np.array([result['yint_err'], result['slope_err']])
            return poly, poly_err

        if not isinstance(model, str):
            return model
        elif model.lower() == 'york':
            return york_fit

    def predict(self, x):
        """
        Calculate the y values predicted by the fit for new x values.

        :param x: the array of x values
        :type x: 1D :class:`numpy.ndarray`

        :return: predicted y values, same shape as x.
        :rtype: :class:`numpy.ndarray`, dtype = :class:`numpy.float`.
        """
        x = self._fix_in_type(x, 'x')
        y = np.zeros_like(x)
        for p, c in enumerate(self._coeffs):
            y += c * x**p

        return y

    def get_any_coeff(self, term):
        """
        Get the coefficient from a term in the polynomial fit.

        Unlike the ``coeffs`` property, this method allows you to ask for any coefficient, even one not returned by the
        fitting function. If a coefficient isn't given by the fitting function, will return 0.0.

        :param term: the term to get the coefficient for, i.e. ``term = 1`` will give the coefficient for the x**1
         term in the fit.
        :type term: int

        :return: the coefficient
        :rtype: :class:`numpy.float`
        """
        if term < 0:
            raise ValueError('power must be >= 0. If you are trying to index the list of coefficients from the end, '
                             'use the coeffs property directly, e.g. fit.coeffs[-1]')
        try:
            return self.coeffs[term]
        except IndexError:
            return 0.0


def _rolling_input_helper(A, window, edges, force_centered):
    """
    Internal helper that verifies that all of the inputs are of the correct type and value
    """
    if A.ndim != 1:
        raise NotImplementedError('Rolling operations not yet implemented for arrays with dimension != 1')
    elif not isinstance(window, int) or window < 1:
        raise ValueError('window must be a positive integer (> 0)')

    if force_centered and window % 2 == 0:
        window += 1

    # First we pad the input array ourselves to have control over what values are used to extend the array
    extend_width = int(window/2)
    if edges == 'zeros':
        A = np.concatenate([np.zeros((extend_width,), dtype=A.dtype), A, np.zeros((extend_width,), dtype=A.dtype)])
    elif edges == 'extend':
        A = np.concatenate([np.repeat(A[0], extend_width), A, np.repeat(A[-1], extend_width)])
    else:
        raise ValueError('edges must be "zeros" or "extend"')

    return A, window


def rolling_mean2(A, window, edges='zeros', force_centered=False):
    """
    Compute a rolling mean over an array.

    :param A: The array to operate on. Must be 1D currently
    :type A: :class:`numpy.ndarray`

    :param window: the size of the averaging window to use.
    :type window: int

    :param edges: optional, how to treat points at edges:

        * 'zeros' (default) prepends and appends 0s to A in order to give enough points for the averaging windows near
          the edge of the array. So for ``A = [1, 2, 3, 4, 5]``, with a window of 3, the first window would average
          ``[0, 1, 2]`` and the last one ``[4, 5, 0]``.
        * 'extend' repeats the first and last values instead of using zeros, so in the above example, the first window
          would be ``[1, 1, 2]`` and the last one ``[4, 5, 5]``.
    :type edges: str

    :param force_centered: if ``False``, does nothing. If ``True``, forces the window to be an odd number by adding one
     if it is even. This matters because with an even number the averaging effectively acts "between" points, so for an
     array of ``[1, 2, 3]``, a window of 2 would make the windows ``[0, 1]``, ``[1, 2]``, ``[2, 3]``, and ``[3, 0]``.
     As a result, the returned array has one more element than the input. Usually it is more useful to have windows
     centered on values in A, so this paramter makes that more convenient.
    :type force_centered: bool

    :return: an array, ``B``, either the same shape as A or one longer.
    :rtype: :class:`numpy.ndarray`.
    """
    A, window = _rolling_input_helper(A, window, edges, force_centered)

    # The fact that we can do this with a convolution was described https://stackoverflow.com/a/22621523
    # The idea is, for arrays, convolution is basically a dot product of one array with another as one slides
    # over the other. So using a window function that is 1/N, where N is the window width, this is exactly an average.
    conv_fxn = np.ones((window,), dtype=A.dtype) / window

    # Using mode='valid' will restrict the running mean to t
    return np.convolve(A, conv_fxn, mode='valid')


def rolling_nanmean(A, window, edges='zeros', force_centered=False):
    """
    Convenience wrapper around :func:`rolling_op`  that does a nanmean, after replacing masked values in A with NaN

    If A is not a masked array (so is just a regular :class:`numpy.ndarray`) it is not changed before doing the nanmean.

    See :func:`rolling_op` for documentation of the other parameters.
    """
    if isinstance(A, ma.masked_array):
        A = A.filled(np.nan)

    A = rolling_op(np.nanmean, A, window, edges=edges, force_centered=force_centered)
    return ma.masked_where(np.isnan(A), A)


def rolling_nanstd(A, window, edges='zeros', force_centered=False):
    """
    Convenience wrapper around :func:`rolling_op`  that does a nanstd, after replacing masked values in A with NaN

    If A is not a masked array (so is just a regular :class:`numpy.ndarray`) it is not changed before doing the nanstd.

    See :func:`rolling_op` for documentation of the other parameters.
    """
    if isinstance(A, ma.masked_array):
        A = A.filled(np.nan)

    A = rolling_op(np.nanstd, A, window, edges=edges, force_centered=force_centered)
    return ma.masked_where(np.isnan(A), A)


def rolling_op(op, A, window, edges='zeros', force_centered=False):
    """
    Carry out a numpy array operation over a rolling window over an array.

    :param op: the operation to do, function that will receive each window as a subset of a numpy array. Common examples
     are :func:`numpy.mean` and :func:`numpy.sum`

    :param A: The array to operate on. Must be 1D currently
    :type A: :class:`numpy.ndarray`

    :param window: the size of the averaging window to use.
    :type window: int

    :param edges: optional, how to treat points at edges:

        * 'zeros' (default) prepends and appends 0s to A in order to give enough points for the averaging windows near
          the edge of the array. So for ``A = [1, 2, 3, 4, 5]``, with a window of 3, the first window would average
          ``[0, 1, 2]`` and the last one ``[4, 5, 0]``.
        * 'extend' repeats the first and last values instead of using zeros, so in the above example, the first window
          would be ``[1, 1, 2]`` and the last one ``[4, 5, 5]``.
    :type edges: str

    :param force_centered: if ``False``, does nothing. If ``True``, forces the window to be an odd number by adding one
     if it is even. This matters because with an even number the averaging effectively acts "between" points, so for an
     array of ``[1, 2, 3]``, a window of 2 would make the windows ``[0, 1]``, ``[1, 2]``, ``[2, 3]``, and ``[3, 0]``.
     As a result, the returned array has one more element than the input. Usually it is more useful to have windows
     centered on values in A, so this paramter makes that more convenient.
    :type force_centered: bool

    :return: an array, ``B``, either the same shape as A or one longer.
    :rtype: :class:`numpy.ndarray`.
    """
    A, window = _rolling_input_helper(A, window, edges, force_centered)
    return op(rolling_window(A, window), axis=-1)


def rolling_window(a, window):
    """
    Create a view into an array that provides rolling windows in the last dimension.

    For example, given an array, ``A = [0, 1, 2, 3, 4]``, then for ``Awin = rolling_window(A, 3)``,
    ``Awin[0,:] = [0, 1, 3]``, ``Awin[1,:] = [1, 2, 3]``, etc. This works for higher dimensional arrays
    as well.

    :param a: the array to create windows on.
    :type a: :class:`numpy.ndarray`

    :param window: the width of windows to use
    :type window: int

    :return: a read-only view into the strided array of windows.

    From http://www.rigtorp.se/2011/01/01/rolling-statistics-numpy.html.

    Note: the numpy docs (https://docs.scipy.org/doc/numpy/reference/generated/numpy.lib.stride_tricks.as_strided.html#numpy.lib.stride_tricks.as_strided)
    advise caution using the as_strided method used here because effectively what it does is change how numpy iterates
    through at array at the memory level.
    """

    if not isinstance(window, int) or window < 1:
        raise ValueError('window must be a positive integer (>0)')

    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides, writeable=False)


def york_linear_fit(x, std_x, y, std_y, max_iter=100, nboot=10):
    """
    Carries out an iterative, weighted linear regression.

    This linear fit method is from the paper: York, Evensen, Martinez and Delgado, "Unified equations for the slope,
    intercept, and standard errors of the best straight line", American Journal of Physics, 72(3), March 2004.

    The main inputs (``x``, ``std_x``, ``y``, ``std_y``) must all be the same shape arrays, and must not contain any
    non-finite values (NaN or Inf). If given as masked arrays, non-finite values may be present so long as they are
    masked. Additionally, all values in ``std_x`` and ``std_y`` must be > 0.

    :param x: the x values to fit.
    :type x: :class:`numpy.ndarray` or :class:`numpy.ma.masked_array`

    :param std_x: the uncertainty or error in the x values.
    :type std_x: :class:`numpy.ndarray` or :class:`numpy.ma.masked_array`

    :param y: the y values to fit.
    :type y: :class:`numpy.ndarray` or :class:`numpy.ma.masked_array`

    :param std_y: the uncertainty or error in the y values.
    :type std_y: :class:`numpy.ndarray` or :class:`numpy.ma.masked_array`

    :param max_iter: the maximum number of iterations to go through. If this is exceeded, a :class:`ConvergenceError`
     will be raised.
    :type max_iter: int

    :param nboot: number of bootstrap samplings to do. Not implemented currently.
    :type int:

    :return: dictionary containing the keys "slope", "yint", "slope_err", and "yint_err".
    :rtype: dict
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
        if np.abs(b - b_prev) < 1e-15:
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


def r2(y_data, y_pred=None, x=None, model=None):
    """
    Calculate an R2 value for a fit.

    Uses the standard definiton of :math:`R^2` (https://en.wikipedia.org/wiki/Coefficient_of_determination). Can be
    given either predicted y values, or x values and a model to predict the y values with, but not both.

    :param y_data: the true y values.
    :type y_data: :class:`numpy.ndarray` or :class:`numpy.ma.masked_array`

    :param y_pred: the y values predicted by the fit for the same x-values as ``y_data`` are given at. Must be omitted
     if ``x`` and ``model`` are given.
    :type y_pred: :class:`numpy.ndarray` or :class:`numpy.ma.masked_array`

    :param x: x values that ``y_data`` is defined on, to be used to predict the new y values if not given. Must be
     omitted if ``y_pred`` is given.
    :type x: :class:`numpy.ndarray` or :class:`numpy.ma.masked_array`

    :param model: a fitting model to use to predict new y-values from the x values. Must be omitted if ``y_pred`` is
     given.
    :type model: :class:`PolyFitModel`

    :return: the R2 value.
    :rtype: :class:`numpy.float`
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