class UnitConverter(object):
    """Simple class to define unit conversions

    This class offers a way to define simple conversions between units of the same type. It does *not* implement a full
    grammar of units, so it cannot carry out complicated unit conversions or propagate units through calculations. For
    that, consider using something like Pint (https://pint.readthedocs.io/en/0.11/). This is intended to be a quick
    converter when you want to carry out a conversion between arbitrary units without a full refactoring to use
    something like Pint.

    To use, first instantiate an instance of this class, defining the unit that will serve as the "base" unit for this
    family of units. All other conversions will be given relative to this. Conversions can be added during
    initialization or after the fact. When adding conversions, the `metric_prefixes` argument indicates whether or not
    the units given should also recognize the metric prefixes:

     * If `metric_prefixes` is `False`, then no prefixes are recognized
     * If `metric_prefixes` is `True` or `'abbrv'` then the standard prefix abbreviations (e.g. m, c, k, M, G...) are
       recognized.
     * If `metric_prefixes` is `'full'` then the full written-out prefixes are recognized.
     * If `metric_prefixes` is `None`, then the default (specified during initialization) is followed.

    To convert a value, use the `convert` method. Calling the instance as a function is equivalent. Alternatively, the
    `fixed_convert` method returns a function that will always convert an input value to the same unit, specified in the
    call to `fixed_convert`.

    Several unit converters are predefined in this module: `length`, `mass`, `time`, `pressure`, `temperature`, and
    `mixing_ratios`.
    """
    _metric_prefix_abbrv = dict(Y=1e24, Z=1e21, E=1e18, P=1e15, T=1e12, G=1e9, M=1e6, k=1e3, h=1e2, da=1e1,
                                d=1e-1, c=1e-2, m=1e-3, u=1e-6, n=1e-9, p=1e-12, f=1e-15, a=1e-18, z=1e-21, y=1e-24)
    _metric_prefix_full  = dict(yotta=1e24, zetta=1e21, exa=1e18, peta=1e15, tera=1e12, giga=1e9, mega=1e6, kilo=1e3, hecto=1e2, deka=1e1,
                                deci=1e-1, centi=1e-2, milli=1e-3, micro=1e-6, nano=1e-9, pico=1e-12, femto=1e-15, atto=1e-18, zepto=1e-21, yocto=1e-24)

    def __init__(self, base_unit, metric_prefixes=True, **conversions):
        """Create a new converter.

        Parameters
        ----------
        base_unit : str
            the name or abbreviation of the "base" unit for this instance - all other conversions will be relative to
            this one.

        metric_prefixes : bool, str, or None
            whether to recognize metric prefixes for this unit. See class help for allowed values.

        conversions
            keyword-value pairs giving additional units and their conversion factors. The value must be the number of
            the base unit that goes into the new unit. For example, if the base unit is `'seconds'`, then `minutes=60`
            would specify the correct conversion.
        """
        self.base_unit = base_unit
        self.allow_metric = metric_prefixes
        self.conversions = dict()
        self.add_conversion(base_unit, 1.0)
        for unit, scale in conversions.items():
            self.add_conversion(unit, scale)

    def __call__(self, *args, **kwargs):
        """Alias for `convert`
        """
        return self.convert(*args, **kwargs)

    def _get_prefixes(self, metric_prefixes):
        if metric_prefixes is None:
            metric_prefixes = self.allow_metric

        if metric_prefixes is False:
            return dict()
        elif metric_prefixes == 'full':
            return self._metric_prefix_full
        elif metric_prefixes is True or metric_prefixes == 'abbrv':
            return self._metric_prefix_abbrv
        else:
            raise ValueError('Invalid value for "metric_prefixes": {}'.format(metric_prefixes))

    def add_conversion(self, unit, scale, add=0.0, metric_prefixes=None):
        """Add a single new conversion.

        This method offers more flexibility than `addi_multiple_conversions` or the init method, but each new unit
        requires its own method call. The conversion formula is:

        ..math:
            base_unit / scale + add

        Parameters
        ----------
        unit : str
            the name or abbreviation of the new unit

        scale : float
            amount to scale the base unit to reach this unit. That is, assuming a base unit of seconds, the scale for
            minutes is 60.

        add : float
            offset to add to the value after the scaling is done. Note that this happens before any scaling for the
            metric prefix so that, e.g., millikelvin work correctly.

        metric_prefixes : bool, str, or None
            whether to recognize metric prefixes for this unit. See class help for allowed values.
        """
        self.conversions[unit] = self._std_conversion_factory(scale, add)
        prefixes = self._get_prefixes(metric_prefixes)
        for prefix, prescale in prefixes.items():
            self.conversions['{}{}'.format(prefix, unit)] = self._std_conversion_factory(scale, add, prescale)

    def add_multiple_conversions(self, metric_prefixes=None, **conversions):
        """Add multiple conversions with a single call

        Only supports conversions that only require a scaling. If a conversion requires an additive offset,
        `add_conversion` must be used.

        Parameters
        ----------
        metric_prefixes : bool, str, or None
            whether to recognize metric prefixes for this unit. See class help for allowed values.

        conversions
            keyword-value pairs giving additional units and their conversion factors. The value must be the number of
            the base unit that goes into the new unit. For example, if the base unit is `'seconds'`, then `minutes=60`
            would specify the correct conversion.
        """
        for unit, scale in conversions.items():
            self.add_conversion(unit, scale, metric_prefixes=metric_prefixes)

    def add_conversion_lambda(self, unit, conv_from, conv_to, metric_prefixes=None):
        """Add a single conversion by specifying two functions to convert to and from this unit.

        This is the most flexible, but the most cumbersome way to define a conversion between two units. By providing
        one function to convert from the base unit to this unit and another to convert back to the base unit, any
        mathematical conversion is possible.

        Parameters
        ----------
        unit : str
            the name or abbreviation for this unit

        conv_from : callable
            a function that accepts a single input, the value to be converted *from* this unit *to* the base unit for
            this converter. The input will have any scaling due to metric prefixes removed before calling this function.

        conv_to
            a function that accepts a single input, the value to be converted *to* this unit *from* the base unit for
            this converter. Any scaling due to metric prefixes is applied on the return value from this function.

        metric_prefixes : bool, str, or None
            whether to recognize metric prefixes for this unit. See class help for allowed values.
        """
        self.conversions[unit] = self._lambda_conversion_factory(conv_from, conv_to)
        prefixes = self._get_prefixes(metric_prefixes)
        for prefix, prescale in prefixes.items():
            self.conversions['{}{}'.format(prefix, unit)] = self._lambda_conversion_factory(conv_from, conv_to, prescale)

    @staticmethod
    def _std_conversion_factory(scale, add, prescale=1.0):
        def conv_to(v):
            return (v/scale + add) / prescale

        def conv_from(v):
            return (v*prescale - add)*scale

        return {'from': conv_from, 'to': conv_to}

    @staticmethod
    def _lambda_conversion_factory(from_user_fxn, to_user_fxn, prescale=1.0):
        def conv_to(v):
            return to_user_fxn(v) / prescale

        def conv_from(v):
            return from_user_fxn(prescale * v)

        return {'from': conv_from, 'to': conv_to}

    def convert(self, value, old_unit, new_unit):
        """Convert a quantity between units defined on this converter

        Parameters
        ----------
        value : numeric
            the value to convert. May be any numeric type that supports addition/subtraction/multiplication/division
            with scalar floats.

        old_unit : str
            the unit that the value is currently in.

        new_unit : str
            the unit to convert the value to

        Returns
        -------
        numeric
            the value converted to the new unit.

        """
        base_val = self.conversions[old_unit]['from'](value)
        return self.conversions[new_unit]['to'](base_val)

    def fixed_convert(self, new_unit):
        """Create a function to convert values to a specific unit.

        Parameters
        ----------
        new_unit : str
            the unit to convert values to

        Returns
        -------
        callable
            a function with the signature `(value, old_unit)`.
        """
        def conv(value, old_unit):
            return self.convert(value, old_unit, new_unit)
        return conv


time = UnitConverter('s')
time.add_multiple_conversions(metric_prefixes='full', second=1, seconds=1.0)
time.add_multiple_conversions(metric_prefixes=False,
                              min=60.0, minute=60.0, minutes=60.0,
                              hr=3600.0, hour=3600.0, hours=3600.0,
                              day=24 * 3600.0, days=24 * 3600.0,
                              wk=7 * 24 * 3600.0, week=7 * 24 * 3600.0, weeks=7 * 24 * 3600.0)

mass = UnitConverter('g')
mass.add_multiple_conversions(metric_prefixes='full', gram=1.0, grams=1.0)
mass.add_multiple_conversions(metric_prefixes=False,
                              lb=453.592, pound=453.592, pounds=453.592,
                              oz=28.3495, ounce=28.3495, ounces=28.3495,
                              amu=1.660540199e-24)

length = UnitConverter('m', **{'in': 0.0254, 'ft': 0.3048,  'yd': 0.9144, 'mi': 1609.344})
length.add_multiple_conversions(metric_prefixes=False, **{'"': 0.0254, "'": 0.3048})
length.add_multiple_conversions(metric_prefixes='full',
                                meter=1.0, meters=1.0, metre=1.0, metres=1.0,
                                inch=0.0254, inches=0.0254,
                                feet=0.3048,
                                yard=0.9144, yards=0.9144,
                                mile=1609.344, miles=1609.344)

pressure = UnitConverter('Pa', bar=1e5, b=1e5, atm=1.01325e5)
pressure.add_multiple_conversions(metric_prefixes='full',
                                  pascal=1.0, pascals=1.0,
                                  bar=1e5, bars=1e5,
                                  atmosphere=1.01325e5, atmospheres=1.01325e5)
pressure.add_conversion('mmHg', 133.322, metric_prefixes=False)

temperature = UnitConverter('C', **{'degC': 1.0, 'degrees C': 1.0, 'degrees celsius': 1.0})
temperature.add_conversion('F', 5 / 9, 32)
temperature.add_conversion('K', 1., 273.15)

mixing_ratio = UnitConverter('ppp', ppm=1e-6, ppmv=1e-6, ppb=1e-9, ppbv=1e-9, ppt=1e-12, pptv=1e-12, metric_prefixes=False)
_mol_ratios = {'mol/mol': 1.0, 'mol mol^-1': 1.0, 'mol mol-1': 1.0}
mixing_ratio.add_multiple_conversions(metric_prefixes=True, **_mol_ratios)
mixing_ratio.add_multiple_conversions(metric_prefixes='full', **_mol_ratios)