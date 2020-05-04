class UnitConverter(object):
    _metric_prefixes = dict(Y=1e24, Z=1e21, E=1e18, P=1e15, T=1e12, G=1e9, M=1e6, k=1e3, h=1e2, da=1e1,
                            d=1e-1, c=1e-2, m=1e-3, u=1e-6, n=1e-9, p=1e-12, f=1e-15, a=1e-18, z=1e-21, y=1e-24)

    def __init__(self, base_unit, allow_metric=True, **conversions):
        self.base_unit = base_unit
        self.allow_metric = allow_metric
        self.conversions = dict()
        self.add_conversion(base_unit, 1.0)
        for unit, scale in conversions.items():
            self.add_conversion(unit, scale)

    def __call__(self, *args, **kwargs):
        return self.convert(*args, **kwargs)

    def add_conversion(self, unit, scale, add=0.0, metric_prefixes=None):
        metric_prefixes = self.allow_metric if metric_prefixes is None else metric_prefixes
        self.conversions[unit] = self._std_conversion_factory(scale, add)
        if metric_prefixes:
            for prefix, prescale in self._metric_prefixes.items():
                self.conversions['{}{}'.format(prefix, unit)] = self._std_conversion_factory(scale, add, prescale)

    def add_multiple_conversions(self, metric_prefixes=None, **conversions):
        for unit, scale in conversions.items():
            self.add_conversion(unit, scale, metric_prefixes=metric_prefixes)

    def add_conversion_lambda(self, unit, conv_from, conv_to, metric_prefixes=None):
        metric_prefixes = self.allow_metric if metric_prefixes is None else metric_prefixes
        self.conversions[unit] = self._lambda_conversion_factory(conv_from, conv_to)
        if metric_prefixes:
            for prefix, prescale in self._metric_prefixes.items():
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
        base_val = self.conversions[old_unit]['from'](value)
        return self.conversions[new_unit]['to'](base_val)

    def fixed_convert(self, new_unit):
        def conv(value, old_unit):
            return self.convert(value, old_unit, new_unit)
        return conv


time = UnitConverter('s')
time.add_multiple_conversions(metric_prefixes=False, min=60.0, hr=3600.0, day=24 * 3600.0, wk=7 * 24 * 3600.0)

mass = UnitConverter('g', lb=453.592, oz=28.3495, amu=1.660540199e-24, )

length = UnitConverter('m', **{'in': 0.0254, '"': 0.0254, 'ft': 0.3048, "'": 0.3048, 'yd': 0.9144, 'mi': 1609.344})

pressure = UnitConverter('Pa', bar=1e5, b=1e5, atm=1.01325e5)
pressure.add_conversion('mmHg', 133.322, metric_prefixes=False)

temperature = UnitConverter('C')
temperature.add_conversion('F', 5 / 9, 32)
temperature.add_conversion('K', 1., 273.15)

mixing_ratio = UnitConverter('mol/mol', **{'mol mol^-1': 1.0, 'mol mol-1': 1.0})
mixing_ratio.add_multiple_conversions(metric_prefixes=False, ppm=1e-6, ppmv=1e-6, ppb=1e-9, ppbv=1e-9, ppt=1e-12, pptv=1e-12)
