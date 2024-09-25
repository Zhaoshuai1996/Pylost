# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 16:10:37 2020

@author: fraperri (from matplotlib)

https://stackoverflow.com/questions/45332056/decompose-a-float-into-mantissa-and-exponent-in-base-10-without-strings
https://docs.python.org/3/reference/lexical_analysis.html#formatted-string-literals
https://realpython.com/python-f-strings/
"""

# pylint: disable=C0103, C0115, C0116

from numpy import nan, pi


class SimpleUnit:
    def __init__(self, name, fullname=None, base=None, si=True):
        self.name = name
        if fullname is None:
            fullname = name
        self.fullname = fullname
        self.conversions = {self: 1.0}
        self.base_unit = None
        self.si = si

        if base is not None:
            base_unit, conversion = base
            self.base_unit = base[0]
            if callable(base[1]):
                self.add_conversion(*base)
            else:
                self.conversions[base_unit] = conversion
                self.base_unit.conversions[self] = 1.0 / conversion

    def __repr__(self):
        # return self.fullname
        return f'SimpleUnit({self.fullname})'

    def __str__(self):
        return self.name

    def __call__(self, *args):
        """
        SimpleUnit class can be called as function.

        Parameters
        ----------
        *args : value=float or numpy array, current=SimpleUnit, base_only=bool
            - value:     current value to be changed.
            - current:  current unit to be changed.
            - base_only: if False, look into other base (ex: rad to deg)
                         False, by default

        Returns
        -------
        float, numpy array or function

            if (value, unit, [base_only]):
                value updated in the calling unit from the current unit.

            If ():
                conversion to base unit (can be float or function).

            If (value):
                value converted to the base unit.

            If (unit):
                conversion from unit to calling unit (can be float or function).

        """
        if len(args) == 0:
            return self.conversions[self.base_unit]
        new = self
        base_only = False
        if len(args) == 1:
            if isinstance(args[0], SimpleUnit):
                value = 1.0
                current = args[0]
            else:
                value = args[0]
                current = self
                new = self.base_unit
        elif len(args) == 2:
            value, current = args
        else:
            value, current, base_only = args
        if current is None:
            return value
        if str(new) == str(current):
            return value
        for unit in current.conversions:
            if str(new) == str(unit):
                return SimpleUnit._convert(current.conversions[unit], value)
        if str(new.base_unit) == str(current.base_unit):
            value = SimpleUnit._convert(current(), value)
            value = SimpleUnit._convert(new.base_unit.conversions[new], value)
        elif not base_only and new.base_unit in current.conversions:
            value = SimpleUnit._convert(new(), value)
            value = SimpleUnit._convert(current.conversions[new.base_unit], value)
        else:
            print('no unit conversion possible')
            return nan
        return value

    @property
    def SI_unit(self):
        if self.base_unit is None:
            return self
        return self.base_unit

    def add_conversion(self, unit, conversion):
        self.conversions[unit] = conversion

    def get_conversion(self, unit):
        return self.conversions[unit]

    @staticmethod
    def _convert(conversion, value):
        if callable(conversion):
            return conversion(value)
        return conversion * value

    def to_string(self, value, fmt='.2f'):
        return f'{value:{fmt}} {self}'

    def auto(self, value, modulo=5):
        dico = {}
        base_unit = self
        if self.base_unit is not None:
            base_unit = self.base_unit
        for unit, conversion in base_unit.conversions.items():
            if callable(conversion):
                continue
            if not unit.si:
                continue
            dico[unit] = abs(int(f'{unit(value / modulo, self):e}'.split('e')[1]))
        best = min(dico, key=dico.get)
        return best(value, self), best

    def auto_str(self, value, fmt='.2f', modulo=5):
        auto = self.auto(value, modulo)
        return f'{auto[0]:{fmt}} {auto[1]}'

    @property
    def length(self):
        return m

    @property
    def curvature(self):
        return m_inv

    @property
    def angle(self):
        return rad


# class SciRep():
#     from numpy import isinf, isnan
#     """Extract the common logarithm representation from a float.
#         see decimal (_pydecimal.py)
#     """
#     def __init__(self, number):
#         self.number = number
#         self.mantissa, self.exponent = SciRep._from_float(number)
#         self.format = '-.2f'

#     def __repr__(self):
#         return f'{self.mantissa}e{self.exponent:02d}'

#     def __str__(self):
#         return f'{self.mantissa:{self.format}}e{self.exponent:02d}'

#     def __call__(self):
#         return self.exponent

#     def to_string(self, fmt=' .2f'):
#         return f'{self.mantissa:{fmt}}e{self.exponent:02d}'

#     @staticmethod
#     def _from_float(f):
#         """Converts a float to a decimal number, exactly."""
#         if isinstance(f, int): # handle integer inputs
#             k = 0
#             coeff = str(abs(f))
#         elif isinstance(f, float):
#             if isinf(f) or isnan(f):
#                 return f
#             n, d = abs(f).as_integer_ratio()
#             k = d.bit_length() - 1
#             coeff = str(n*5**k)
#         else:
#             raise TypeError("argument must be int or float.")
#         exponent = len(coeff) - (k+1)
#         mantissa = f/(10**exponent)
#         return mantissa, exponent


# XXX not complete

# ----lengths----
m = SimpleUnit('m', 'meters')
km = SimpleUnit('km', 'kilometers', (m, 1e3))
# cm = SimpleUnit('cm', 'centimeters', (m, 1e-2))
mm = SimpleUnit('mm', 'millimeters', (m, 1e-3))
um = SimpleUnit('um', 'micrometers', (m, 1e-6))
nm = SimpleUnit('nm', 'nanometers', (m, 1e-9))
A = SimpleUnit('A', 'angstroms', (m, 1e-10))
pm = SimpleUnit('pm', 'picometers', (m, 1e-12))

inch = SimpleUnit('inch', 'inches', (m, 0.0254), si=False)

# ----curvatures----
m_inv = SimpleUnit('m-1', '1/meters', si=False)

# ----angles----
rad = SimpleUnit('rad', 'radians')
mrad = SimpleUnit('mrad', 'milliradians', (rad, 1e-3))
urad = SimpleUnit('urad', 'microradians', (rad, 1e-6))
# nrad = SimpleUnit('nrad', 'nanoradians', (rad, 1e-9))

deg = SimpleUnit('deg', 'degrees', (rad, pi / 180.0), si=False)
asec = SimpleUnit('asec', 'arcseconds', (rad, pi / (180.0 * 3600.0)), si=False)
amin = SimpleUnit('asec', 'arcseconds', (rad, pi / (180.0 * 60.0)), si=False)

rad.add_conversion(deg, 180.0 / pi)
deg.add_conversion(asec, pi / 180.0)

# ----time based----
s = SimpleUnit('s', 'seconds')
ns = SimpleUnit('ns', 'nanoseconds', (s, 1e-9))
us = SimpleUnit('us', 'microseconds', (s, 1e-6))
ms = SimpleUnit('ms', 'milliseconds', (s, 1e-3))
mn = SimpleUnit('min', 'minutes', (s, 60.0))
h = SimpleUnit('h', 'hours', (s, 3600.0))
d = SimpleUnit('d', 'days', (s, 86400.0))

hz = SimpleUnit('hz', 'Hertz', (s, lambda x: 1.0 / x), si=False)
s.add_conversion(hz, lambda x: 1.0 / x)

# ----misc----
pix = SimpleUnit('pix', 'pixels', si=False)
fr = SimpleUnit('fr', 'fringes', si=False)
wav = SimpleUnit('wave', 'waves', si=False)
frad = SimpleUnit('frad', 'fringesradians', si=False)
unitless = SimpleUnit('unitless', 'unitless', si=False)

# ------------test------------
if __name__ == '__main__':
    print(mm(0.001316687050641363, m), mm)
    print(hz(0.001316687050641363, mn), hz)
    print(fr(0.001316687050641363, um))
    print(unitless.SI_unit)
    # print(deg(pi, rad))
    # print(mrad(90, amin))
    # print(hz(25, mn))
    # print(h(4e-6, hz))
    # print(s(2, nm))
    # print(urad(asec))
    # print(amin(deg))
    # print(urad(1))

    # test = um.auto(4200.204574846783)
    # print(*test)
    # print(test[1].to_string(test[0]))

    # print(h.auto_str(-.0004574846783))
