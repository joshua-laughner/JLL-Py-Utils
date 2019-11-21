def pot_temp(pres, temp):
    """
    Calculate potential temperature

    :param pres: pressure in hPa
    :param temp: temperature in K
    :return: potential temperature in K
    """
    return temp * (1000/pres) ** 0.286
