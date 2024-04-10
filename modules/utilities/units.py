




def get_unit(param):
    """
    return unit of param
    """
    unit_dict = {
        "Ux" : "m/s",
        "Uy" : "m/s",
        'nut' : r'$m^2/s$', 
        'nuTilda' : r'$m^2/s$', 
        'nutSource' : r'$m^2/s^2$', 
        'epsilonSource' : r'$m^2/s^4$', 
        'p' : r'$m^2/s^2$'
    }

    if param not in unit_dict.keys():
        unit = param
        #logger.warning(f"unit of {param} is not specified")
    else:
        unit = unit_dict[param]

    return unit
