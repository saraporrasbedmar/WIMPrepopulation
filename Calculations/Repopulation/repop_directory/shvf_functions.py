import numpy as np
from scipy.integrate import simpson


def _SHVF_Grand2012(Vmax_array, params, verbose=True):
    """
    SubHalo Velocity Function (SHVF) - number of subhalos as a
    function of Vmax. Power law formula.
    Definition taken from Grand 2012.07846.

    :param Vmax_array: float or array-like [km/s]
        Maximum radial velocity of a bound particle in the subhalo.

    :return: float or array-like
        Number of subhalos defined by the Vmax input.
    """
    if params is None:
        params = [5., -4.]
        if verbose:
            print('   -> SHVF or SHMF: default parameters chosen: ',
                  params)

    return 10 ** params[0] * Vmax_array ** params[1]


def _SHVF_custom(Vmax_array, SHVF_model, params=None, verbose=True):

    if type(SHVF_model) == str:
        try:
            return eval(SHVF_model, {'zz': Vmax_array, 'ci': params})
        except NameError:
            raise NameError(
                'Unrecognized type of metallicity dependency.' + '\n'
                + 'Implemented models: ' + '\n'
                + str([*model_list]) + '\n'
                + 'If the string is a expression to evaluate,' + '\n'
                + 'there is something wrong in it, check it.' + '\n'
                + 'Independent variable must be called \'zz\' ' + '\n'
                + 'and parameters be an array or list called \'ci\'.' + '\n'
                + 'Inputs given:' + '\n'
                + '\'metall_model\': ' + str(SHVF_model) + '\n'
                + '\'ci\': ' + str(params)
            )

    elif callable(SHVF_model):
        if params is None:
            return SHVF_model(Vmax_array)
        else:
            return SHVF_model(Vmax_array, params)

    else:
        raise ValueError('No SHVF or SHMF model chosen.\n'
                   + 'Accepted models: ' + str(model_list.keys()) + '\n'
                   + 'Model not recognized: ' + SHVF_model)
        # raise ValueError(
        #     'Unrecognized type of metallicity dependency.' + '\n'
        #     + 'Implemented models: ' + '\n'
        #     + str([*model_list]) + '\n'
        #     + 'If the string is a expression to evaluate,' + '\n'
        #     + 'there is something wrong in it, check it.' + '\n'
        #     + 'Independent variable must be called \'zz\' ' + '\n'
        #     + 'and parameters be an array or list called \'ci\'.' + '\n'
        #     + 'Inputs given:' + '\n'
        #     + '\'metall_model\': ' + str(SHVF_model) + '\n'
        #     + '\'ci\': ' + str(paramstosave)
        # )



def SHVF_Grand2012_int(V1, V2, SHVF_bb, SHVF_mm):
    """
    Integration of the SHVF defined above.
    (numba needed an analytical expression.)

    :param V1: float or array-like [km/s]
        Minimum limit of integration of the SHVF.
    :param V2: float or array-like [km/s]
        Maximum limit of integration of the SHVF.

    :return: float or array-like
        Integrated SHVF.
    """
    return int(np.rint(10 ** SHVF_bb
                       / (SHVF_mm + 1) *
                       (V2 ** (SHVF_mm + 1)
                        - V1 ** (SHVF_mm + 1))))


# -----------------------------------------------------------------------------
model_list = {
    'SHVF_Grand2012': _SHVF_Grand2012,
}


def SHVF_model(Vmax_array, SHVF_model=None,
               SHVF_params=None, verbose=True):

    if SHVF_model in model_list.keys():
        return model_list[SHVF_model](
            Vmax_array=Vmax_array, params=SHVF_params, verbose=verbose)
    else:
        return _SHVF_custom(
            Vmax_array=Vmax_array, SHVF_model=None,
            params=SHVF_params, verbose=verbose)


def SHVF_model_integral(
        Vmax_min, Vmax_max,
        SHVF_model_int=None, SHVF_params_int=None, verbose_int=True):

    if SHVF_model_int in model_list.keys():
        vmax_array = np.geomspace(Vmax_min, Vmax_max, num=150)

        yy = SHVF_model(vmax_array, SHVF_model=SHVF_model_int,
                        SHVF_params=SHVF_params_int, verbose=verbose_int)

        return int(np.rint(simpson(y=yy * np.log(10) * vmax_array,
                                   x=np.log10(vmax_array))))
    else:
        ValueError('No SHVF or SHMF model chosen.\n'
                   + 'Accepted models: ' + str(model_list.keys()) + '\n'
                   + 'Model not recognized: ' + SHVF_model_int)

print()
print(SHVF_model_integral(0.01, 120, SHVF_model_int='SHVF_Grand2012',
                          SHVF_params_int=[5.68, -3.92]))
print(SHVF_Grand2012_int(0.01, 120, 5.68, -3.92))
print(SHVF_model_integral(0.01, 120, SHVF_model_int='SHVF_Grand2012',
                          SHVF_params_int=[5.68, -3.92]) -
SHVF_Grand2012_int(0.01, 120, 5.68, -3.92))
print((SHVF_model_integral(0.01, 120, SHVF_model_int='SHVF_Grand2012',
                          SHVF_params_int=[5.68, -3.92]) -
SHVF_Grand2012_int(0.01, 120, 5.68, -3.92))/SHVF_Grand2012_int(0.01, 120, 5.68, -3.92))

print((SHVF_model_integral(0.1, 120, SHVF_model_int='SHVF_Grand2012',
                          SHVF_params_int=[5.68, -3.92]) -
SHVF_Grand2012_int(0.1, 120, 5.68, -3.92))/SHVF_Grand2012_int(0.1, 120, 5.68, -3.92))

print((SHVF_model_integral(1, 120, SHVF_model_int='SHVF_Grand2012',
                          SHVF_params_int=[5.68, -3.92]) -
SHVF_Grand2012_int(1, 120, 5.68, -3.92))/SHVF_Grand2012_int(1, 120, 5.68, -3.92))

print(SHVF_model_integral(1, 120, SHVF_model_int='SHVF_Grand2012',
                          SHVF_params_int=[7.78, -3.92]) ,
SHVF_Grand2012_int(1, 120, 7.78, -3.92))