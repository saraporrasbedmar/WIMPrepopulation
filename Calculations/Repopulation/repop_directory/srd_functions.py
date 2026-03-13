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

    return 10 ** params[0] * Vmax_array * params[1]


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

# -----------------------------------------------------------------------------
model_list = {
    'SHVF_Grand2012': _SHVF_Grand2012,
}


def srd_model(Vmax_array, SHVF_model=None,
               SHVF_params=None, verbose=True):

    if SHVF_model in model_list.keys():
        return model_list[SHVF_model](
            Vmax_array=Vmax_array, params=SHVF_params, verbose=verbose)
    else:
        return _SHVF_custom(
            Vmax_array=Vmax_array, SHVF_model=None,
            params=SHVF_params, verbose=verbose)
