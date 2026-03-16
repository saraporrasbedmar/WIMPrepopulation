import numpy as np
from scipy.integrate import simpson


def power_law(Vmax, V0, slope):
    """
    SubHalo Velocity Function (SHVF) - number of subhalos as a
    function of Vmax. Power law formula.
    Definition taken from Grand 2012.07846.

    :param Vmax_array: float or array-like [km/s]
        Maximum radial velocity of a bound particle in the subhalo.

    :return: float or array-like
        Number of subhalos defined by the Vmax input.
    """
    return 10 ** V0 * Vmax ** slope


def SHVF_Grand2012_int(V1, V2, SHVF_bb, SHVF_mm):
    """
    Integration of the SHVF defined above.

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
    'SHVF_Grand2012': power_law,
}

def SHVF_model(Vmax_array, SHVF_model=None,
               SHVF_params=None):

    if SHVF_model in model_list.keys():
        return model_list[SHVF_model](
            Vmax=Vmax_array, V0=SHVF_params[0], slope=SHVF_params[1])


def SHVF_model_integral(
        Vmax_min, Vmax_max,
        SHVF_model_int=None, SHVF_params_int=None):

    if SHVF_model_int in model_list.keys():
        vmax_array = np.geomspace(Vmax_min, Vmax_max, num=150)

        yy = SHVF_model(vmax_array, SHVF_model=SHVF_model_int,
                        SHVF_params=SHVF_params_int)

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
print()
