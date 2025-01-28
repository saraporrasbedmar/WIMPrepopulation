import yaml
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colorbar as colorbarr
import matplotlib.patches as mpatches

from scipy.optimize import newton
from scipy.integrate import simpson
from scipy.optimize import curve_fit

all_size = 24
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['axes.labelsize'] = all_size
plt.rcParams['lines.markersize'] = 10
plt.rc('font', size=all_size)
plt.rc('axes', titlesize=all_size)
plt.rc('axes', labelsize=all_size)
plt.rc('xtick', labelsize=all_size)
plt.rc('ytick', labelsize=all_size)
plt.rc('legend', fontsize=20)
plt.rc('figure', titlesize=all_size)
plt.rc('xtick', top=True, direction='in')
plt.rc('ytick', right=True, direction='in')
plt.rc('xtick.major', size=10, width=2, top=False, pad=10)
plt.rc('ytick.major', size=10, width=2, right=True, pad=10)
plt.rc('xtick.minor', size=7, width=1.5, top=False)
plt.rc('ytick.minor', size=7, width=1.5)

data_release_dmo = np.loadtxt(
    '../Data_subhalo_simulations/dmo_table.txt', skiprows=3)
data_release_hydro = np.loadtxt(
    '../Data_subhalo_simulations/hydro_table.txt', skiprows=3)

data_release_dmo = data_release_dmo[
                   data_release_dmo[:, 0] > 0.184, :]
data_release_hydro = data_release_hydro[
                     data_release_hydro[:, 0] > 0.184, :]

data_release_dmo = data_release_dmo[np.argsort(data_release_dmo[:, 1])]
data_release_hydro = data_release_hydro[np.argsort(data_release_hydro[:, 1])]


def read_config_file(ConfigFile):
    with open(ConfigFile, 'r') as stream:
        try:
            parsed_yaml = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return parsed_yaml


x_cumul = np.geomspace(1., 120., num=25)

path_input = '../Repopulation/input_files/input_paper2024.yml'

input_data = read_config_file(path_input)

print(sum(data_release_dmo[:, 1] > 50) / 6.)
print(sum(data_release_hydro[:, 1] > 20) / 6.)


def SHVF_Grand2012_int(V1, V2,
                       SHVF_bb,
                       SHVF_mm):
    return (np.rint(10 ** SHVF_bb
                       / (SHVF_mm + 1) *
                       (V2 ** (SHVF_mm + 1)
                        - V1 ** (SHVF_mm + 1))))


def xx(mmax, mmin, SHVF_bb, SHVF_mm, root):
    return SHVF_Grand2012_int(mmin, mmax, SHVF_bb, SHVF_mm) - root


num_max = float(input_data['repopulations']['num_subs_max'])
repop_factor = input_data['repopulations']['inc_factor']
print(repop_factor)

plt.figure()

colors = [input_data['repopulations']['inc_factor'], 2, 2.5]
bb = input_data['SHVF']['dmo']['bb']# + np.log10(6.)

for nn, repop_factor in enumerate(colors):
    mmin_array = []
    total_subs = []
    iii = 0

    m_min = float(input_data['SHVF']['RangeMin'])

    while m_min < input_data['SHVF']['RangeMax']:

        if (SHVF_Grand2012_int(
                m_min, m_min * repop_factor,
                bb,
                input_data['SHVF']['dmo']['mm'])
                > num_max):

            m_max = newton(
                xx, m_min,
                args=[m_min,
                      bb,
                      input_data['SHVF']['dmo']['mm'],
                      num_max])
            new_mmin = m_max

        else:
            m_max = np.minimum(
                m_min * repop_factor,
                input_data['SHVF']['RangeMax'])

            new_mmin = m_min * repop_factor
            print(m_min, m_max, SHVF_Grand2012_int(
                m_min, m_max,
                bb,
                input_data['SHVF']['dmo']['mm']))

            if (SHVF_Grand2012_int(
                m_max, np.minimum(
                        m_max * repop_factor,
                        input_data['SHVF']['RangeMax']),
                bb,
                input_data['SHVF']['dmo']['mm']) < 1.) and (
                    m_max <input_data['SHVF']['RangeMax']
            ):
                    m_max = input_data['SHVF']['RangeMax']
                    new_mmin = m_max
                    print('help')
                    print(m_min, m_max, SHVF_Grand2012_int(m_min,
                               m_max,
                               bb,
                               input_data['SHVF']['dmo']['mm']))
            mmin_array.append(m_min)
            total_subs.append(SHVF_Grand2012_int(m_min,
                               m_max,
                               bb,
                               input_data['SHVF']['dmo']['mm']))

        m_min = new_mmin
        iii += 1
        # print(m_min)
    print(repop_factor, iii)

    sum_subs = [sum(total_subs[i:]) for i in range(0, len(total_subs), 1)]
    print(total_subs)
    plt.scatter(mmin_array, sum_subs, label=repop_factor,
                color=plt.cm.CMRmap(nn / float(len(colors))))

    subs_grand = []
    for i in range(len(mmin_array)):
        subs_grand.append(sum(data_release_dmo[:, 1] > mmin_array[i]))
    plt.scatter(mmin_array, np.array(subs_grand)/6., marker='x',
                color=plt.cm.CMRmap(nn / float(len(colors))))

    subs_grand = []
    for i in range(len(mmin_array)):
        subs_grand.append(SHVF_Grand2012_int(mmin_array[i],
                               input_data['SHVF']['RangeMax'],
                               bb,
                               input_data['SHVF']['dmo']['mm']))
    plt.scatter(mmin_array, np.array(subs_grand), marker='+',
                color=plt.cm.CMRmap(nn / float(len(colors))),
                s=400)

print(iii)
plt.legend()
plt.xscale('log')
# plt.yscale('log')

plt.ylim(0, 50)
plt.xlim(10, 120)

plt.grid(which='both', axis='y')

plt.show()