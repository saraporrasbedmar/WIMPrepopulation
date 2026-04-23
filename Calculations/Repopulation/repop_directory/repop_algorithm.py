import os
import yaml
import copy
import time
import h5py
import psutil
import inspect

import numpy as np

from scipy.optimize import newton, minimize
from scipy.interpolate import UnivariateSpline
from scipy.integrate import cumulative_simpson, quad

from astropy import units as u
from astropy import constants as c


def memory_usage_psutil():
    # return the memory usage in MB
    process = psutil.Process(os.getpid())
    mem = process.memory_info()[0] / float(10 ** 6)
    return mem


def read_config_file(ConfigFile):
    with open(ConfigFile, 'r') as stream:
        try:
            parsed_yaml = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return parsed_yaml


mass_energy2 = [
    (u.si.kg**2, u.si.J**2.,
     lambda x: (x * c.c.value**4), lambda x: (x / c.c.value**4)),
    (u.kg**2. / u.m**5, u.J**2. / u.m**5,
     lambda x: x * c.c**4, lambda x: x / c.c**4)
]


class RepopAlgorithm:
    def __init__(self, path_input):

        self.path_output = None
        self.configuration = None

        if isinstance(path_input, str):
            self.input_dict = read_config_file(path_input)
        else:
            self.input_dict = path_input

        self.input_dict_strings = copy.deepcopy(self.input_dict)

        try:
            self.rng = np.random.default_rng(
                seed=self.input_dict['repopulations']['rng_seed'])

            print('Warning: seed for np.random has been explicitly set'
                  ' to: seed='
                  + str(self.input_dict['repopulations']['rng_seed'])
                  + '. Only use this setting for testing purposes.')
        except ValueError:
            self.rng = np.random.default_rng(seed=None)

        self._num_subs_max = int(5e5)

        self._number_highest = int(
            self.input_dict['repopulations']['number_highest'])

        self._its = self.input_dict['repopulations']['number_iterations']

        try:
            self._prntfrq = int(
                self.input_dict['repopulations']['print_frequency'])
            if self._prntfrq <= 0:
                self._prntfrq = None
        except ValueError:
            self._prntfrq = None

        self.RangeMin = self.input_dict['repopulations']['RangeMin']
        self.RangeMax = self.input_dict['repopulations']['RangeMax']

        for key, value in self.input_dict['cosmo_constants'].items():
            self.input_dict['cosmo_constants'][key] = (
                self.input_dict['cosmo_constants'][key]['value']
                * u.Unit(self.input_dict['cosmo_constants'][key]['unit']
                         )
            )

        for key, value in self.input_dict['host'].items():
            if 'unit' in self.input_dict['host'][key]:
                self.input_dict['host'][key] = (
                    self.input_dict['host'][key]['value']
                    * u.Unit(self.input_dict['host'][key]['unit']))

        self._m_min = None
        self._m_max = None

        self.subhalo_data = {}

    def run(self, path_output, configuration=None):

        if configuration is None:
            config_list = self.input_dict['configurations'].keys()
        elif isinstance(configuration, str):
            config_list = [configuration]
        elif isinstance(configuration, list):
            config_list = configuration
        else:
            raise TypeError(
                'Configuration type not accepted.\n'
                + 'type: ' + type(configuration) + '\n'
                + 'configuration value: ' + configuration)

        self.path_output = path_output + '/'
        print(path_output)

        if not os.path.exists(path_output + '/'):
            os.makedirs(path_output + '/')

        for ii in config_list:
            self.configuration = ii
            aa = self.SHVF_integral(
                self.RangeMin, self.RangeMax, force_no_fraction=True)
            print()
            print(self.configuration)
            print(self.RangeMin, self.RangeMax)
            print('    Max number of repop subhalos: %i' % aa)

            if self._number_highest > aa:
                print(
                    'Warning: number of requested subhalos to save is\n'
                    'higher than the total subhalos that are created.\n'
                    'Therefore, all subhalos will be saved.\n'
                    'Change this parameter if needed in:\n'
                    "input_dict['repopulations']['number_highest']")

            if self.input_dict['repopulations']['save_full_repop']:
                self.interior_full_repop()
            else:
                self.interior_brightest()

        # Save input data in a file in the outputs directory
        file_inputs = open(self.path_output + 'input_data.yml', 'w')
        aaa = copy.deepcopy(self.input_dict_strings)
        aaa = self.change_callables_into_strings(aaa)
        yaml.dump(aaa, file_inputs,
                  default_flow_style=False, allow_unicode=True)
        file_inputs.close()
        return

    def change_callables_into_strings(self, data_dict):
        if isinstance(data_dict, dict):
            for key, value in data_dict.items():
                if callable(value):
                    data_dict[key] = inspect.getsource(value).strip()
                elif isinstance(value, dict):
                    self.change_callables_into_strings(value)
                elif isinstance(value, list):
                    for idx, item in enumerate(value):
                        if callable(item):
                            value[idx] = inspect.getsource(item).strip()
                        elif isinstance(item, dict):
                            self.change_callables_into_strings(item)

        elif isinstance(data_dict, list):
            for idx, item in enumerate(data_dict):
                if callable(item):
                    data_dict[idx] = inspect.getsource(item).strip()
                elif isinstance(item, dict):
                    self.change_callables_into_strings(item)
        return data_dict

    def calculate_formula(self, xx, formula, params=None):

        if isinstance(formula, str):

            if hasattr(self, formula):
                return getattr(self, formula)(xx, **params)

            bb = {}

            if isinstance(params, float) or isinstance(params, int):
                bb['params'] = params
            elif isinstance(params, list):
                bb['params'] = np.array(params, dtype=float)

            if isinstance(xx, list):
                xx = np.array(xx)
            bb['xx'] = xx

            # Evaluate formula
            try:
                aa = eval(formula, {}, bb)
            except Exception as e:
                raise ValueError(
                    f'Error evaluating formula ' + formula
                    + f' with parameters {bb}: {e}')
            return aa

        elif callable(formula):
            if params is not None:
                if isinstance(params, dict):
                    return formula(xx, **params)
                return formula(xx, params)
            else:
                return formula(xx)

    def get_parameter(self, name, parametrization=None):
        """
        Retrieve parameter, computing if necessary.
        Uses values to avoid recomputation.
        """
        if name in self.subhalo_data:
            return self.subhalo_data[name]

        if hasattr(self, name):
            if isinstance(parametrization, dict):
                self.subhalo_data[name] = getattr(self, name)(
                    **parametrization)
                return self.subhalo_data[name]
            self.subhalo_data[name] = getattr(self, name)()
            return self.subhalo_data[name]

        formula = parametrization.get('formula', None)

        if isinstance(formula, str):

            if hasattr(self, formula):
                if isinstance(parametrization, dict):
                    self.subhalo_data[name] = getattr(self, formula)(
                        **parametrization['params'])
                    return self.subhalo_data[name]
                self.subhalo_data[name] = getattr(self, formula)()
                return self.subhalo_data[name]

            bb = {}

            params = parametrization.get('params', None)
            if isinstance(params, float) or isinstance(params, int):
                bb['params'] = params
            elif isinstance(params, list):
                bb['params'] = np.array(params, dtype=float)

            variables = parametrization.get('variables', None)
            if isinstance(variables, str):
                bb[variables] = self.get_parameter(variables)
            elif isinstance(variables, list):
                for var in variables:
                    bb[var] = self.get_parameter(var)

            # Evaluate formula
            try:
                aa = eval(parametrization['formula'], {}, bb)
            except Exception as e:
                raise ValueError(
                    f'Error evaluating formula '
                    + parametrization['formula']
                    + f' with parameters {bb}: {e}')

            self.subhalo_data[name] = aa
            return self.subhalo_data[name]

        elif callable(formula):

            vars_for_func = {}
            variables = parametrization.get('variables', [])
            if isinstance(variables, str):
                variables = [variables]
            for var in variables:
                vars_for_func[var] = self.get_parameter(var)

            params = parametrization.get('params', None)
            if params is not None:
                vars_for_func['params'] = params

            self.subhalo_data[name] = formula(**vars_for_func)
            return self.subhalo_data[name]

    def calculate_characteristics_subhalo(
            self, Vmax=None, D_GC=None, position_Earth=None):

        self.subhalo_data = {}

        if Vmax is None:
            try:
                self._m_max = np.min((
                    newton(
                        self.xx, self._m_min * 1.05,
                        args=[self._m_min, self._num_subs_max]),
                    self.RangeMax
                ))
            except:
                self._m_max = self.RangeMax

            self._num_subhalos = self.SHVF_integral(
                Vmax_min=self._m_min, Vmax_max=self._m_max,
                force_no_fraction=False)

            print(self._m_min, self._m_max, self._num_subhalos)

            # Montecarlo algorithm for creating of subhalo Vmax
            x = np.geomspace(self._m_min, self._m_max, num=2000)
            y = self.calculate_formula(
                x,
                self.input_dict['configurations'][
                    self.configuration]['SHVF']['formula'],
                self.input_dict['configurations'][
                    self.configuration]['SHVF']['params']
            )

            cumul = cumulative_simpson(
                y=y * x * np.log(10), x=np.log10(x), initial=0)
            cumul /= cumul[-1]
            spline = UnivariateSpline(
                cumul, x, s=0, k=1, ext=1)

            self.subhalo_data['Vmax'] = (
                    spline(self.rng.random(self._num_subhalos))
                    * u.Unit(
                self.input_dict['repopulations'][
                    'params_to_save']['Vmax']['unit']))
        else:
            self.subhalo_data['Vmax'] = Vmax

        if D_GC is None:
            # Montecarlo algorithm for creating of subhalo D_GC
            x = np.linspace(
                0. * u.kpc,
                (self.input_dict['host']['R_vir'].to(
                    u.Unit(self.input_dict
                           ['repopulations']['params_to_save']
                           ['D_GC']['unit']))),
                num=2000)

            if (self.input_dict['repopulations']['use_spherical_shells']
                    and self._Rcut < 8.5):
                x = np.linspace(
                    8.5 - self._Rcut, 8.5 + self._Rcut,
                    num=3000)

            y = self.calculate_formula(
                x,
                self.input_dict['configurations'][
                    self.configuration]['SRD']['formula'],
                self.input_dict['configurations'][
                    self.configuration]['SRD']['params']
            )

            cumul = cumulative_simpson(y=y, x=x, initial=0)
            cumul /= cumul[-1]
            x_min = ((np.array(cumul) - 1e-8) < 0).argmin() - 1
            spline = UnivariateSpline(
                cumul[x_min:], x[x_min:], s=0, k=1, ext=1)

            self.subhalo_data['D_GC'] = (
                    spline(self.rng.random(self._num_subhalos))
                    * u.Unit(self.input_dict
                             ['repopulations']['params_to_save']
                             ['D_GC']['unit']))
        else:
            self.subhalo_data['D_GC'] = D_GC

        if position_Earth is None:
            position_Earth = self.input_dict['host']['position_Earth']

        # Random distribution of subhalos around the celestial sphere
        self.subhalo_data['galactocentric_theta'] = self.rng.uniform(
            0, 2 * np.pi, len(self.subhalo_data['Vmax'])) * u.rad

        self.subhalo_data['galactocentric_phi'] = np.arccos(
            2 * self.rng.uniform(0, 1, len(self.subhalo_data['Vmax']))
            - 1) * u.rad

        # Positions of the subhalos
        self.subhalo_data['galactocentric_X'] = (
                self.subhalo_data['D_GC']
                * np.cos(self.subhalo_data['galactocentric_theta'])
                * np.sin(self.subhalo_data['galactocentric_phi']))
        self.subhalo_data['galactocentric_Y'] = (
                self.subhalo_data['D_GC']
                * np.sin(self.subhalo_data['galactocentric_theta'])
                * np.sin(self.subhalo_data['galactocentric_phi']))
        self.subhalo_data['galactocentric_Z'] = (
                self.subhalo_data['D_GC']
                * np.cos(self.subhalo_data['galactocentric_phi']))

        self.subhalo_data['D_Earth'] = ((
            self.subhalo_data['galactocentric_X']
            - position_Earth[0]) ** 2
            + (self.subhalo_data['galactocentric_Y']
               - position_Earth[1]) ** 2
            + (self.subhalo_data['galactocentric_Z']
               - position_Earth[2]) ** 2
            ) ** 0.5

        if 'Cv' in self.input_dict[
            'configurations'][self.configuration].keys():
            self.get_parameter('Cv',
                parametrization=self.input_dict['configurations'][
                    self.configuration]['Cv'])
            # self.get_parameter('C200_from_Cv', None)

        for pn in self.input_dict['repopulations']['params_to_save']:
            self.get_parameter(
                pn,
                self.input_dict['repopulations']['params_to_save'][pn])

        if self.input_dict['configurations'][
                    self.configuration]['use_Roche']:
            self.subhalo_data['survives_Roche'] = (
                    self.get_parameter('R_t')
                    > self.get_parameter('R_s')
            )

        self.subhalo_data['engulfs_Earth'] = (
            self.get_parameter('R_s') > self.get_parameter('D_Earth'))

        return

    def xx(self, mmax, mmin, root):
        return (self.SHVF_integral(
            Vmax_min=np.max((mmin, 1e-20)),
            Vmax_max=mmax, force_no_fraction=True) - root)

    def store_dict_as_hdf(self, group, d):

        for key, value in d.items():
            if value is None:
                group.create_group(key)
                continue

            # Case 1: nested dict
            if isinstance(value, dict):
                subgroup = group.create_group(key)
                self.store_dict_as_hdf(subgroup, value)
                continue

            # Case 2: callable: convert to string
            if callable(value):
                value = inspect.getsource(value).strip()

            # Case 3: list of callables or mixed lists
            if isinstance(value, list):
                # create subgroup and save each element
                list_group = group.create_group(key)
                for i, element in enumerate(value):
                    item_name = f"item_{i}"

                    if callable(element):
                        element = inspect.getsource(element).strip()

                    if isinstance(element, dict):
                        item_group = list_group.create_group(item_name)
                        self.store_dict_as_hdf(item_group, element)
                    else:
                        if isinstance(element, str):
                            data = np.string_(element)
                            dtype = h5py.string_dtype(encoding="utf-8")
                        else:
                            data = element
                            dtype = None
                        list_group.create_dataset(item_name, data=data,
                                                  dtype=dtype)
                continue

            # Case 4: primitive value
            if isinstance(value, str):
                data = np.string_(value)
                dtype = h5py.string_dtype("utf-8")
            else:
                data = value
                dtype = None

            if key in group:
                del group[key]
            group.create_dataset(key, data=data, dtype=dtype)

    def interior_full_repop(self):

        with h5py.File(self.path_output + 'fullrepop_'
                       + self.configuration + '.h5', 'a') as f:
            input_group = f.create_group(f'inputs')
            aaa = copy.deepcopy(self.input_dict_strings)
            aaa['configurations'] = aaa[
                'configurations'][self.configuration]
            self.store_dict_as_hdf(input_group, aaa)

            for iter_idx in range(self._its):

                if iter_idx % self._prntfrq == 0:
                    print('    %s %s: it %d' % (
                        time.strftime(
                            ' %Y-%m-%d %H:%M:%S', time.gmtime()),
                        self.configuration, iter_idx))
                    progress = open(self.path_output + 'progress_'
                                    + self.configuration + '.txt', 'a')
                    progress.write(
                        self.configuration
                        + ', iteration ' + str(iter_idx))
                    progress.write(
                        '        %.3f  %s\n' %
                        (memory_usage_psutil(),
                         time.strftime(
                             ' %Y-%m-%d %H:%M:%S', time.gmtime())))
                    progress.close()

                iter_group = f.create_group(f'iteration_{iter_idx}')
                datasets = {}

                # We calculate our subhalo population in bins
                self._m_min = self.RangeMin

                while self._m_min < self.RangeMax:

                    self.calculate_characteristics_subhalo()

                    for key, array in self.subhalo_data.items():
                        if key not in iter_group.keys():
                            datasets[key] = iter_group.create_dataset(
                                key,
                                shape=(0,),
                                maxshape=(None,),
                                dtype=array.dtype,
                                chunks=True,
                                compression='gzip'
                            )
                            try:
                                datasets[key].attrs['units'] = str(
                                    self.subhalo_data[key].unit)
                            except AttributeError:
                                datasets[key].attrs['units'] = ''

                        dataset = datasets[key]
                        current_size = dataset.shape[0]
                        new_size = current_size + len(array)
                        dataset.resize((new_size,))
                        dataset[current_size:new_size] = array
                    self._m_min = self._m_max
            f.flush()
        return

    def interior_brightest(self):

        with h5py.File(self.path_output + 'brightest_'
                       + self.configuration + '.h5', 'a') as f:
            input_group = f.create_group(f'inputs')
            aaa = copy.deepcopy(self.input_dict_strings)
            aaa['configurations'] = aaa[
                'configurations'][self.configuration]
            self.store_dict_as_hdf(input_group, aaa)

            datasets = {}

            for iter_idx in range(self._its):

                if iter_idx % self._prntfrq == 0:
                    print('    %s %s: it %d' % (
                        time.strftime(
                            ' %Y-%m-%d %H:%M:%S', time.gmtime()),
                        self.configuration, iter_idx))
                    progress = open(self.path_output + 'progress_'
                                    + self.configuration + '.txt', 'a')
                    progress.write(
                        self.configuration
                        + ', iteration ' + str(iter_idx))
                    progress.write(
                        '        %.3f  %s\n' %
                        (memory_usage_psutil(),
                         time.strftime(
                             ' %Y-%m-%d %H:%M:%S', time.gmtime())))
                    progress.close()

                iter_group = f.create_group(f'iteration_{iter_idx}')

                highest_dict = {}
                for ii in self.input_dict[
                    'repopulations']['params_to_order_by']:
                    highest_dict[ii] = {}

                # We calculate our subhalo population in bins
                self._m_min = self.RangeMin

                while self._m_min < self.RangeMax:

                    self.calculate_characteristics_subhalo()

                    for ii in self.input_dict[
                        'repopulations']['params_to_order_by']:
                        data_bright = self.get_parameter(ii)

                        if not self.input_dict[
                        'repopulations']['allow_break_Roche']:
                            if ('survives_Roche'
                                    not in self.subhalo_data.keys()):
                                self.subhalo_data['survives_Roche'] = (
                                    self.get_parameter('R_t')
                                    > self.get_parameter('R_s')
                                )
                            data_bright *= self.get_parameter(
                                'survives_Roche')

                        if not self.input_dict[
                            'repopulations']['allow_engulf_Earth']:
                            self.get_parameter('engulfs_Earth')
                            data_bright *= ~self.get_parameter(
                                'engulfs_Earth')

                        if self._number_highest < self._num_subhalos:
                            temp = np.argpartition(
                                -data_bright, self._number_highest)
                            highest_indexes = temp[:self._number_highest]

                            for key, array in self.subhalo_data.items():
                                if key not in highest_dict[ii].keys():
                                    highest_dict[ii][key] = (
                                        array[highest_indexes].copy()
                                    )
                                else:
                                    highest_dict[ii][key] = np.append(
                                        highest_dict[ii][key],
                                        array[highest_indexes])
                        else:

                            for key, array in self.subhalo_data.items():
                                if key not in highest_dict[ii].keys():
                                    highest_dict[ii][key] = (
                                        array.copy()
                                    )
                                else:
                                    highest_dict[ii][key] = np.append(
                                        highest_dict[ii][key], array)

                    self._m_min = self._m_max

                for ii in self.input_dict[
                    'repopulations']['params_to_order_by']:

                    highest_group = iter_group.create_group(
                        f'highest_' + str(ii))

                    if len(highest_dict[ii][ii]) > self._number_highest:
                        temp = np.argpartition(
                            -highest_dict[ii][ii], self._number_highest)
                        highest_indexes = temp[:self._number_highest]

                        for key, array in highest_dict[ii].items():
                            datasets[key] = (
                                    highest_group.create_dataset(
                                        key,
                                        data=array[highest_indexes],
                                        chunks=True,
                                        compression='gzip'
                                    ))
                            try:
                                datasets[key].attrs['units'] = str(
                                    self.subhalo_data[key].unit)
                            except AttributeError:
                                datasets[key].attrs['units'] = ''

                    else:
                        for key, array in highest_dict[ii].items():
                            datasets[key] = (
                                    highest_group.create_dataset(
                                        key,
                                        data=array,
                                        chunks=True,
                                        compression='gzip'
                                    ))
                            try:
                                datasets[key].attrs['units'] = str(
                                    self.subhalo_data[key].unit)
                            except AttributeError:
                                datasets[key].attrs['units'] = ''
                print('Memory in use: %.1f MB' % memory_usage_psutil())
            f.flush()
        return

    # ----------- General formulas -------------------------------------
    def Rmax(self, Vmax=None, Cv=None, cosmo_H_0=None, unit=None):
        """
        Calculate Rmax of a subhalo.

        :param V: float or array-like [km/s]
            Maximum circular velocity inside a subhalo.
        :param C: float or array-like
            Subhalo concentration.

        :return: float or array-like [kpc]
            Rmax of the subhalo given by the inputs.
        """
        if Vmax is None:
            Vmax = self.get_parameter('Vmax')
        if Cv is None:
            Cv = self.get_parameter('Cv')
        if cosmo_H_0 is None:
            cosmo_H_0 = self.input_dict['cosmo_constants']['H_0']
        if unit is None:
            try:
                unit = self.input_dict['repopulations'][
                    'params_to_save']['Rmax']['unit']
            except KeyError:
                unit = 'kpc'

        return (Vmax / cosmo_H_0 * np.sqrt(2. / Cv)).to(u.Unit(unit))

    def R_s(self, Vmax=None, Cv=None, cosmo_H_0=None,
            density_profile=None, unit=None):
        """
        Calculate scale radius (R_s) of a subhalo following the NFW
        analytical expression for a subhalo density profile.

        :param V: float or array-like [km/s]
            Maximum circular velocity inside a subhalo.
        :param C: float or array-like
            Subhalo concentration.

        :return: float or array-like [kpc]
            R_s of the subhalo given by the inputs.
        """
        if Vmax is None:
            Vmax = self.get_parameter('Vmax')
        if Cv is None:
            Cv = self.get_parameter('Cv')
        if cosmo_H_0 is None:
            cosmo_H_0 = self.input_dict['cosmo_constants']['H_0']
        if unit is None:
            try:
                unit = self.input_dict['repopulations'][
                    'params_to_save']['R_s']['unit']
            except KeyError:
                unit = 'kpc'

        RmaxoverrS = self.RmaxoverrS(density_profile)

        return (self.Rmax(Vmax, Cv, cosmo_H_0) / RmaxoverrS
                ).to(u.Unit(unit))

    def R_t(self, Vmax=None, Cv=None, D_GC=None,
            cosmo_H_0=None, cosmo_G=None,
            host_rho_0=None, host_r_s=None,
            density_profile_sub=None, density_profile_host=None,
            unit=None):
        """
        Calculation of tidal radius (R_t) of a subhalo, following the
        NFW analytical expression for a subhalo density profile.

        Definition of R_t: 1603.04057 King radius pg 14

        :param V: float or array-like [km/s]
            Maximum circular velocity inside a subhalo.
        :param C: float or array-like
            Subhalo concentration.
        :param D_GC: float or array-like [kpc]
            Distance from the center of the subhalo to the
            Galactic Center (GC).

        :return: float or array-like [kpc]
            Tidal radius of the subhalo given by the inputs.
        """
        if Vmax is None:
            Vmax = self.get_parameter('Vmax')
        if Cv is None:
            Cv = self.get_parameter('Cv')
        if D_GC is None:
            D_GC = self.get_parameter('D_GC')
        if cosmo_H_0 is None:
            cosmo_H_0 = self.input_dict['cosmo_constants']['H_0']
        if cosmo_G is None:
            cosmo_G = self.input_dict['cosmo_constants']['G']
        if density_profile_host is None:
            density_profile_host = self.input_dict['host'][
                'density_profile']
        if host_rho_0 is None:
            host_rho_0 = self.input_dict['host']['rho_0']
        if host_r_s is None:
            host_r_s = self.input_dict['host']['r_s']
        if unit is None:
            try:
                unit = self.input_dict['repopulations'][
                    'params_to_save']['R_t']['unit']
            except KeyError:
                unit = 'kpc'

        Rmax = self.Rmax(Vmax=Vmax, Cv=Cv, cosmo_H_0=cosmo_H_0)
        c200 = self.get_parameter('C200_from_Cv')
        M_subhalo = self.mass_from_Vmax(
            c200, Vmax, Rmax, cosmo_G, density_profile_sub)
        M_host = self.M_encapsulated(
            D_GC, host_rho_0, host_r_s, density_profile_host)

        return (D_GC * (M_subhalo / (3 * M_host)) ** (1/3.)
                ).to(u.Unit(unit))

    def M_encapsulated(self, radius, rho_0, r_s, density_profile,
                       unit=None):
        # TODO que meto aqui como defaults
        """
        Mass encapsulated up to a certain radius.
        We assume a known density profile for the (sub)halo.

        :param R: float or array-like [kpc]
            Radius up to which we integrate the density profile.

        :return: float or array-like [Msun]
            Host mass encapsulated up to R.
        """
        # if radius is None:
        #     radius = self.get_parameter('D_GC')
        # if rho_0 is None:
        #     rho_0 = self.input_dict['host']['rho_0']
        # if r_s is None:
        #     r_s = self.input_dict['host']['r_s']
        if unit is None:
            try:
                unit = self.input_dict['repopulations'][
                    'params_to_save']['M_encapsulated']['unit']
            except KeyError:
                unit = 'Msun'

        return (4 * np.pi * rho_0 * r_s ** 3
                * self.ff(
                    radius / r_s, density_profile=density_profile)
                ).to(u.Unit(unit))

    def C200_from_Cv(self, Cv=None, density_profile=None):
        """
        Formula to find c200 knowing Cv to input in the Newton
        root-finding method.

        :param c200: float or array-like
            c200 of subhalo (concentration definition)
        :param Cv: float or array-like
            Cv of subhalo (concentration definition)

        :return: float or array-like
            The output will be 0 when you find the c200 for a
            specific Cv.
        """
        if Cv is None:
            Cv = self.get_parameter('Cv')

        RmaxoverrS = self.RmaxoverrS(density_profile)

        def int_interior(c200i, Cvi):
            return (200 * c200i ** 3 / self.ff(c200i)
                    * self.ff(RmaxoverrS) / RmaxoverrS ** 3
                    - Cvi)

        if isinstance(Cv, float):
            c200 = newton(int_interior, x0=40.0, args=[Cv])
        else:
            c200 = np.array([
                newton(int_interior, x0=40.0, args=[i.value])
                for i in Cv])

        return c200 * u.dimensionless_unscaled

    def mass_from_Vmax(self, radius_normalized, Vmax=None, Rmax=None,
                       cosmo_G=None, density_profile=None,
                       unit=None):
        """
        Mass from a subhalo assuming a NFW profile.
        Theoretical steps in Moline16.

        :param Vmax: float or array-like [km/s]
            Maximum radial velocity of a bound particle in the subhalo.
        :param Rmax: float or array-like [kpc]
            Radius at which Vmax happens (from the subhalo center).
        :param c200: float or array-like
            Concentration of the subhalo in terms of mass.
        :return: float or array-like [Msun]
            Mass from the subhalo assuming a NFW profile.
        """
        if Vmax is None:
            Vmax = self.get_parameter('Vmax')
        if Rmax is None:
            Rmax = self.get_parameter('Rmax')
        if cosmo_G is None:
            cosmo_G = self.input_dict['cosmo_constants']['G']
        if unit is None:
            try:
                unit = self.input_dict['repopulations'][
                    'params_to_save']['mass_from_Vmax']['unit']
            except KeyError:
                unit = 'Msun'

        Rmax_over_rs = self.RmaxoverrS(density_profile)

        return (Vmax ** 2 * Rmax / cosmo_G
                * self.ff(radius_normalized)
                / self.ff(Rmax_over_rs)).to(u.Unit(unit))

    def theta_s(self, Vmax=None, Cv=None, D_Earth=None, cosmo_H_0=None,
                unit='degree'):
        # Angular size of subhalos (up to R_s)
        if Vmax is None:
            Vmax = self.get_parameter('Vmax')
        if Cv is None:
            Cv = self.get_parameter('Cv')
        if D_Earth is None:
            D_Earth = self.get_parameter('D_Earth')
        if cosmo_H_0 is None:
            cosmo_H_0 = self.input_dict['cosmo_constants']['H_0']

        return np.arctan(
            self.R_s(Vmax, Cv, cosmo_H_0) / D_Earth).to(u.Unit(unit))

    # ----------- Cv ---------------------------------------------------

    def Cv_Mol2021_redshift0_scattered(self,
            Vmax=None,
            c0=1.75e5, c1=-0.90368, c2=0.2749, c3=-0.028,
            sigma_scatter=0.):
        # Median subhalo concentration depending on its Vmax and
        # its redshift (here z=0).
        # Moline et al. 2110.02097

        # Create a scatter in the concentration parameter of the
        # repopulated population.
        # Scatter in logarithmic scale, following a Gaussian distribution.
        #
        # :param C: float or array-like
        #     Concentration of a subhalo (according to the concentration
        #     law).
        # :return: float or array-like
        #     Subhalos with scattered concentrations.
        #
        # V - max radial velocity of a bound particle in the subhalo [km/s]
        if Vmax is None:
            Vmax = self.get_parameter('Vmax')
        ci = [c0, c1, c2, c3]
        Vmax = (Vmax * u.s / u.km).to(1)

        yy = ci[0] * (1 + (sum([ci[i + 1] * np.log10(Vmax/10.) ** (i + 1)
                                for i in range(3)])))
        try:
            scatter = 10 ** self.rng.normal(
                loc=0, scale=sigma_scatter, size=len(Vmax))
        except TypeError:
            scatter = 10 ** self.rng.normal(
                loc=0, scale=sigma_scatter, size=1)

        return yy * scatter * u.dimensionless_unscaled

    # ----------- J-FACTORS --------------------------------------------
    def J_general(
            self, radius_normalized=None, D_Earth=None,
            density_profile=None,
            calculate_from=None, unit=None,
            Vmax=None, Cv=None, cosmo_G=None, cosmo_H_0=None,
            Mass=None, Cdelta=None, rho_crit=None,
            rho_0=None, r_s=None
            ):
        """
        J-factor enclosing whole subhalo as a function of the
        subhalo Vmax.

        :param V: float or array-like  [km/s]
            Maximum circular velocity inside a subhalo.
        :param D_Earth: float or array-like [kpc]
            Distance between the subhalo and the Earth.
        :param C: float or array-like
            Subhalo concentration.
        :param unit: str
            Change the output units of the Jfactors.

        :return: float or array-like
            Jfactor of a whole subhalo.
        """
        if calculate_from is None:
            calculate_from = self.input_dict['repopulations'][
                    'params_to_save']['J_abs_vel']['calculate_from']
        if D_Earth is None:
            D_Earth = self.get_parameter('D_Earth')
        if density_profile is None:
            density_profile = self.input_dict['configurations'][
                self.configuration]['internal_density_profile']

        if radius_normalized is None:
            # TODO fix this ;)
            radius_normalized = 1.

        yy = self.fff(radius_normalized, density_profile) / D_Earth**2.
        print(self.ff(radius_normalized))

        if calculate_from == 'Vmax_Rmax':
            if Vmax is None:
                Vmax = self.get_parameter('Vmax')
            if Cv is None:
                Cv = self.get_parameter('Cv')
            if cosmo_G is None:
                cosmo_G = self.input_dict['cosmo_constants']['G']
            if cosmo_H_0 is None:
                cosmo_H_0 = self.input_dict['cosmo_constants']['H_0']

            Rmax_over_rs = self.RmaxoverrS(density_profile)
            print(Rmax_over_rs)

            yy *= (cosmo_H_0 / 4. / np.pi / cosmo_G ** 2
                   * np.sqrt(Cv / 2) * Vmax ** 3
                   * Rmax_over_rs**3.
                   / self.ff(Rmax_over_rs, density_profile)**2.)

        elif calculate_from == 'rho0_rS':
            if rho_0 is None:
                rho_0 = self.get_parameter('rho_0')
            if r_s is None:
                r_s = self.get_parameter('r_s')

            yy *= 4. * np.pi * rho_0**2. * r_s**3.

        elif calculate_from == 'mass_Cdelta':
            if Mass is None:
                Mass = self.get_parameter('Mass')
            if Cdelta is None:
                Cdelta = self.get_parameter('Cdelta')
            if rho_crit is None:
                rho_crit = self.input_dict['cosmo_constants']['rho_crit']

            yy *= (200 / 3. * rho_crit * Mass * Cdelta**3.
                   / self.ff(Cdelta, density_profile)**2.)

        else:
            raise ValueError(
                f'Error at inicializing the way to calculate the Jfactor.'
                + f' with value {calculate_from}.'
                  'Allowed values are: mass_Cdelta, Vmax_Rmax, and rho0_rS')

        if unit is None:
            try:
                unit = self.input_dict['repopulations'][
                    'params_to_save']['J_abs_vel']['unit']
            except KeyError:
                unit = 'GeV2 cm-5'

        return yy.to(u.Unit(unit), equivalencies=mass_energy2)

    def J_whole_fromVmax(
            self, Vmax=None, D_Earth=None, Cv=None,
            cosmo_G=None, cosmo_H_0=None, unit=None):
        """
        J-factor enclosing whole subhalo as a function of the
        subhalo Vmax.

        :param V: float or array-like  [km/s]
            Maximum circular velocity inside a subhalo.
        :param D_Earth: float or array-like [kpc]
            Distance between the subhalo and the Earth.
        :param C: float or array-like
            Subhalo concentration.
        :param unit: str
            Change the output units of the Jfactors.

        :return: float or array-like
            Jfactor of a whole subhalo.
        """
        if Vmax is None:
            Vmax = self.get_parameter('Vmax')
        if D_Earth is None:
            D_Earth = self.get_parameter('D_Earth')
        if Cv is None:
            Cv = self.get_parameter('Cv')
        if cosmo_G is None:
            cosmo_G = self.input_dict['cosmo_constants']['G']
        if cosmo_H_0 is None:
            cosmo_H_0 = self.input_dict['cosmo_constants']['H_0']
        if unit is None:
            try:
                unit = self.input_dict['repopulations'][
                    'params_to_save']['J_abs_vel']['unit']
            except KeyError:
                unit = 'GeV2 cm-5'

        yy = (2.1625758423 ** 3. / D_Earth ** 2.
              / self.ff(2.1625758423) ** 2
              * cosmo_H_0 / 12 / np.pi / cosmo_G ** 2
              * np.sqrt(Cv / 2) * Vmax ** 3)

        return yy.to(u.Unit(unit), equivalencies=mass_energy2)


    def Js_fromVmax(
            self, Vmax=None, D_Earth=None, Cv=None,
            cosmo_G=None, cosmo_H_0=None, unit=None):
        """
        Jfactor enclosing the subhalo up to rs as a function of Vmax.

        :param V: float or array-like  [km/s]
            Maximum circular velocity inside a subhalo.
        :param D_Earth: float or array-like [kpc]
            Distance between the subhalo and the Earth.
        :param C: float or array-like
            Subhalo concentration.
        :param units: str
            Change the output units of the Jfactors.

        :return: float or array-like
            Jfactor of a subhalo up to rs.
        """
        if Vmax is None:
            Vmax = self.get_parameter('Vmax')
        if D_Earth is None:
            D_Earth = self.get_parameter('D_Earth')
        if Cv is None:
            Cv = self.get_parameter('Cv')
        if cosmo_G is None:
            cosmo_G = self.input_dict['cosmo_constants']['G']
        if cosmo_H_0 is None:
            cosmo_H_0 = self.input_dict['cosmo_constants']['H_0']
        if unit is None:
            try:
                unit = self.input_dict['repopulations'][
                    'params_to_save']['Js_vel']['unit']
            except KeyError:
                unit = 'GeV2 cm-5'

        return 7/8. * self.J_whole_fromVmax(
            Vmax, D_Earth, Cv,
            cosmo_G=cosmo_G, cosmo_H_0=cosmo_H_0, unit=unit
        ).to(u.Unit(unit), equivalencies=mass_energy2)

    def J03_fromVmax(
            self, Vmax=None, D_Earth=None, Cv=None,
            cosmo_G=None, cosmo_H_0=None, unit=None):
        """
        Jfactor enclosing the subhalo up to 0.3 degrees as a
        function of Vmax.

        :param V: float or array-like  [km/s]
            Maximum circular velocity inside a subhalo.
        :param D_Earth: float or array-like [kpc]
            Distance between the subhalo and the Earth.
        :param C: float or array-like
            Subhalo concentration.
        :param units: str
            Change the output units of the Jfactors.

        :return: float or array-like
            Jfactor of a subhalo up to 0.3 degrees.
        """
        if Vmax is None:
            Vmax = self.get_parameter('Vmax')
        if D_Earth is None:
            D_Earth = self.get_parameter('D_Earth')
        if Cv is None:
            Cv = self.get_parameter('Cv')
        if cosmo_G is None:
            cosmo_G = self.input_dict['cosmo_constants']['G']
        if cosmo_H_0 is None:
            cosmo_H_0 = self.input_dict['cosmo_constants']['H_0']
        if unit is None:
            try:
                unit = self.input_dict['repopulations'][
                    'params_to_save']['J03_vel']['unit']
            except KeyError:
                unit = 'GeV2 cm-5'

        return (self.J_whole_fromVmax(
            Vmax, D_Earth, Cv, cosmo_G=cosmo_G, cosmo_H_0=cosmo_H_0,
            unit=unit)
                * (1 - 1
                   / (1 + 2.163 * D_Earth * np.tan(0.15 * np.pi / 180.)
                      / self.Rmax(Vmax, Cv, cosmo_H_0)) ** 3)
                ).to(u.Unit(unit), equivalencies=mass_energy2)


    # ------- Internal density profile ---------------------------------
    def ff(self, x, density_profile=None):
        if density_profile is None:
            density_profile = self.input_dict['configurations'][
                self.configuration]['internal_density_profile']

        if isinstance(x, list):
            x = np.array(x)

        if density_profile == 'NFW':
            return np.log(1. + x) - x / (1. + x)

        elif density_profile == 'Burkert':
            return (0.25 * np.log(1. + x**2.)
                    + 0.5 * np.log(1 + x)
                    - 0.5 * np.arctan(x))

        else:
            try:
                formula = density_profile['formula']
                try:
                    params = density_profile['params']
                except KeyError:
                    params = []

                int_total = np.zeros_like(x)

                def integrand(x_prime):
                    rho_x = self.calculate_formula(
                        x_prime, formula, params)
                    return x_prime ** 2 * rho_x

                if isinstance(x, float) or isinstance(x, int):
                    int_total = quad(
                        lambda x_prime: integrand(x_prime),
                        a=0., b=x)[0]

                elif isinstance(x, list) or isinstance(x, np.ndarray):
                    for ni, xi in enumerate(x):
                        int_total[ni] = quad(
                            lambda x_prime: integrand(x_prime),
                            a=0., b=xi)[0]

                return int_total

            except Exception as e:
                raise ValueError(
                    f'Error evaluating formula ' + formula
                    + f' with parameters {params}: {e}')

    def fff(self, x, density_profile=None):
        if density_profile is None:
            density_profile = self.input_dict['configurations'][
                self.configuration]['internal_density_profile']

        if isinstance(x, list):
            x = np.array(x)

        if density_profile == 'NFW':
            return (1 - 1 / (1 + x) ** 3.) / 3.

        elif density_profile == 'Burkert':
            return 0.25 * (2. - 1 / (1 + x)
                           - 1 / (1 + x ** 2) - np.arctan(x))

        else:
            try:
                formula = density_profile['formula']
                try:
                    params = density_profile['params']
                except KeyError:
                    params = []

                int_total = np.zeros_like(x)

                def integrand(x_prime):
                    rho_x = self.calculate_formula(
                        x_prime, formula, params)
                    return x_prime ** 2 * rho_x ** 2

                if isinstance(x, float) or isinstance(x, int):
                    int_total = quad(
                        lambda x_prime: integrand(x_prime),
                        a=0., b=x)[0]

                elif isinstance(x, list) or isinstance(x, np.ndarray):
                    for ni, xi in enumerate(x):
                        int_total[ni] = quad(
                            lambda x_prime: integrand(x_prime),
                            a=0., b=xi)[0]

                return int_total

            except Exception as e:
                raise ValueError(
                    f'Error evaluating formula ' + formula
                    + f' with parameters {params}: {e}')

    def RmaxoverrS(self, density_profile=None):

        if density_profile is None:
            density_profile = self.input_dict['configurations'][
                self.configuration]['internal_density_profile']

        try:
            return density_profile['RmaxoverrS']

        except (TypeError, KeyError):

            if density_profile == 'NFW':
                return 2.16257584237016

            elif density_profile == 'Burkert':
                return 3.244597456471571

            else:
                try:
                    def funcc(xx):
                        return -self.ff(xx, density_profile) / xx

                    argwhere = minimize(funcc, x0=3.)

                    density_profile['RmaxoverrS'] = argwhere['x'][0]
                    return argwhere['x'][0]

                except Exception as e:
                    raise ValueError(
                        f'Error evaluating density profile'
                        + density_profile + f': {e}')

    # ----------- SRD --------------------------------------------------
    def srd_constant(self, xx, args):
        return args * np.ones_like(xx)

    def srd_exponential(self, xx, exp_fit, last_subhalo):

        try:
            xx = xx.to(u.kpc).value
        except AttributeError:
            xx = xx
        last_subhalo = (
                last_subhalo['value']
                * u.Unit(last_subhalo['unit']).to(u.kpc))

        return (exp_fit[1] * np.exp(exp_fit[0] / xx * exp_fit[2])
                * (xx >= last_subhalo))

    # ----------- SHVF -------------------------------------------------
    def power_law(self, Vmax, V0, slope):
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

    def SHVF_integral(self, Vmax_min, Vmax_max,
                      formula=None, params=None,
                      force_no_fraction=None):
        if formula is None:
            formula = self.input_dict['configurations'][
                self.configuration]['SHVF']['formula']
        if params is None:
            params = self.input_dict['configurations'][
                self.configuration]['SHVF']['params']

        fraction = 1.
        if not force_no_fraction:
            if self.input_dict['repopulations']['use_spherical_shells']:
                cv_mean = self.calculate_formula(
                    Vmax_max * u.km / u.s,
                    self.input_dict['configurations'][
                        self.configuration]['Cv']['formula'],
                    {'c0': self.input_dict['configurations'][
                        self.configuration]['Cv']['params']['c0'],
                     'sigma_scatter': 0.}
                )
                c200 = self.C200_from_Cv(cv_mean)
                # print((Vmax_max * u.km / u.s / (
                #             self.input_dict['cosmo_constants']['H_0']
                #             * np.sqrt(2. * cv_mean))).to(u.kpc))
                M = self.mass_from_Vmax(
                    Vmax=Vmax_max * u.km / u.s,
                    Rmax=(Vmax_max * u.km / u.s / (
                            self.input_dict['cosmo_constants']['H_0']
                            * np.sqrt(2. * cv_mean))).to(u.kpc),
                    c200=c200).to(u.Msun)[0]
                print(c200, M)

                # print(cv_mean)
                # print(c200)
                # print(M)
                self._Rcut = self.R_Cut(Vmax_max)
                print(self._Rcut)
                fraction = self.dist_frac(self._Rcut)
                print(fraction)
                # print()

        print('total number subh', quad(
            self.calculate_formula,
            a=Vmax_min, b=Vmax_max,
            args=(formula, params))[0])
        return int(np.rint(fraction * quad(
            self.calculate_formula,
            a=Vmax_min, b=Vmax_max,
            args=(formula, params))[0]))

    def dist_frac(self, Rcut, formula=None, params=None):
        if formula is None:
            formula = self.input_dict['configurations'][
                self.configuration]['SRD']['formula']
        if params is None:
            params = self.input_dict['configurations'][
                self.configuration]['SRD']['params']

        R_vir = self.input_dict['host']['R_vir'].value

        if Rcut < 8.5:
            return (quad(
                self.calculate_formula, a=8.5 - Rcut, b=8.5 + Rcut,
                args=(formula, params))[0]
                    / quad(
                self.calculate_formula, a=0., b=R_vir,
                args=(formula, params))[0])
        elif Rcut + 8.5 >= R_vir:
            return 1.
        else:
            return (quad(
                self.calculate_formula, a=0., b=8.5 + Rcut,
                args=(formula, params))[0]
                    / quad(
                self.calculate_formula, a=0., b=R_vir,
                args=(formula, params))[0])

    # max radial dist. from earth at which subhalo of mass M might be observed
    def R_Cut(self, M, D_D=80., M_D=2.e8, C_D_corr=19,
              **kwargs):
        # NOTE: fraction of luminosity of Draco used as
        M = 8226.1 * M ** 3.72
        print('mass from Vmax', M)

        # cutoff is R = .1, i.e. 10%
        # C = self.C_200(M, 0.1, **kwargs)  # este 0.1 es la distancia a la que
        def C_200(M, x, ci=[19.9, -0.195, 0.089, 0.089, -0.54]):
            return ci[0] * (1 + (
                sum([(ci[i + 1] * np.log10(M * 0.7 / 10 ** 8)) ** (i + 1) for i
                     in range(3)]))) * (1 + ci[4] * np.log10(x / 402.))

        C = C_200(M, 1, **kwargs)  # TODO: try with boost and/or upper scatter

        C_D_corr = C_200(M_D, D_D)
        print('C_D_corr, C', C_D_corr, C)
        # estaría el
        # subhalo, 0.1 kpc,
        # del centro de la galaxia. Es muy pequeña, lo cual nos da una C
        # mayor (subhalos cerca del centro están más concentrados)
        # y por tanto un R_Cut mayor, con lo que repoblamos una región más
        # grande (es decir, estamos siendo conservadoras en sentido
        # de no dejarnos posibles subhalos relevantes sin simular)
        return ((M * (D_D)**2 * C**3 * self.ff(C_D_corr)**2
                / (M_D * 0.1 * C_D_corr**3 * self.ff(C)**2))**.5)#.value



    # def C_200(self, M, x, ci=[19.9, -0.195, 0.089, 0.089, -0.54]):
    #     return (ci[0]
    #             * (1 + sum([(ci[i + 1] * np.log10(M.value * 0.677 / 10 ** 8)
    #                          ) ** (i + 1)
    #                         for i in range(3)]))
    #             * (1 + ci[4] * np.log10(
    #                 x / self.input_dict['host']['R_vir'].value)))
