import os
import sys
import yaml
import copy
import time
import h5py
import psutil
import inspect

import numpy as np

from scipy.optimize import newton
from scipy.interpolate import UnivariateSpline
from scipy.integrate import simpson, cumtrapz

from astropy import units as u

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


def ff(c):
    return np.log(1. + c) - c / (1. + c)


class RepopAlgorithm:
    def __init__(self, path_input):

        self.path_output = None
        self.configuration = None

        if type(path_input) == str:
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

        self.input_dict['repopulations']['num_subs_max'] = int(float(
            self.input_dict['repopulations']['num_subs_max']))
        self.input_dict['repopulations']['num_brightest'] = int(
            self.input_dict['repopulations']['num_brightest'])

        self.RangeMin = self.input_dict['repopulations']['RangeMin']
        self.RangeMax = self.input_dict['repopulations']['RangeMax']

        for key, value in self.input_dict['cosmo_constants'].items():
            self.input_dict['cosmo_constants'][key] = (
                self.input_dict['cosmo_constants'][key]['value']
                * u.Unit(self.input_dict['cosmo_constants'][key]['unit']
                         )#.decompose()
            )

        for key, value in self.input_dict['host'].items():
            self.input_dict['host'][key] = (
                self.input_dict['host'][key]['value']
                * u.Unit(self.input_dict['host'][key]['unit']
                         )#.decompose()
            )

        self.subhalo_data = {}
        self.units = {}

        for key, value in self.input_dict['units'].items():
            self.units[key] = u.Unit(value)
            # self.units[key] = self.units[key]#.decompose()
        print('aaaaaa')

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
            print()
            print(self.configuration)
            print(self.RangeMin, self.RangeMax)
            print('    Number of repop subhalos: %i'
                  % self.SHVF_integral(self.RangeMin, self.RangeMax))
            self.interior_full_repop()

        # if full_repop:
        #     self.interior_full_repop()
        # else:
        #     elf.interior_loop_singularbrightest()

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
                    f'Error evaluating formula' + formula
                    + f'with parameters {bb}: {e}')
            return aa

        elif callable(formula):
            if params is not None:
                if isinstance(params, dict):
                    return formula(xx, **params)
                return formula(xx, params)
            else:
                return formula(xx)

    def get_parameter(self, name, parametrization):
        """
        Retrieve parameter, computing if necessary.
        Uses values to avoid recomputation.
        """
        if name in self.subhalo_data:
            return self.subhalo_data[name]

        if hasattr(self, name):
            self.subhalo_data[name] = getattr(self, name)()
            return

        formula = parametrization.get('formula', None)

        if isinstance(formula, str):

            bb = {}

            params = parametrization.get('params', None)
            if isinstance(params, float) or isinstance(params, int):
                bb['params'] = params
            elif isinstance(params, list):
                bb['params'] = np.array(params, dtype=float)

            variables = parametrization.get('variables', None)
            if isinstance(variables, str):
                bb[variables] = self.get_parameter(variables, None)
            elif isinstance(variables, list):
                for var in variables:
                    bb[var] = self.get_parameter(var, None)

            # Evaluate formula
            try:
                aa = eval(parametrization['formula'], {}, bb)
            except Exception as e:
                raise ValueError(
                    f'Error evaluating formula'
                    + parametrization['formula']
                    + f'with parameters {bb}: {e}')

            self.subhalo_data[name] = aa
            return

        elif callable(formula):

            vars_for_func = {}
            variables = parametrization.get('variables', [])
            if isinstance(variables, str):
                variables = [variables]
            for var in variables:
                vars_for_func[var] = self.get_parameter(var, None)

            params = parametrization.get('params', None)
            if params is not None:
                vars_for_func['params'] = params

            self.subhalo_data[name] = formula(**vars_for_func)
            return

    def calculate_characteristics_subhalo(
            self, Vmax=None, Distgc=None, position_Earth=None):
        if Vmax is None:
            Vmax = self.subhalo_data['Vmax']
        if Distgc is None:
            Distgc = self.subhalo_data['Distgc']
        if position_Earth is None:
            position_Earth = self.input_dict['host']['position_Earth']

        # Random distribution of subhalos around the celestial sphere
        num_subs = len(Vmax)
        self.subhalo_data['gal_theta'] = self.rng.uniform(
            0, 2 * np.pi, num_subs)
        self.units['Vmax'] = 'kpc'
        self.subhalo_data['gal_phi'] = np.arccos(
            2 * self.rng.uniform(0, 1, num_subs) - 1)
        self.units['Vmax'] = 'kpc'

        # Positions of the subhalos
        self.subhalo_data['repop_Xs'] = (
                Distgc * np.cos(self.subhalo_data['gal_theta'])
                * np.sin(self.subhalo_data['gal_phi']))
        self.units['Vmax'] = 'kpc'
        self.subhalo_data['repop_Ys'] = (
                Distgc * np.sin(self.subhalo_data['gal_theta'])
                * np.sin(self.subhalo_data['gal_phi']))
        self.units['Vmax'] = 'kpc'
        self.subhalo_data['repop_Zs'] = (
                Distgc * np.cos(self.subhalo_data['gal_phi']))
        self.units['Vmax'] = 'kpc'

        self.subhalo_data['D_Earth'] = ((
            self.subhalo_data['repop_Xs'] - position_Earth[0]) ** 2
            + (self.subhalo_data['repop_Ys'] - position_Earth[1]) ** 2
            + (self.subhalo_data['repop_Zs'] - position_Earth[2]) ** 2
            ) ** 0.5


        repop_C = self.Cv_Mol2021_redshift0_scattered(
            self.subhalo_data['Vmax'])
        scatter = self.rng.normal(
            loc=0,
            scale=self.input_dict['configurations'][self.configuration][
                'Cv']['params']['sigma_scatter'],
            size=num_subs)
        self.subhalo_data['Cv'] = repop_C * 10 ** scatter
        self.get_parameter('C200_from_Cv', None)

        for param in self.input_dict['repopulations']['columns_to_save']:
            self.get_parameter(
                param,
                self.input_dict['repopulations']['columns_to_save'][param])

        if self.input_dict['repopulations']['use_Roche']:
            self.subhalo_data['survives_Roche'] = (
                    self.R_t(
                        self.subhalo_data['Vmax'],
                        self.subhalo_data['Cv'],
                        self.subhalo_data['Distgc'])
                    > self.R_s(self.subhalo_data['Vmax'],
                               self.subhalo_data['Cv']))

        return

    def interior_loop_brightest(self):
        # We have 6 variables we want to save in our files,
        # change this number if necessary
        # (output from 'calculate_characteristics_subhalo()')

        brightest_Js = np.zeros((2 * self.input_dict['repopulations']['num_brightest'], 6))
        brightest_J03 = np.zeros((2 * self.input_dict['repopulations']['num_brightest'], 6))


        # We calculate our subhalo population in bins to save memory
        m_min = self.RangeMin

        while m_min < self.RangeMax:

            self.subhalo_data = {}

            try:
                m_max = np.min((
                    newton(
                        self.xx,
                        m_min,
                        args=[m_min, self.input_dict['repopulations']['num_subs_max']]),
                    self.RangeMax
                ))
            except RuntimeError:
                m_max = self.RangeMax

            num_subhalos = self.SHVF_integral(
                Vmax_min=m_min, Vmax_max=m_max)

            print(m_min, m_max, num_subhalos)

            # Montecarlo algorithm for creating the Vmax of subhalos
            x = np.geomspace(m_min, m_max, num=2000)
            y = self.calculate_formula(
                x, self.input_dict['configurations'][
                    self.configuration]['SHVF']['formula'],
                self.input_dict['configurations'][
                    self.configuration]['SHVF']['params']
            )
            cumul = [simpson(y=y[:i], x=x[:i]) for i in range(1, len(x))]
            cumul /= cumul[-1]
            x_mean = (x[1:] + x[:-1]) / 2.
            x_min = ((np.array(cumul) - 1e-8) < 0).argmin() - 1
            spline = UnivariateSpline(
                cumul[x_min:], x_mean[x_min:], s=0, k=1, ext=0)

            self.subhalo_data['Vmax'] = spline(self.rng.random(num_subhalos))
            self.units['Vmax'] = 'km/s'

            # Montecarlo algorithm for creating the Distgc of subhalos
            x = np.linspace(0., self.input_dict['host']['R_vir'], num=2000)
            y = self.calculate_formula(
                x,
                self.input_dict['configurations'][
                    self.configuration]['SRD']['formula'],
                self.input_dict['configurations'][
                    self.configuration]['SRD']['params']
            )
            cumul = [simpson(y=y[:i], x=x[:i]) for i in range(1, len(x))]
            cumul /= cumul[-1]
            x_mean = (x[1:] + x[:-1]) / 2.
            x_min = ((np.array(cumul) - 1e-8) < 0).argmin() - 1
            spline = UnivariateSpline(
                cumul[x_min:], x_mean[x_min:], s=0, k=1, ext=0)
            self.subhalo_data['Distgc'] = spline(self.rng.random(num_subhalos))
            self.units['Vmax'] = 'kpc'

            self.calculate_characteristics_subhalo()

            for new_sub in range(self.input_dict['repopulations']['num_brightest']):

                bright_Js = np.argmax(new_data[:, 0])

                while (self.R_t(new_data[bright_Js, 4],
                           new_data[bright_Js, 6],
                           new_data[bright_Js, 2],
                           cosmo_H_0, cosmo_G,
                           host_rho_0, host_r_s)
                       < R_s(new_data[bright_Js, 4],
                             new_data[bright_Js, 6],
                             cosmo_H_0)) \
                        and (new_data[bright_Js, 0] > 1.):
                    print('broken Js')
                    print(new_data[bright_Js, :])

                    new_data[bright_Js, 0] = 0.
                    bright_Js = np.argmax(new_data[:, 0])

                brightest_Js[
                self.input_dict['repopulations']['num_brightest'] + new_sub, :] = new_data[
                    bright_Js, [0, 2, 3, 4, 5, 6]]
                new_data[bright_Js, 0] = 0.

            for new_sub in range(self.input_dict['repopulations']['num_brightest']):

                bright_J03 = np.argmax(new_data[:, 1])

                while (R_t(new_data[bright_J03, 4],
                           new_data[bright_J03, 6],
                           new_data[bright_J03, 2],
                           cosmo_H_0, cosmo_G,
                           host_rho_0, host_r_s)
                       < R_s(new_data[bright_J03, 4],
                             new_data[bright_J03, 6],
                             cosmo_H_0)) \
                        and (new_data[bright_J03, 1] > 1.):
                    print('broken J03')
                    print(new_data[bright_J03, :])

                    new_data[bright_J03, 1] = 0.
                    bright_J03 = np.argmax(new_data[:, 1])

                brightest_J03[
                self.input_dict['repopulations']['num_brightest'] + new_sub, :] = new_data[bright_J03,
                                                    1:]
                new_data[bright_J03, 1] = 0.

            # if sum(new_data[:, 0]) > 1.:
            #     for new_sub in range(self.input_dict['repopulations']['num_brightest']):
            #
            #         while sum(new_data[:, 0]) > 1.:
            #
            #             bright_Js = np.argmax(new_data[:, 0])
            #
            #             brightest_Js[
            #             self.input_dict['repopulations']['num_brightest'] + new_sub, :] = new_data[
            #                 bright_Js, [0, 2, 3, 4, 5, 6]]
            #             new_data[bright_Js, 0] = 0.

            # while (R_t(new_data[bright_Js, 4],
            #                new_data[bright_Js, 6],
            #                new_data[bright_Js, 2],
            #                cosmo_H_0, cosmo_G,
            #                host_rho_0, host_r_s)
            #            < R_s(new_data[bright_Js, 4],
            #                  new_data[bright_Js, 6],
            #                  cosmo_H_0)):
            #         print('subhalo broken (Js)')
            # new_data[bright_Js, 0] = 0.
            # bright_Js = np.argmax(new_data[:, 0])

            # if sum(new_data[:, 1]) > 1.:
            #     for new_sub in range(self.input_dict['repopulations']['num_brightest']):
            #         while sum(new_data[:, 1]) > 1.:
            #             bright_J03 = np.argmax(new_data[:, 1])
            #
            #             brightest_J03[
            #             self.input_dict['repopulations']['num_brightest'] + new_sub, :] = new_data[bright_J03, 1:]
            #             new_data[bright_J03, 1] = 0.

            # while (R_t(new_data[bright_J03, 4],
            #            new_data[bright_J03, 6],
            #            new_data[bright_J03, 2],
            #            cosmo_H_0, cosmo_G,
            #            host_rho_0, host_r_s)
            #        < R_s(new_data[bright_J03, 4],
            #              new_data[bright_J03, 6],
            #              cosmo_H_0)):
            #     print('subhalo broken (J03)')
            # new_data[bright_J03, 1] = 0.
            # bright_J03 = np.argmax(new_data[:, 1])

            # We take the brightest subhalos only
            brightest_Js = brightest_Js[
                           np.argsort(brightest_Js[:, 0])[::-1],
                           :]
            brightest_J03 = brightest_J03[
                            np.argsort(brightest_J03[:, 0])[::-1], :]

            m_min = new_mmin

        return (brightest_Js[:self.input_dict['repopulations']['num_brightest'], :],
                    brightest_J03[:self.input_dict['repopulations']['num_brightest'], :])

    def xx(self, mmax, mmin, root):
        return self.SHVF_integral(Vmax_min=mmin, Vmax_max=mmax) - root

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

            datasets = {}

            for iter_idx in range(
                    self.input_dict['repopulations']['its']):

                if (iter_idx
                        % self.input_dict['repopulations']['print_freq']
                        == 0):
                    print('    %s %s: it %d' % (
                        time.strftime(
                            ' %Y-%m-%d %H:%M:%S', time.gmtime()),
                        self.configuration, iter_idx))
                    progress = open(self.path_output + 'progress_'
                                    + self.configuration + '.txt'
                                    , 'a')
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

                # We calculate our subhalo population in bins
                m_min = self.RangeMin

                while m_min < self.RangeMax:

                    self.subhalo_data = {}

                    try:
                        m_max = np.min((
                            newton(
                                self.xx, m_min,
                                args=[m_min, self.input_dict[
                                    'repopulations']['num_subs_max']]),
                            self.RangeMax
                        ))
                    except RuntimeError:
                        m_max = self.RangeMax

                    num_subhalos = self.SHVF_integral(
                        Vmax_min=m_min, Vmax_max=m_max)

                    print(m_min, m_max, num_subhalos)


                    # Montecarlo algorithm for creating of subhalo Vmax
                    x = np.geomspace(m_min, m_max, num=200)
                    y = self.calculate_formula(
                        x,
                        self.input_dict['configurations'][
                            self.configuration]['SHVF']['formula'],
                        self.input_dict['configurations'][
                            self.configuration]['SHVF']['params']
                    )
                    cumul = cumtrapz(
                        y=y * x * np.log(10), x=np.log10(x), initial=0)
                    cumul /= cumul[-1]
                    spline = UnivariateSpline(
                        cumul, x, s=0, k=1, ext=0)

                    self.subhalo_data['Vmax'] = spline(self.rng.random(
                        num_subhalos)) * self.units['Vmax']

                    # Montecarlo algorithm for creating of subhalo Distgc
                    x = np.linspace(
                        0.,
                        (self.input_dict['host']['R_vir'].to(
                            self.units['Distgc'])),
                        num=2000)
                    y = self.calculate_formula(
                        x,
                        self.input_dict['configurations'][
                            self.configuration]['SRD']['formula'],
                        self.input_dict['configurations'][
                            self.configuration]['SRD']['params']
                    )
                    cumul = cumtrapz(y=y, x=x, initial=0)
                    cumul /= cumul[-1]
                    x_min = ((np.array(cumul) - 1e-8) < 0).argmin() - 1
                    spline = UnivariateSpline(
                        cumul[x_min:], x[x_min:], s=0, k=1, ext=0)

                    self.subhalo_data['Distgc'] = (
                            spline(self.rng.random(num_subhalos))
                            * self.units['Distgc'])

                    # self.calculate_characteristics_subhalo()

                    # if self.input_dict['repopulations']['saveall']:

                    for key, array in self.subhalo_data.items():
                        # If dataset already exists, get it
                        if key in iter_group:
                            datasets = iter_group[key]
                        else:
                            # Create dataset for new key
                            datasets[key] = iter_group.create_dataset(
                                key,
                                shape=(0,),
                                maxshape=(None,),
                                dtype=array.dtype,
                                chunks=True,
                                compression='gzip'
                            )

                            # Save metadata for new key
                            if key in self.input_dict['units'].keys():
                                datasets[key].attrs['units'] = str(
                                self.input_dict_strings['units'][key])

                        # Append the new batch data to the dataset
                        dataset = datasets[key]
                        current_size = dataset.shape[0]
                        new_size = current_size + num_subhalos
                        dataset.resize((new_size,))
                        dataset[current_size:new_size] = array

                    m_min = m_max
            f.flush()
        return

    '''
    def sort_by_parameter(self, param_name):
        """
        Return indices to sort objects based on a parameter.
        """
        param_data = self.get_parameter(
            param_name).value  # get raw numpy array
        return np.argsort(param_data)

    def get_sorted_parameters(self, param_name):
        indices = self.sort_by_parameter(param_name)
        sorted_params = {}
        for key in self.subhalo_data:
            sorted_params[key] = self.get_parameter(key)[indices]
        # Also include computed parameters if needed
        for key in self.values:
            # values contains computed subhalo_data
            # retrieve and sort as well
            pass
        return sorted_params
    '''
    # ----------- General formulas -------------------------------------
    def R_max(self, Vmax=None, Cv=None, cosmo_H_0=None):
        """
        Calculate R_max of a subhalo.

        :param V: float or array-like [km/s]
            Maximum circular velocity inside a subhalo.
        :param C: float or array-like
            Subhalo concentration.

        :return: float or array-like [kpc]
            R_max of the subhalo given by the inputs.
        """
        if Vmax is None:
            Vmax = self.get_parameter('Vmax', None)
        if Cv is None:
            Cv = self.get_parameter('Cv', None)
        if cosmo_H_0 is None:
            cosmo_H_0 = self.input_dict['cosmo_constants']['H_0']

        return Vmax / cosmo_H_0 * np.sqrt(2. / Cv) * 1e3

    def R_s(self, Vmax=None, Cv=None, cosmo_H_0=None):
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
            Vmax = self.get_parameter('Vmax', None)
        if Cv is None:
            Cv = self.get_parameter('Cv', None)
        if cosmo_H_0 is None:
            cosmo_H_0 = self.input_dict['cosmo_constants']['H_0']

        return self.R_max(Vmax, Cv, cosmo_H_0) / 2.163

    def R_t(self, Vmax=None, Cv=None, Distgc=None,
            cosmo_H_0=None, cosmo_G=None,
            host_rho_0=None, host_r_s=None):
        """
        Calculation of tidal radius (R_t) of a subhalo, following the
        NFW analytical expression for a subhalo density profile.

        Definition of R_t: 1603.04057 King radius pg 14

        :param V: float or array-like [km/s]
            Maximum circular velocity inside a subhalo.
        :param C: float or array-like
            Subhalo concentration.
        :param Distgc: float or array-like [kpc]
            Distance from the center of the subhalo to the
            Galactic Center (GC).

        :return: float or array-like [kpc]
            Tidal radius of the subhalo given by the inputs.
        """
        if Vmax is None:
            Vmax = self.get_parameter('Vmax', None)
        if Cv is None:
            Cv = self.get_parameter('Cv', None)
        if Distgc is None:
            Distgc = self.get_parameter('Distgc', None)
        if cosmo_H_0 is None:
            cosmo_H_0 = self.input_dict['cosmo_constants']['H_0']
        if cosmo_G is None:
            cosmo_G = self.input_dict['cosmo_constants']['G']
        if host_rho_0 is None:
            host_rho_0 = self.input_dict['host']['rho_0']
        if host_r_s is None:
            host_r_s = self.input_dict['host']['r_s']

        R_max = self.R_max(Vmax=Vmax, Cv=Cv, cosmo_H_0=cosmo_H_0)
        c200 = self.get_parameter('C200_from_Cv', None)
        M_subhalo = self.mass_from_Vmax(Vmax, R_max, c200, cosmo_G)
        M_host = self.Mhost_encapsulated(Distgc, host_rho_0, host_r_s)

        return Distgc * (M_subhalo / (3 * M_host)) ** (1/3.)

    def Mhost_encapsulated(self, Distgc=None,
                           host_rho_0=None, host_r_s=None):
        """
        Host mass encapsulated up to a certain radius. We are following
        a NFW density profile for the host.

        :param R: float or array-like [kpc]
            Radius up to which we integrate the density profile.

        :return: float or array-like [Msun]
            Host mass encapsulated up to R.
        """
        if Distgc is None:
            Distgc = self.get_parameter('Distgc', None)
        if host_rho_0 is None:
            host_rho_0 = self.input_dict['host']['rho_0']
        if host_r_s is None:
            host_r_s = self.input_dict['host']['r_s']

        return (4 * np.pi * host_rho_0
                * host_r_s ** 3
                * (np.log((host_r_s + Distgc) / host_r_s)
                   - Distgc / (host_r_s + Distgc)))

    def C200_from_Cv(self, Cv=None):
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
            Cv = self.get_parameter('Cv', None)

        def int_interior(c200i, Cvi):
            return (200 * (np.log(1. + 2.163) - 2.163 / (1. + 2.163))
                    / ff(c200i) * (c200i / 2.163) ** 3 - Cvi)

        if type(Cv) == float:
            c200 = newton(int_interior, x0=40.0, args=[Cv])
        else:
            c200 = np.array([newton(int_interior, x0=40.0, args=[i])
                             for i in Cv])

        return c200

    def mass_from_Vmax(self, Vmax=None, R_max=None, c200=None,
                       cosmo_G=None):
        """
        Mass from a subhalo assuming a NFW profile.
        Theoretical steps in Moline16.

        :param Vmax: float or array-like [km/s]
            Maximum radial velocity of a bound particle in the subhalo.
        :param R_max: float or array-like [kpc]
            Radius at which Vmax happens (from the subhalo center).
        :param c200: float or array-like
            Concentration of the subhalo in terms of mass.
        :return: float or array-like [Msun]
            Mass from the subhalo assuming a NFW profile.
        """
        if Vmax is None:
            Vmax = self.get_parameter('Vmax', None)
        if R_max is None:
            R_max = self.get_parameter('R_max', None)
        if c200 is None:
            try:
                self.get_parameter('c200', None)
            except:
                self.get_parameter('C200_from_Cv', None)
            c200 = self.get_parameter('C200_from_Cv', None)

        if cosmo_G is None:
            cosmo_G = self.input_dict['cosmo_constants']['G']

        return (Vmax ** 2 * R_max / cosmo_G
                * ff(c200)
                / (np.log(1. + 2.163) - 2.163 / (1. + 2.163)))

    def theta_s(self, Vmax=None, Cv=None, D_Earth=None, cosmo_H_0=None):
        # Angular size of subhalos (up to R_s)
        if Vmax is None:
            Vmax = self.get_parameter('Vmax', None)
        if Cv is None:
            Cv = self.get_parameter('Cv', None)
        if D_Earth is None:
            D_Earth = self.get_parameter('D_Earth', None)
        if cosmo_H_0 is None:
            cosmo_H_0 = self.input_dict['cosmo_constants']['H_0']

        return 180 / np.pi * np.arctan(
            self.R_s(Vmax, Cv, cosmo_H_0) / D_Earth)

    # ----------- Cv ---------------------------------------------------
    def Cv_Mol2021_redshift0_scattered(
            self, V, c0=1.75e5, c1=-0.90368, c2=0.2749, c3=-0.028):
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
        ci = [c0, c1, c2, c3]
        yy = ci[0] * (1 + (sum([ci[i + 1] * np.log10(V) ** (i + 1)
                                for i in range(3)])))
        return yy

    # ----------- J-FACTORS --------------------------------------------
    def J_abs_vel(self, Vmax=None, D_Earth=None, Cv=None,
                  cosmo_G=None, cosmo_H_0=None,
                  change_units=True):
        """
        J-factor enclosing whole subhalo as a function of the
        subhalo Vmax.

        :param V: float or array-like  [km/s]
            Maximum circular velocity inside a subhalo.
        :param D_Earth: float or array-like [kpc]
            Distance between the subhalo and the Earth.
        :param C: float or array-like
            Subhalo concentration.
        :param change_units: Bool
            Change the output units of the Jfactors.

        :return: float or array-like
            Jfactor of a whole subhalo.
            Units in which it can be returned:
            -> [Msun**2 / kpc**5] with change_units=False
            -> [GeV**2 / cm**5] with change_units=True
        """
        if Vmax is None:
            Vmax = self.get_parameter('Vmax', None)
        if D_Earth is None:
            D_Earth = self.get_parameter('D_Earth', None)
        if Cv is None:
            Cv = self.get_parameter('Cv', None)
        if cosmo_G is None:
            cosmo_G = self.input_dict['cosmo_constants']['G']
        if cosmo_H_0 is None:
            cosmo_H_0 = self.input_dict['cosmo_constants']['H_0']

        yy = (2.163 ** 3. / D_Earth ** 2.
              / (np.log(1. + 2.163) - 2.163 / (1. + 2.163)) ** 2
              * cosmo_H_0 / 12 / np.pi / cosmo_G ** 2
              * np.sqrt(Cv / 2) * Vmax ** 3
              * 1e-3)

        if change_units:
            yy *= 4.446e6  # GeV ^ 2 cm ^ -5 Msun ^ -2 kpc ^ 5
        return yy


    def Js_vel(self, Vmax=None, D_Earth=None, Cv=None,
               cosmo_G=None, cosmo_H_0=None,
               change_units=True):
        """
        Jfactor enclosing the subhalo up to rs as a function of Vmax.

        :param V: float or array-like  [km/s]
            Maximum circular velocity inside a subhalo.
        :param D_Earth: float or array-like [kpc]
            Distance between the subhalo and the Earth.
        :param C: float or array-like
            Subhalo concentration.
        :param change_units: Bool
            Change the output units of the Jfactors.

        :return: float or array-like
            Jfactor of a subhalo up to rs.
            Units in which it can be returned:
            -> [Msun**2 / kpc**5] with change_units=False
            -> [GeV**2 / cm**5] with change_units=True
        """
        if Vmax is None:
            Vmax = self.get_parameter('Vmax', None)
        if D_Earth is None:
            D_Earth = self.get_parameter('D_Earth', None)
        if Cv is None:
            Cv = self.get_parameter('Cv', None)
        if cosmo_G is None:
            cosmo_G = self.input_dict['cosmo_constants']['G']
        if cosmo_H_0 is None:
            cosmo_H_0 = self.input_dict['cosmo_constants']['H_0']

        return self.J_abs_vel(
            Vmax, D_Earth, Cv,
            cosmo_G=cosmo_G, cosmo_H_0=cosmo_H_0,
            change_units=change_units) * 7 / 8

    def J03_vel(self, Vmax=None, D_Earth=None, Cv=None,
                cosmo_G=None, cosmo_H_0=None,
                change_units=True):
        """
        Jfactor enclosing the subhalo up to 0.3 degrees as a
        function of Vmax.

        :param V: float or array-like  [km/s]
            Maximum circular velocity inside a subhalo.
        :param D_Earth: float or array-like [kpc]
            Distance between the subhalo and the Earth.
        :param C: float or array-like
            Subhalo concentration.
        :param change_units: Bool
            Change the output units of the Jfactors.

        :return: float or array-like
            Jfactor of a subhalo up to 0.3 degrees.
            Units in which it can be returned:
            -> [Msun**2 / kpc**5] with change_units=False
            -> [GeV**2 / cm**5] with change_units=True
        """
        if Vmax is None:
            Vmax = self.get_parameter('Vmax', None)
        if D_Earth is None:
            D_Earth = self.get_parameter('D_Earth', None)
        if Cv is None:
            Cv = self.get_parameter('Cv', None)
        if cosmo_G is None:
            cosmo_G = self.input_dict['cosmo_constants']['G']
        if cosmo_H_0 is None:
            cosmo_H_0 = self.input_dict['cosmo_constants']['H_0']

        return (self.J_abs_vel(
            Vmax, D_Earth, Cv, cosmo_G=cosmo_G, cosmo_H_0=cosmo_H_0,
            change_units=change_units)
                * (1 - 1
                   / (1 + 2.163 * D_Earth * np.tan(0.15 * np.pi / 180.)
                      / self.R_max(Vmax, Cv, cosmo_H_0)) ** 3))

    # ----------- SRD --------------------------------------------------
    def srd_constant(self, xx, args):
        return args * np.ones_like(xx)

    def srd_exponential(self, xx, exp_fit, last_subhalo):
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
                      formula=None, params=None):
        if formula is None:
            formula = self.input_dict['configurations'][
                self.configuration]['SHVF']['formula']
        if params is None:
            params = self.input_dict['configurations'][
                self.configuration]['SHVF']['params']

        vmax_array = np.geomspace(Vmax_min, Vmax_max, num=150)

        yy = self.calculate_formula(
            vmax_array, formula=formula, params=params)

        return int(np.rint(simpson(
            y=yy * np.log(10) * vmax_array, x=np.log10(vmax_array))))
