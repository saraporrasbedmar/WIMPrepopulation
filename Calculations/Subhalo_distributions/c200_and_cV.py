import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rc('font', size=20)
plt.rc('axes', titlesize=16)
plt.rc('axes', labelsize=22)
plt.rc('xtick', labelsize=22)
plt.rc('ytick', labelsize=22)
plt.rc('legend', fontsize=18)
plt.rc('figure', titlesize=17)
plt.rc('xtick', top=True, direction='in')
plt.rc('ytick', right=True, direction='in')
plt.rc('xtick.major', size=7, width=1.5, top=True, pad=7)
plt.rc('ytick.major', size=7, width=1.5, right=True, pad=7)
plt.rc('xtick.minor', size=4, width=1)
plt.rc('ytick.minor', size=4, width=1)


def ff(c):
   return np.log(1. + c) - c / (1. + c)


def cv_fromc200(c200):
    return (c200/2.163)**3. * 200. * ff(2.163) / ff(c200)

def gaussian(xx, mean, sigma10, aa):
    return aa * np.exp(-(xx - mean) ** 2. / 2. / sigma10 ** 2.)

def gaussiannorm(xx, mean, sigma10, aa):
    return aa * (1. / (
            10**xx * np.log(10.)*np.sqrt(2.*np.pi)*sigma10)
            * np.exp(-(xx - mean) ** 2. / 2. / sigma10 ** 2.))

def gaussianLOG(xx, mean, sigma10):
    return (1. / (
            10**xx * np.log(10.)*np.sqrt(2.*np.pi)*sigma10)
            * np.exp(-(xx - mean) ** 2. / 2. / sigma10 ** 2.))


def gaussiannormln(xx, mean, sigma10, aa):
    return (aa / (
            10**xx * np.sqrt(2.*np.pi)*sigma10)
            * np.exp(-(xx - mean) ** 2. / 2. / sigma10 ** 2.))


data_c200 = np.loadtxt(
    '../Data_subhalo_simulations/moline17_C200_scatter.txt')

plt.subplots(1, 2)
plt.subplot(121)

plt.scatter(data_c200[:, 0], data_c200[:, 1])

xxx = np.linspace(0.8, 2.2, num=100)

aaa = curve_fit(gaussian, xdata=data_c200[:, 0], ydata=data_c200[:, 1])
print(aaa[0])
plt.plot(xxx, gaussian(xxx, aaa[0][0], aaa[0][1], aaa[0][2]))


aaa = curve_fit(gaussiannorm,
                xdata=data_c200[:, 0], ydata=data_c200[:, 1],
                p0=[1.5, 1., 3.])
print(aaa[0])
plt.plot(xxx, gaussiannorm(xxx, aaa[0][0], aaa[0][1], aaa[0][2]))
plt.plot(xxx, gaussiannorm(xxx, 1.5, 0.15, 1.),
         marker='x')


aaa = curve_fit(gaussianLOG,
                xdata=data_c200[:, 0], ydata=data_c200[:, 1],
                p0=[1.5, 0.15])
print(aaa[0])
plt.plot(xxx, gaussianLOG(xxx, aaa[0][0], aaa[0][1]))


aaa = curve_fit(gaussiannormln,
                xdata=data_c200[:, 0], ydata=data_c200[:, 1],
                p0=[1.5, 1., 3.])
print(aaa[0])
plt.plot(xxx, gaussiannormln(xxx, aaa[0][0], aaa[0][1], aaa[0][2]))


# plt.plot(xxx, gaussiannorm(xxx, 1.5, 0.15, 3.))

# plt.yscale('log')

plt.xlabel('log10(C200)')
plt.text(x=0.8, y=2, s='sigma=%.2f' %aaa[0][1])

mean_cv = cv_fromc200(10**aaa[0][0])
std_cv = np.log10(cv_fromc200(10**np.array(
    [aaa[0][0]-aaa[0][1], aaa[0][0]+aaa[0][1]])))
print(std_cv - np.log10(mean_cv))
print(np.log10(mean_cv))


# ------------------------------------------------------------------
plt.subplot(122)
cv_array = np.log10(cv_fromc200(10**data_c200[:, 0]))
plt.scatter(cv_array, data_c200[:, 1])
aaa = curve_fit(gaussian, xdata=cv_array, ydata=data_c200[:, 1],
                # p0=[]
                )

print(aaa[0])

plt.text(x=4, y=2, s='sigma=%.2f' %aaa[0][1])
xxx = np.linspace(cv_array[0], cv_array[-1], num=100)
plt.plot(xxx, gaussian(xxx, aaa[0][0], aaa[0][1], aaa[0][2]))

plt.xlabel('log10(Cv)')

plt.close('all')
# ------------------------------------------------------------------
data_c200 = np.loadtxt(
    '../Data_subhalo_simulations/moline17_C200_scatterv2.txt')

print('\n\nNew stuff')

plt.subplots(1, 2, figsize=(17, 4))

plt.subplots_adjust(wspace=0, hspace=0)

plt.subplot(121)

meandatax = []
meandatay = []

for i in range(int(len(data_c200[1:-1, 0])/2)):
    aa = (data_c200[2*i+1, 0] + data_c200[2*i+2, 0])/2.
    data_c200[2 * i + 1, 0] = aa
    data_c200[2 * i + 2, 0] = aa

for i in range(int(len(data_c200[1:, 0]) / 2)+1):
    aa = (data_c200[2*i, 1] + data_c200[2*i+1, 1])/2.
    data_c200[2 * i, 1] = aa
    data_c200[2 * i + 1, 1] = aa

for i in range(int(len(data_c200[1:, 0]) / 2)+1):
    aa = (data_c200[2*i, 0] + data_c200[2*i+1, 0])/2.
    meandatax.append(aa)

    aa = (data_c200[2*i, 1] + data_c200[2*i+1, 1])/2.
    meandatay.append(aa)

data_c200 = np.insert(data_c200, 0, [data_c200[0, 0], 0.], axis=0)
data_c200 = np.append(data_c200, [[data_c200[-1, 0], 0.]], axis=0)
plt.plot(data_c200[:, 0], data_c200[:, 1], label='M17 histogram', lw=2)

xxx = np.linspace(0.8, 2.2, num=1000)

aaa = curve_fit(gaussiannorm,
                xdata=meandatax, ydata=meandatay,
                p0=[1.5, 1., 3.])
print(aaa[0])
plt.plot(xxx, gaussiannorm(xxx, aaa[0][0], aaa[0][1], aaa[0][2]),
         label='Our fit, Eq.$\,$(B.2)', lw=2)
errors = np.sqrt(np.diag(aaa[1]))
print(aaa[1])
plt.text(x=1.36, y=0.5,
         s=r'$\mu$' ' = %.2f' r' $\pm$ ' '%.2f'
           '\n'
           r'$\sigma$' ' = %.2f' r' $\pm$ ' '%.2f'
           % (aaa[0][0], errors[0], aaa[0][1], errors[1]),
         fontsize=24)


plt.legend(loc=1)

plt.ylim(0., 2.8)
plt.xlim(data_c200[0, 0]*0.99, data_c200[-1, 0]*1.01)

plt.ylabel(r'N$_\mathrm{sub}$/(N$_\mathrm{t}$ 0.05dex)')
plt.xlabel(r'log$_{10}$(c$_{200}$)')

plt.subplot(122)

plt.tick_params('y', labelleft=False)

cv = np.copy(data_c200)
cv[:, 0] = np.log10(cv_fromc200(10**cv[:, 0]))
meandatax = np.log10(cv_fromc200(10**np.array(meandatax)))

plt.plot(cv[:, 0], cv[:, 1], lw=2,
         label=r'c$_\mathrm{V}$(c$_{200}$), Eq.$\,$(B.1)')

xxx = np.linspace(4., 6.5, num=1000)

aaa = curve_fit(gaussiannorm,
                xdata=meandatax, ydata=meandatay,
                p0=[1.5, 1., 3.])
errors = np.sqrt(np.diag(aaa[1]))
print(aaa[1], errors)
print(aaa[0])
plt.plot(xxx, gaussiannorm(xxx, aaa[0][0], aaa[0][1], aaa[0][2]),
         label='Our fit, Eq.$\,$(B.2)', lw=2)


plt.legend(loc=1)

plt.text(x=4.7, y=0.5,
         s=r'$\mu$' ' = %.2f' r' $\pm$ ' '%.2f'
           '\n'
           r'$\sigma$' ' = %.2f' r' $\pm$ ' '%.2f'
           % (aaa[0][0], errors[0], aaa[0][1], errors[1]),
         fontsize=24)

plt.ylim(0., 2.8)
plt.xlim(cv[0, 0]*0.99, cv[-1, 0]*1.01)

plt.xlabel(r'log$_{10}$(c$_\mathrm{V}$)')

plt.savefig('c200cv.png',
            bbox_inches='tight')
plt.savefig('c200cv.pdf',
            bbox_inches='tight')

plt.show()
