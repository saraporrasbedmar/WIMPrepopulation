import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit



def ff(c):
   return np.log(1. + c) - c / (1. + c)


def cv_fromc200(c200):
    return (c200/2.163)**3. * 200. * ff(2.163) / ff(c200)

def gaussian(xx, mean, sigma10, aa):
    return aa * np.exp(
        -(xx - mean) ** 2. / 2. / sigma10 ** 2.)


data_c200 = np.loadtxt(
    '../Data_subhalo_simulations/moline17_C200_scatter.txt')

plt.subplots(1, 2)
plt.subplot(121)

plt.scatter(data_c200[:, 0], data_c200[:, 1])

aaa = curve_fit(gaussian, xdata=data_c200[:, 0], ydata=data_c200[:, 1])

print(aaa)

xxx = np.linspace(0.8, 2.2, num=100)
plt.plot(xxx, gaussian(xxx, aaa[0][0], aaa[0][1], aaa[0][2]))

plt.xlabel('log10(C200)')
plt.text(x=0.8, y=2, s='sigma=%.2f' %aaa[0][1])

mean_cv = cv_fromc200(10**aaa[0][0])
std_cv = np.log10(cv_fromc200(10**np.array(
    [aaa[0][0]-aaa[0][1], aaa[0][0]+aaa[0][1]])))
print(std_cv - np.log10(mean_cv))
print(np.log10(mean_cv))
print(std_cv, [aaa[0][0]-aaa[0][1], aaa[0][0]+aaa[0][1]])

plt.subplot(122)
cv_array = np.log10(cv_fromc200(10**data_c200[:, 0]))
plt.scatter(cv_array, data_c200[:, 1])
aaa = curve_fit(gaussian, xdata=cv_array, ydata=data_c200[:, 1],
                # p0=[]
                )

print(aaa)

plt.text(x=4, y=2, s='sigma=%.2f' %aaa[0][1])
xxx = np.linspace(cv_array[0], cv_array[-1], num=100)
plt.plot(xxx, gaussian(xxx, aaa[0][0], aaa[0][1], aaa[0][2]))

plt.xlabel('log10(Cv)')

plt.show()