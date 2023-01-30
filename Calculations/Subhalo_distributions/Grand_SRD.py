#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 20:37:40 2022

@author: saraporras
"""

import numpy as np
import matplotlib.pyplot as plt

import scipy.optimize as opt
from scipy    import integrate
from matplotlib.lines import Line2D



plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rc('font',   size=20)
plt.rc('axes',   titlesize=16)
plt.rc('axes',   labelsize=16)
plt.rc('xtick',  labelsize=21)
plt.rc('ytick',  labelsize=22)
plt.rc('legend', fontsize=20)
plt.rc('figure', titlesize=17)
plt.rc('xtick', top=True, direction='in')
plt.rc('ytick', right=True, direction='in')
plt.rc('xtick.major', size=7, width=1.5, top=True)
plt.rc('ytick.major', size=7, width=1.5, right=True)
plt.rc('xtick.minor', size=4, width=1)
plt.rc('ytick.minor', size=4, width=1)


#        Rmax[kpc]        Vmax[km/s]      Radius[Mpc]
Grand_dmo = np.loadtxt('../Data_subhalos_simulations/RmaxVmaxRadDMO0_1.txt')
Grand_hydro = np.loadtxt('../Data_subhalos_simulations/RmaxVmaxRadFP0_1.txt')



#Grand_hydro = Grand_hydro[Grand_hydro[:,1]>1e-4, :]
Grand_hydro = Grand_hydro[Grand_hydro[:,1]>np.min(Grand_dmo[:,1]), :]

#Grand_dmo[:,2] *= 1e3
#Grand_hydro[:,2] *= 1e3

Grand_dmo = Grand_dmo[np.argsort(Grand_dmo[:,2])]
Grand_hydro = Grand_hydro[np.argsort(Grand_hydro[:,2])]

# Los mayores Dist_Gc son 264 kpc, pero entonces cuál es R200?
R200 = 0.265 # kpc
R_max = 0.220

#%% SRD

def encontrar_SRD_sinVol (data):
    n_final = []
    a=0
    for delta in range(len(bins)-2):
        X1limit = np.where(data[:,2]>=bins[delta])[0][0]
        X2limit = np.where(data[:,2]>bins[delta+1])[0][0]
        
        y = X2limit-X1limit
#        vol = 4/3 * np.pi * (bins[delta+1]**3 - bins[delta]**3)# * 1e9

        n_final.append(y)#/vol)
        a += y
#        print(delta, X2limit, X1limit)
        
    y = len(data)-X2limit
    n_final.append(y)#/(4/3 * np.pi * (bins[-1]**3 - bins[-2]**3)))

    a += y
    print(a)
    
    return n_final   

   
def encontrar_SRD (data):
    n_final = []
    a=0
    for delta in range(len(bins)-2):
        X1limit = np.where(data[:,2]>=bins[delta])[0][0]
        X2limit = np.where(data[:,2]>bins[delta+1])[0][0]
        
        y = X2limit-X1limit
        vol = 4/3 * np.pi * (bins[delta+1]**3 - bins[delta]**3)# * 1e9

        n_final.append(y/vol)
        a += y
#        print(delta, X2limit, X1limit)
        
    y = len(data)-X2limit
    n_final.append(y/(4/3 * np.pi * (bins[-1]**3 - bins[-2]**3)))

    a += y
    print(a)
    
    return n_final     



def N_Dgc_Cosmic_old(R, R0, aa, bb, amplitude):
    # Number of subhalos (NOT DM DENSITY) at a distance to the GC R.
    return (R/R0)**aa*np.exp(-bb*(R - R0)/R0) * amplitude
def N_Dgc_Cosmic(R, R0, aa, bb):
    # Number of subhalos (NOT DM DENSITY) at a distance to the GC R.
    return (R/R0)**aa*np.exp(-bb*(R - R0)/R0)

def resilient(R, yy_args):
    yy = ((R<=yy_args[0][0]*yy_args[0][1]/yy_args[0][2]) * 
          N_Dgc_Cosmic(yy_args[0][0]*yy_args[0][1]/yy_args[0][2], yy_args[0][0], yy_args[0][1], yy_args[0][2])
        + (R> yy_args[0][0]*yy_args[0][1]/yy_args[0][2]) * 
        N_Dgc_Cosmic(R, yy_args[0][0], yy_args[0][1], yy_args[0][2]))
        
    return yy

def fragile(R, yy_args, x_cut):
    yy = N_Dgc_Cosmic(R, yy_args[0][0], yy_args[0][1], yy_args[0][2]) * (R >= x_cut)
        
    return yy

#def multiplicar_por_cascaron_esf(xxx, data):
#    for cosa in range(len(xxx)-1):
#        vol = (xxx[cosa+1]**3-xxx[cosa]**3) * 4/3.*np.pi
#        data[cosa] *= vol
#    data[-1] *= (4/3.*np.pi)
#    return data

#---------------------------------------------------------------------------------------------------


plt.close('all')
plt.figure(figsize=(12,7))
num_bins = 15


bins = np.linspace(0, R200, num=num_bins)
x_med = (bins[:-1]+bins[1:])/2.

srd_dmo = np.array(encontrar_SRD(Grand_dmo))/len(Grand_dmo)
srd_hydro = np.array(encontrar_SRD(Grand_hydro))/len(Grand_hydro)


plt.plot(x_med*1e3, srd_dmo, '-', color='k')
plt.plot(x_med*1e3, srd_dmo, label='Data', color='k', marker='.', ms=10)
plt.plot(x_med*1e3, srd_hydro, '-', color='limegreen')
plt.plot(x_med*1e3, srd_hydro, color='limegreen', marker='.', ms=10)



xxx = np.linspace(3e-3, R200*1e3, num=3000)



xmin = 2
xmax = 11
linear_tend_dmo = np.polyfit(np.log10(x_med[xmin:xmax]*1e3), srd_dmo[xmin:xmax], 1)
print('max res dmo')
print(linear_tend_dmo)

xmin = 5
xmax = 11
linear_tend_hydro = np.polyfit(np.log10(x_med[xmin:xmax]*1e3), srd_hydro[xmin:xmax], 1)
print('max res hydro')
print(linear_tend_hydro)

plt.plot(xxx, np.log10(xxx) * linear_tend_dmo[0] + linear_tend_dmo[1], '-', color='grey', alpha=0.7, label='res_maximum')
plt.plot(xxx, np.log10(xxx) * linear_tend_hydro[0] + linear_tend_hydro[1], '-', color='limegreen', alpha=0.7)



xxx = np.linspace(3e-3, R200, num=3000)
X_max = np.where(x_med>=R_max)[0][0]

yy_dmo = opt.curve_fit(N_Dgc_Cosmic, x_med[:X_max], srd_dmo[:X_max], p0=[0.5, 0.5, 5])
peak_dmo = yy_dmo[0][0]*1e3*yy_dmo[0][1]/yy_dmo[0][2]
plt.plot(xxx*1e3, resilient(xxx, yy_dmo), 'k', linestyle='dashdot', alpha=0.7, label='Resilient')
plt.plot(xxx*1e3, fragile(xxx, yy_dmo, np.min(Grand_dmo[:,2])), 'k', alpha=0.7, label='Fragile', linestyle=':')
#plt.plot(x_med*1e3, resilient(x_med, yy_dmo), 'xk', ms=10, alpha=0.7)
plt.axvline(peak_dmo, alpha=0.5, color='k')
print('DMO:  ', yy_dmo[0])


yy_hyd = opt.curve_fit(N_Dgc_Cosmic, x_med[:X_max], srd_hydro[:X_max], p0=[0.5, 0.5, 5])
peak_hyd = yy_hyd[0][0]*1e3*yy_hyd[0][1]/yy_hyd[0][2]
plt.plot(xxx*1e3, resilient(xxx, yy_hyd), linestyle='dashdot', alpha=0.7, color='limegreen')
plt.plot(xxx*1e3, fragile(xxx, yy_hyd, np.min(Grand_hydro[:,2])), alpha=0.7, color='limegreen', ls=':')
plt.axvline(peak_hyd, alpha=0.5, color='limegreen')
#plt.plot(x_med*1e3, resilient(x_med, yy_hyd), 'x', ms=10, alpha=0.7, color='limegreen')
print('Hydro:', yy_hyd[0])


plt.axvline(R_max*1e3, alpha=0.7, linestyle='--')#, label='220 kpc')
plt.annotate(r'R$_\mathrm{vir}$', (170, 32), color='b', rotation=45, alpha=0.7)

plt.axvline(8.5, linestyle='--', alpha=1, color='Sandybrown')
plt.annotate('Earth', (8.6, 35), color='Sandybrown', rotation=45)

plt.ylabel(r'n(r) = $\frac{N(r)}{N_{Tot}\,Volumen}$ [Mpc$^{-3}$]', size=24)
plt.xlabel('r [kpc]', size=26)

legend_elements = [Line2D([0], [0], marker='o', color='w', label='Frag',
                          markerfacecolor='k', markersize=8),
                    Line2D([0], [0], marker='o', color='w', label='Res',
                          markerfacecolor='limegreen', markersize=8)]
legend1 = plt.legend(legend_elements, ['DMO', 'Hydro'], loc=8, handletextpad=0.2)#,handlelength=1)
leg = plt.legend(framealpha=1, loc=1)
plt.gca().add_artist(legend1) 

plt.xscale('log')
plt.xlim(6,300)

plt.ylim(0, 60)

plt.savefig('SRD_density')



#------------------------ N(r)/Ntot figure ------------------------------

srd_dmo_sinVol = np.array(encontrar_SRD_sinVol(Grand_dmo))/len(Grand_dmo)
srd_hydro_sinVol = np.array(encontrar_SRD_sinVol(Grand_hydro))/len(Grand_hydro)

x_kpc = x_med *1e3

plt.figure(2, figsize=(10,8))

plt.plot(x_kpc, srd_dmo_sinVol, '-', color='k')
plt.plot(x_kpc, srd_dmo_sinVol, color='k', marker='.', ms=10, label='DMO')

plt.plot(x_kpc, srd_hydro_sinVol, '-', color='g')
plt.plot(x_kpc, srd_hydro_sinVol, color='g', marker='.', ms=10, label='Hydro')


volume_array = 4/3.*np.pi * (bins[1:]**3 - bins[:-1]**3)

xmin = 6
xmax = 11
linear_tend_dmo = np.polyfit(np.log10(x_kpc[xmin:xmax]), np.log10(srd_dmo_sinVol[xmin:xmax]), 1)
print('max res dmo')
print(linear_tend_dmo)

xmin = 6
xmax = 11
linear_tend_hydro = np.polyfit(np.log10(x_kpc[xmin:xmax]), np.log10(srd_hydro_sinVol[xmin:xmax]), 1)
print('max res hydro')
print(linear_tend_hydro)

plt.plot(x_kpc, x_kpc**linear_tend_dmo[0] * 10**linear_tend_dmo[1], '-', color='grey', alpha=0.7, label='res_maximum')
plt.plot(x_kpc, x_kpc**linear_tend_hydro[0] * 10**linear_tend_hydro[1], '-', color='limegreen', alpha=0.7)



#plt.axvline(peak_dmo, alpha=0.5, color='k')
#plt.axvline(peak_hyd, alpha=0.5, color='g')
plt.axvline(R_max*1e3, alpha=0.5, label='220 kpc')


plt.ylabel(r'n(r) = $\frac{N(r)}{N_{Tot}}$')
plt.xlabel('r [kpc]', size=24)

plt.legend(framealpha=1)

plt.xscale('log')
plt.yscale('log')


print()
print('Place of [Mpc] and number of subs below the resilient-disrupted cut:')
print('%.5f ----- %r' %(yy_dmo[0][0]*yy_dmo[0][1]/yy_dmo[0][2], 
      np.shape(Grand_dmo  [Grand_dmo  [:,2]<=peak_dmo/1e3, :])[0]))
print('%.5f ----- %r' %(yy_hyd[0][0]*yy_hyd[0][1]/yy_hyd[0][2], 
      np.shape(Grand_hydro[Grand_hydro[:,2]<=peak_hyd/1e3, :])[0]))


#%%
plt.figure(figsize=(12,9))

xxx = np.logspace(-10, np.log10(R200), num=1000)

yy_resDMO = np.array(srd_dmo_sinVol, dtype=float)
X_lim = np.where(x_med >= peak_dmo*1e-3)[0][0]
yy_resDMO[:X_lim] = (resilient(x_med, yy_dmo)*len(Grand_dmo)*volume_array)[:X_lim]

yy_res = np.array(srd_hydro_sinVol, dtype=float)
X_lim = np.where(x_med >= peak_hyd*1e-3)[0][0]
yy_res[:X_lim] = (resilient(x_med, yy_hyd)*len(Grand_hydro)*volume_array)[:X_lim]

yy_resDMO = np.log10(yy_resDMO)
yy_res = np.log10(yy_res)


yy_ultraresilient_dmo = (np.log10(x_med) * linear_tend_dmo[0] + linear_tend_dmo[1]) * len(Grand_dmo) * volume_array
yy_ultraresilient_hydro = (np.log10(x_med) * linear_tend_hydro[0] + linear_tend_hydro[1]) * len(Grand_hydro) * volume_array

yy_ultraresilient_dmo = np.log10(yy_ultraresilient_dmo)
yy_ultraresilient_hydro = np.log10(yy_ultraresilient_hydro)


# --------------------------


plt.plot(x_med*1e3, np.log10(srd_dmo_sinVol), '-', color='k')
plt.plot(x_med*1e3, np.log10(srd_dmo_sinVol), color='k', marker='.', ms=10, label='Frag data')

plt.plot(x_med*1e3, np.log10(srd_hydro_sinVol), '-', color='g')
plt.plot(x_med*1e3, np.log10(srd_hydro_sinVol), color='g', marker='.', ms=10)


yRes_dmo = opt.curve_fit(N_Dgc_Cosmic, x_med[:X_max], yy_resDMO[:X_max], p0=[0.5, 0.5, 2])
plt.plot(x_med*1e3, yy_resDMO, 'xk', ms=10, label='Resil data')
plt.plot(xxx*1e3, N_Dgc_Cosmic(xxx, yRes_dmo[0][0], yRes_dmo[0][1], yRes_dmo[0][2]), '--k', alpha=0.7, label='Resilient')
print(yRes_dmo[0])

yRes_hydro = opt.curve_fit(N_Dgc_Cosmic, x_med[:X_max], yy_res[:X_max], p0=[0.5, 0.5, 2])
plt.plot(x_med*1e3, yy_res, 'xg', ms=10)
plt.plot(xxx*1e3, N_Dgc_Cosmic(xxx, yRes_hydro[0][0], yRes_hydro[0][1], yRes_hydro[0][2]), '--g', alpha=0.7)
print(yRes_hydro[0])
print()


yy_dmo = opt.curve_fit(N_Dgc_Cosmic, x_med[:X_max], np.log10(srd_dmo_sinVol[:X_max]), p0=[0.5, 0.5, 5])
plt.plot(xxx*1e3, N_Dgc_Cosmic(xxx, yy_dmo[0][0], yy_dmo[0][1], yy_dmo[0][2])*(xxx>=np.min(Grand_dmo[:,2])), 
         'k', linestyle='dotted', alpha=0.7, label='Fragile')
print(yy_dmo[0])

yy_hyd = opt.curve_fit(N_Dgc_Cosmic, x_med[:X_max], np.log10(srd_hydro_sinVol[:X_max]), p0=[0.5, 0.5, 5])
plt.plot(xxx*1e3, N_Dgc_Cosmic(xxx, yy_hyd[0][0], yy_hyd[0][1], yy_hyd[0][2])*(xxx>=np.min(Grand_hydro[:,2])), 
         'g', linestyle='dotted', alpha=0.7)
print(yy_hyd[0])


plt.plot(x_med*1e3, yy_ultraresilient_dmo, '+k', ms=10, label='Ultra res')
plt.plot(x_med*1e3, yy_ultraresilient_hydro, '+g', ms=10)


plt.axvline(peak_dmo, alpha=0.5, color='k')
plt.axvline(peak_hyd, alpha=0.5, color='g')
plt.axvline(R_max*1e3, alpha=0.5, label='220 kpc')
plt.axvline(8.5, alpha=0.5, label='Earth position', color='r')


plt.ylabel(r'n(r) = $log10(N(r))$')
plt.xlabel('r [kpc]', size=24)

plt.legend(framealpha=1)
plt.savefig('SRD')

#plt.xscale('log')
#plt.yscale('log')


#%% 
#plt.close('all')
plt.figure(5)
R_vir = 220
def ten_elev(xx, rr, aa, bb):
    return 10**N_Dgc_Cosmic(xx, rr, aa, bb)
integral_dmo = integrate.quad(ten_elev, 0, R_vir, args=(yRes_dmo[0][0]*1e3, yRes_dmo[0][1], yRes_dmo[0][2]))[0]
plt.plot(x_med*1e3, 10**yy_resDMO/integral_dmo, 'xk', ms=10, label='Resil data')
plt.plot(xxx*1e3, (10**N_Dgc_Cosmic(xxx*1e3, yRes_dmo[0][0]*1e3, yRes_dmo[0][1], yRes_dmo[0][2])/integral_dmo),
        '--k', alpha=0.7)


integral_hyd = integrate.quad(ten_elev, 0, R_vir, args=(yRes_hydro[0][0]*1e3, yRes_hydro[0][1], yRes_hydro[0][2]))[0]
plt.plot(x_med*1e3, 10**yy_res/integral_hyd, 'xg', ms=10)
plt.plot(xxx*1e3, (10**N_Dgc_Cosmic(xxx*1e3, yRes_hydro[0][0]*1e3, yRes_hydro[0][1], yRes_hydro[0][2])/integral_hyd),
        '--g', alpha=0.7)

plt.legend()

plt.ylabel(r'$\frac{N(r)}{N_{tot}}$', size=24)
plt.xlabel('r [kpc]', size=24)

plt.axvline(peak_dmo, alpha=0.5, color='k')
plt.axvline(peak_hyd, alpha=0.5, color='g')
plt.axvline(R_vir, alpha=0.5, label='220 kpc')
#%% SRD


#plt.figure()
#n_dmo, bins_dmo, patches = plt.hist(Grand_dmo[:,2], color='k', alpha=0.5, bins=40)
#n_hydro, bins_hydro, patches = plt.hist(Grand_hydro[:,2], color='g', alpha=0.5, bins=40)
#plt.ylabel('N(r)')
#plt.xlabel('r [kpc]')


#plt.figure()
#xxx = np.linspace(0, 100, num=100)
#R0 = 100
#aa = 1
#bb = 50
#plt.plot(xxx, (xxx/R0)**aa, label='alpha')
#plt.plot(xxx, np.exp(-bb*(xxx - R0)/R0)*1e-21, label='exp *1e-21')
#plt.plot(xxx, (xxx/R0)**aa*np.exp(-bb*(xxx - R0)/R0)*1e-28, label='cosmic *1e-28')
#
#plt.annotate('R0=%r\naa=%.2f\nbb=%.2f' %(R0, aa, bb), (40, 2.75))
#
#plt.legend()


def fracc_interior(peak, args):
    return (integrate.quad(ten_elev, 0, peak*1e3, args=(
            args[0]*1e3, args[1], args[2]))[0]
            /integrate.quad(ten_elev, 0, 220, args=(
            args[0]*1e3, args[1], args[2]))[0])

print(fracc_interior(peak_dmo, yRes_dmo[0]))
print(fracc_interior(peak_hyd, yRes_hydro[0]))

print(fracc_interior(peak_dmo, yy_dmo[0]))
print(fracc_interior(peak_hyd, yy_hyd[0]))



plt.ylabel(r'$\frac{N(r)}{N_{tot}}$', size=24)
plt.xlabel('r [kpc]', size=24)

plt.axvline(peak_dmo, alpha=0.5, color='k')
plt.axvline(peak_hyd, alpha=0.5, color='g')
plt.axvline(R_vir, alpha=0.5, label='220 kpc')

plt.show()
