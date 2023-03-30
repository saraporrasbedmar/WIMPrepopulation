# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 11:57:33 2018

@author: ajand
"""

#!/usr/bin/python

"""
changes power law - log par:
    - exp_num_sh
    - Pieri_mass_dist
    - rho_VL
    - rej_samp (max)
    - make_local_data (20 - 6)
    - VLmass (=) VLmass506
"""

import math
import os
import numpy  as np
import random as rdm
   
import matplotlib.pyplot as plt

from scipy    import integrate
from optparse import OptionParser
import scipy.optimize as opt

plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rc('font',   size=20)
plt.rc('axes',   titlesize=16)
plt.rc('axes',   labelsize=16)
plt.rc('xtick',  labelsize=18)
plt.rc('ytick',  labelsize=18)
plt.rc('legend', fontsize=18)
plt.rc('figure', titlesize=17)
plt.rc('xtick', top=True, direction='in')
plt.rc('ytick', right=True, direction='in')
plt.rc('xtick.major', size=7, width=1.5, top=True)
plt.rc('ytick.major', size=7, width=1.5, right=True)
plt.rc('xtick.minor', size=4, width=1)
plt.rc('ytick.minor', size=4, width=1)

#%%
#############
# FUNCTIONS #
#############

#----------- SUBHALO MASS/VELOCITY FUNCION -----------------------------

def SHVF_Grand2012_DMO(V, mm=-3.9557, bb=5.7313):
    # SubHalo Velocity Function - number of subhs defined by their Vmax. DMO simulation.
    # Grand 2012.07846
    #
    # V - max radial velocity of a bound particle in the subhalo [km/s]
    return 10**bb * V**mm


def SHVF_Grand2012_hydro(V, mm=-3.9313, bb=5.0796):
    # SubHalo Velocity Function - number of subhs defined by their Vmax. DMO simulation.
    # Grand 2012.07846
    #
    # V - max radial velocity of a bound particle in the subhalo [km/s]
    return 10**bb * V**mm





#----------- CONCENTRATIONS --------------------------------------------

def Cv_GrandDMO(Vmax, mm=-0.603, bb=5.498):
    # Concentration based on Grand 2012.07846. DMO simulation.
    return 10**bb * Vmax**mm


def Cv_GrandHydro(Vmax, mm=-0.628, bb=5.242):
    # Concentration based on Grand 2105.04560. Hydro simulation.
    return 10**bb * Vmax**mm


def C_Scatt_DMO(C, sig=.29):
    # Create a scatter in the concentrations from relation Rmax-Vmax in DMo sim
    return np.random.lognormal(np.log(C), sig)


def C_Scatt_Hydro(C, sig=.22):
    # Create a scatter in the concentrations from relation Rmax-Vmax in Hydro sim
    return np.random.lognormal(np.log(C), sig)



#----------- J-FACTORS -----275.027 * 0.7**2--------------------------------------------

def J_abs_vel(V, D_earth, C):
    # Jfactor enclosing whole subhalo as a function of the Vmax of the subhalo
    #
    # V       - maximum velocity inside the subhalos [km/s]
    # D_Earth - distancia centro subhalo-Tierra [kpc]
    # C       - concentracion del subhalo
    return (2.163**3. / D_earth**2. /ff(2.163)**2
            *H_0 /12/np.pi/G**2
            * np.sqrt(C/2) * V**3)


def Js_vel(V, D_earth, C):
    # Jfactor enclosing the subhalo up to rs as a function of the Vmax of the subhalo
    #
    # V       - maximum velocity inside the subhalos [km/s]
    # D_Earth - distancia centro subhalo-Tierra [kpc]
    # C       - concentracion del subhalo
    return (2.163**3. / D_earth**2. /ff(2.163)**2
            *H_0 /12/np.pi/G**2
            * np.sqrt(C/2) * V**3 * 7/8)


def J03_vel(V, D_earth, C, ctes_Rmax):
    # Jfactor enclosing the subhalo up to 0.3 degrees as a function of the Vmax of the subhalo
    #
    # V       - maximum velocity inside the subhalos [km/s]
    # D_Earth - distancia centro subhalo-Tierra [kpc]
    # C       - concentracion del subhalo
    return (2.163**3. / D_earth**2. /ff(2.163)**2
            *H_0 /12/np.pi/G**2
            * np.sqrt(C/2) * V**3 * 
            (1 - 1/(1 + 2.163*D_earth * np.tan(0.15*np.pi/180.) / R_max(V, ctes_Rmax))**3))



#----------- MISCELLANEAN DEFINITIONS -------------------------------------

def R_Cut(M, **kwargs):
    # Max radial dist. from Earth at which subhalo of mass M or Vmax V might be observed
    # NOTE: fraction of luminosity of Draco used as cutoff is R = .1
    
    # este 0.1 es la distancia a la que estaría el subhalo, 0.1 kpc, 
    #del centro de la galaxia. Es muy pequeña, lo cual nos da una C mayor
    # (subhalos cerca del centro están más concentrados)
    #y por tanto un R_Cut mayor, con lo que repoblamos una región más grande 
    # (es decir, estamos siendo conservadoras en sentido
    #de no dejarnos posibles subhalos relevantes sin simular)
    

    M = Mtidal_fromVmax(M)
    C = Cm_Mol16(M, 0.1, **kwargs) # TODO: try with boost and/or upper scatter
    return ((M * D_D**2 * C**3 * ff(C_D_corr)**2) / (.1 * ff(C)**2 * M_D * C_D_corr**3))**.5
 
        
def ff(c):
    return np.log(1.+c) - c/(1.+c)




#----------- ROCHE CUT -------------------------------------

def R_s(M, ctes_RmaxVmax):
    # Para Roche al menos
    return R_max(M, ctes_RmaxVmax)/2.163


def R_t(M, ctes_RmaxVmax, Cv, DistGC):
    # 1603.04057 King radius pg 14
    # Para Roche al menos
    Rmax = R_max(M, ctes_RmaxVmax)
    c200 = C200_from_Cv(Cv)
    M = mass_from_Vmax(M, Rmax, c200)
    return ((M / (3 * M_Cont(DistGC)))**(1./3)) * DistGC


def M_Cont(R):
    # 1603.04057 pg 14 la explican: masa del halo hasta ese radio
    # Para Roche al menos
    return 4 * math.pi * rho_0 * r_s_MW**3 * NFW_int(R, r_s=r_s_MW)


def NFW_int(R, r_s=56.7728/2.163): # NOTE: proportionality constants excluded
    return np.log((r_s + R)/r_s) - R/(r_s + R)



# Finding NFW mass from Vmax -----------------------------------------

def R_max(Vmax, VmaxRmax_relat):
    return Vmax**VmaxRmax_relat[0] * 10**VmaxRmax_relat[1]


def mass_from_Vmax (Vmax, Rmax, c200):
    # From Moline16
    return Vmax**2  * Rmax *3.086e16 *1e9 / 6.6741e-11 / 1.989e30 * ff(c200) / ff(2.163)


def def_Cv (c200, Cv):
    return 200 * ff(2.163) / ff(c200) * (c200/2.163)**3 - Cv


def C200_from_Cv(Cv):

    C200_med = np.zeros(1)#len(Cv))
    
    for i in range(1):
        C200_med[i] = opt.root(def_Cv, 40, args=Cv)['x'][0]
    
    return C200_med

#----------- REPOPULATION -----------------------------------------

def N_Dgc_Cosmic(R, R0, aa, bb):
    # Number of subhalos (NOT DM DENSITY) at a distance to the GC R.
    return (R/R0)**aa*np.exp(-bb*(R - R0)/R0)


def Nr_Ntot_resilient_DMO(dist_gc):
    # Number of subhalos at a certain distance of the GC. DMO arguments.
    if resilient:
        args=[1071.47476, 0.38394437, 2.31235586]
    else:
        args=[1011.38716, 0.4037927,  2.35522213]
    yy = N_Dgc_Cosmic(dist_gc, args[0], args[1], args[2])
    
    if resilient==False:
        return 10**yy * (dist_gc >= 6.6176)
    else:
        return 10**yy


def Nr_Ntot_resilient_hydro(dist_gc):
    # Number of subhalos at a certain distance of the GC. Hydro arguments.
    if resilient:
        args=[898.1587, 0.55858766, 2.53959436]
    else:
        args=[666.49179, 0.75291017, 2.90546523]
    yy = N_Dgc_Cosmic(dist_gc, args[0], args[1], args[2])
    
    if resilient==False:
        return 10**yy * (dist_gc >= 13.5905)
    else:
        return 10**yy



def rho_host(R):
    return rho_VL(R)


def ab_int(R):
    return integrate.quad(lambda x: rho_host(x), 0, R)[0]


def dist_frac(Rcut, intg=ab_int): # NOTE: using NFW radial distribution
    # Fraction of subhalos in the distance range Rcut
    if Rcut < 8.5:
        return (intg(8.5 + Rcut) - intg(8.5 - Rcut)) / intg(R_vir)
    elif Rcut + 8.5 >= R_vir: return 1
    else: return intg(8.5 + Rcut) / intg(R_vir)


def rej_samp(xmin, xmax, pdf, num_subhalos, mass=1):
    results = []
 
    # Compute the maximum of the function
    fn_max = np.max(pdf(np.linspace(xmin, xmax, num=1000)))

    while (len(results) < num_subhalos):
        x = rdm.uniform(xmin, xmax)
        h = rdm.uniform(0, fn_max)

        if h < pdf(x):
            results.append(x)

    return results


def exp_num_sh(M1, M2, Rcut, mass_pdf, dist_cut=True):
    # Expected number of subhalos between M1 and M2, R1 and R2
    # (rounded down to nearest int)
    if dist_cut:
        if printt:
            print ('     Dist frac:', dist_frac(Rcut))
        return math.floor(dist_frac(Rcut)*integrate.quad(mass_pdf, M1, M2)[0])
    else:
        if printt:
            print ('     WARNING: no detectability distance cut applied')
        return math.floor(integrate.quad(mass_pdf, M1, M2)[0])


def make_local_data(M1, M2, mass_pdf, rad_pdf, dist_cut=True):
    Rcut = 10#R_Cut(M2)
    if Rcut > 8.5:
        dist_cut=False
    num_subhalos = exp_num_sh(M1, M2, Rcut, mass_pdf, dist_cut=dist_cut)
    samp_ms = rej_samp(M1, M2, mass_pdf, num_subhalos=num_subhalos)
    if printt:
        print ('     Total subhalos generated in this range: ', num_subhalos)
        print ('     Cutting at: %.2f kpc' % Rcut)
    
    #if Rcut < 8.5:
    if dist_cut:
        samp_rs = rej_samp(8.5 - Rcut, 8.5 + Rcut, rad_pdf, num_subhalos=num_subhalos, mass=0)
    else:
        samp_rs = rej_samp(0, R_vir, rad_pdf, num_subhalos=num_subhalos, mass=0)
    return samp_ms, samp_rs


def local_population(mmin, mmax, mass_pdf, rad_pdf, inc_factor=2, dist_cut=True):
    m = mmin
    ms, rs = [], []
    while (m*inc_factor < mmax):
        if printt:
            print()
            print ('Populating subhalos between %.2f - %.2f %s' %(m, m*inc_factor, RangeUnit))
        partial_ms, partial_rs = make_local_data(m, m*inc_factor, mass_pdf, rad_pdf, dist_cut=dist_cut)
        ms.extend(partial_ms)
        rs.extend(partial_rs)
        m *= inc_factor
    
    if printt:
        print()
        print ('Populating subhalos between %.2f - %.2f %s' %(m, mmax, RangeUnit))
    partial_ms, partial_rs = make_local_data(m, mmax, mass_pdf, rad_pdf, dist_cut=dist_cut)
    ms.extend(partial_ms)
    rs.extend(partial_rs)
    return np.array(ms), np.array(rs)







#%%
#################
# PARSE OPTIONS #
#################


usage = 'Usage: %prog [options]'
description = 'This program populates the region around the earth with low-mass dark matter subhalos.'
parser = OptionParser(usage=usage, description=description)

##NUEVO
rutaVLII = '/home/saraporras/Desktop/TFM/Repopulation_codes/newVLtable.txt'
parser.add_option('-i','--infile', default = rutaVLII,
                  help = 'select parent simulation', dest='infile')


(opts, arguments) = parser.parse_args()
are_args = len(arguments)

if (not are_args):
    print ()
    parser.print_help()


#################
# SET CONSTANTS #
#################



### Simulation-Specific Constants ###

VL = np.loadtxt(opts.infile)

if opts.infile == rutaVLII:

    ### ~~ VL II ~~ ###
    Rmax_halo = 56.7728 # from VLII website
    Vmax = 201.033 # from VLII website
#    R_vir = 402. # from Pieri
    R_vir = 220
    M_tot = 1.05e12 # from Pieri (mass within R_vir). M_halo is 1.93596e12 (from Pieri, VLII website)
    alpha = 1.9 #lo que me ha salido del ajuste
    C1 = (alpha-1)*2.11e9  #2.26e9 #lo que me ha salido del ajuste
#    rad_dist = rho_VL
    rho_0 = 8.1e6 
    r_s_MW = 21.

elif opts.infile == 'AQdata/AqA_NoMainHalo.txt':

    ### ~~ AQA ~~ ###
    Rmax_halo = 28.35 # from AQ website
    Vmax = 208.75 # from AQ website
    R_vir = 433 # from Pieri
    M_tot = 4.2e11 # from Pieri (mass within R_vir). M_halo is 1.93596e12 (from Pieri, VLII website)
    C1 = 4.15e10 # TODO: calculate!
#    rad_dist = rho_Ein
    alpha = 1.9
    rho_0 = 2.8e6
    r_s_MW = 20

else:

    print ('Unrecognized file. Please modify script to specify constants.')
    exit(1)





### General Constants ###

G = 4.297e-6 # gravitational constant
H_0 = 71.1/1000 # Hubble constant

# Draco constants
D_D = 80. # D_GC of Draco
M_D = 2.0e8 # mass of Draco
C_D = 19 # concentration of Draco
#C_D_corr = Cm_Mol16(M_D,D_D)
#correccion nueva de Draco
rho_crit = 0.7 * 0.7 * 277 #NUEVO:  densidad crítica, Msun/kpc^3





# REPOPULATION TOOLS -------------------------------------------
its = 1
resilient = False
hydro = False
brightest = 1
printt = True

path_name = 'Resprueba_'+str(resilient)

# Min and max velocities to repopulate
RangeMin, RangeMax = 2., 7.
RangeUnit = 'km/s (Vmax)'

VmaxRmax_Mean_dmo = [1.302, -1.444]  # DMO median
VmaxRmax_80pc_dmo = [1.302, -1.245] # DMO 80 percent

VmaxRmax_Mean_hyd = [1.314, -1.315]  # Hydro median
VmaxRmax_80pc_hyd = [1.314, -1.205]  # Hydro 80 percent



print()
print('Max number of repop subh in DMO: %e' %integrate.quad(SHVF_Grand2012_DMO, RangeMin, RangeMax)[0])
    

    
    
#%%
###################
# RUN COMPUTATION #
###################
headerS = (('#\n# Resilient: '+str( resilient) + '; ' + str(its) + 
          (' iterations \n# Js (<r_s) (GeV^2 cm^-5)'
          '       Dgc (kpc)      D_Earth (kpc)'
          '                   Vmax (km/s)                   ang size (deg)'
          '               Cv \n#\n')))



header03 = (('#\n# Resilient: '+str( resilient) + '; ' + str(its) + 
          (' iterations \n# J03 (<0.3deg) (GeV^2 cm^-5)'
          '       Dgc (kpc)      D_Earth (kpc)'
          '                   Vmax (km/s)                   ang size (deg)'
          '               Cv \n#\n')))
          


if not os.path.exists(path_name):
    os.makedirs(path_name)

if hydro:
    file_Js_hyd = open(path_name + '/Js_Hydro_results.txt', 'w')
    file_J03_hyd = open(path_name + '/J03_Hydro_results.txt', 'w')
    
    file_Js_hyd.write(headerS)
    file_J03_hyd.write(header03)


else:
    file_Js_dmo = open(path_name + '/Js_DMO_results.txt', 'w')
    file_J03_dmo = open(path_name + '/J03_DMO_results.txt', 'w')

    file_Js_dmo.write(headerS)
    file_J03_dmo.write(header03)



          
def cosa_computar ():         
    for it in range(its):
        print()
        print('Iteration ', it)
        
    
        if hydro:
        
            # Repopulation happens, you firstly obtain masses and distances to GC
            repop_Vmax, repop_DistShGc = local_population(RangeMin, RangeMax, 
                                                          SHVF_Grand2012_hydro, Nr_Ntot_resilient_hydro,
                                                          dist_cut=False, inc_factor=2)
            numberSH_woRoche = len(repop_Vmax)
            if printt:
                print()
                print('Number of repopulated subhalos: ', numberSH_woRoche)
         
        
        
            
            if printt:
                print()
                print('Calculating distances to the Earth....')
            # Random distribution of subhalos around the celestial sphere
            repop_Theta  = 2 * math.pi * np.random.random(len(repop_Vmax))
            repop_phis   =     math.pi * np.random.random(len(repop_Vmax))
            
            
            # Positions of th subhalos
            repop_Xs, repop_Ys, repop_Zs = (repop_DistShGc * np.cos(repop_Theta) * np.sin(repop_phis), 
                                            repop_DistShGc * np.sin(repop_Theta) * np.sin(repop_phis),
                                            repop_DistShGc * np.cos(repop_phis))
            
            repop_DistShEarth = ((repop_Xs - 8.5)**2 + repop_Ys**2 + repop_Zs**2)**(.5)
            
#            del (repop_Xs,     repop_Ys,     repop_Zs,
#                 repop_Theta, repop_phis)
        
            if printt:
                print()
                print('Calculating concentrations....')
            
            repop_C = Cv_GrandHydro(repop_Vmax)
            repop_C = C_Scatt_Hydro(repop_C)
            
                
            if printt:
                print()
                print('Calculating J-factors....')
        
            repop_Js     = ((3.241*10**(-22))**5*(1.12*10**57)**2 * 
                          Js_vel(repop_Vmax, repop_DistShEarth, repop_C))
            
            
            repop_J03     = ((3.241*10**(-22))**5*(1.12*10**57)**2 * 
                          J03_vel(repop_Vmax, repop_DistShEarth,
                                         repop_C, VmaxRmax_Mean_hyd))
            
            
            # Angle of subhalos
            repop_Theta     = 180 / np.pi * np.arctan(R_s(repop_Vmax,     VmaxRmax_Mean_hyd)/repop_DistShEarth)
            
            bright_Js = np.where(np.max(repop_Js)==repop_Js)[0][0]
            
            while (R_t(repop_Vmax[bright_Js], VmaxRmax_Mean_hyd, repop_C[bright_Js], repop_DistShGc[bright_Js])
                                < R_s(repop_Vmax[bright_Js], VmaxRmax_Mean_hyd)):
                repop_Js[bright_Js] = 0.
                bright_Js = np.where(np.max(repop_Js)==repop_Js)[0][0]
                print('Roche')
                
            np.savetxt(file_Js_hyd, np.column_stack((repop_Js [bright_Js],
                                               repop_DistShGc [bright_Js], 
                                               repop_DistShEarth [bright_Js],
                                               repop_Vmax [bright_Js], 
                                               repop_Theta [bright_Js], 
                                               repop_C [bright_Js])))
            
            
            bright_Js = np.where(np.max(repop_J03)==repop_J03)
            while (R_t(repop_Vmax[bright_Js], VmaxRmax_Mean_hyd, repop_C[bright_Js], repop_DistShGc[bright_Js])
                                < R_s(repop_Vmax[bright_Js], VmaxRmax_Mean_hyd)):
                repop_J03[bright_Js] = 0.
                bright_Js = np.where(np.max(repop_J03)==repop_J03)[0][0]
                print('Roche')
    
            np.savetxt(file_J03_hyd, np.column_stack((repop_J03 [bright_Js],
                                               repop_DistShGc [bright_Js], 
                                               repop_DistShEarth [bright_Js],
                                               repop_Vmax [bright_Js], 
                                               repop_Theta [bright_Js], 
                                               repop_C [bright_Js])))
        
        
        
        
        
        # ---------- DMO -------------------------------------------------------------------------------
        else:
            repop_Vmax_DMO, repop_DistShGc_DMO = local_population(RangeMin, RangeMax,
                                                                  SHVF_Grand2012_DMO, Nr_Ntot_resilient_DMO,
                                                                  dist_cut=False, inc_factor=2)
            
            
            repop_Theta  = 2 * math.pi * np.random.random(len(repop_Vmax_DMO))
            repop_phis   =     math.pi * np.random.random(len(repop_Vmax_DMO))
            
            
            repop_Xs_DMO, repop_Ys_DMO, repop_Zs_DMO = (repop_DistShGc_DMO * np.cos(repop_Theta) * np.sin(repop_phis), 
                                            repop_DistShGc_DMO * np.sin(repop_Theta) * np.sin(repop_phis),
                                            repop_DistShGc_DMO * np.cos(repop_phis))
            
            repop_DistShEarth_DMO = ((repop_Xs_DMO - 8.5)**2 + repop_Ys_DMO**2 + repop_Zs_DMO**2)**(.5)
            
#            del (repop_Xs_DMO, repop_Ys_DMO, repop_Zs_DMO,
#                 repop_Theta, repop_phis)
            
        
            
            repop_C_DMO = Cv_GrandDMO(repop_Vmax_DMO)
            repop_C_DMO = C_Scatt_DMO(repop_C_DMO)
        
            
            repop_Js_DMO = ((3.241*10**(-22))**5*(1.12*10**57)**2 * 
                          Js_vel(repop_Vmax_DMO, repop_DistShEarth_DMO, repop_C_DMO))
        
            
            repop_J03_DMO = ((3.241*10**(-22))**5*(1.12*10**57)**2 * 
                          J03_vel(repop_Vmax_DMO, repop_DistShEarth_DMO,
                                         repop_C_DMO, VmaxRmax_Mean_dmo))
        
            
            repop_Theta_DMO = 180 / np.pi * np.arctan(R_s(repop_Vmax_DMO, VmaxRmax_Mean_dmo)/repop_DistShEarth_DMO)
            
        
            
            bright_Js_dmo = np.where(np.max(repop_Js_DMO)==repop_Js_DMO)
            while (R_t(repop_Vmax_DMO[bright_Js_dmo], VmaxRmax_Mean_dmo, repop_C_DMO[bright_Js_dmo],
                       repop_DistShGc_DMO[bright_Js_dmo]) < R_s(repop_Vmax_DMO[bright_Js_dmo], VmaxRmax_Mean_dmo)):
                repop_Js_DMO[bright_Js_dmo] = 0.
                bright_Js_dmo = np.where(np.max(repop_Js_DMO)==repop_Js_DMO)[0][0]
                print('Roche')
            
            np.savetxt(file_Js_dmo, np.column_stack((repop_Js_DMO [bright_Js_dmo],
                                               repop_DistShGc_DMO [bright_Js_dmo],
                                               repop_DistShEarth_DMO [bright_Js_dmo],
                                               repop_Vmax_DMO [bright_Js_dmo], 
                                               repop_Theta_DMO [bright_Js_dmo],
                                               repop_C_DMO [bright_Js_dmo])))
        
        
            bright_Js_dmo = np.where(np.max(repop_J03_DMO)==repop_J03_DMO)
            while (R_t(repop_Vmax_DMO[bright_Js_dmo], VmaxRmax_Mean_dmo, repop_C_DMO[bright_Js_dmo],
                       repop_DistShGc_DMO[bright_Js_dmo]) < R_s(repop_Vmax_DMO[bright_Js_dmo], VmaxRmax_Mean_dmo)):
                repop_J03_DMO[bright_Js_dmo] = 0.
                bright_Js_dmo = np.where(np.max(repop_J03_DMO)==repop_J03_DMO)[0][0]
                print('Roche')
                  
            np.savetxt(file_J03_dmo, np.column_stack((repop_J03_DMO [bright_Js_dmo],
                                               repop_DistShGc_DMO [bright_Js_dmo],
                                               repop_DistShEarth_DMO [bright_Js_dmo],
                                               repop_Vmax_DMO [bright_Js_dmo], 
                                               repop_Theta_DMO [bright_Js_dmo],
                                               repop_C_DMO [bright_Js_dmo])))

cosa_computar()

if hydro:
    file_Js_hyd.close()
    file_J03_hyd.close()
    
else:
    file_Js_dmo.close()
    file_J03_dmo.close()


