# import autograd.numpy as np
import numpy as np
from autograd import grad

import matplotlib.pyplot as plt # plotting

import scipy.integrate as integ #integration
import scipy.interpolate as interp
import scipy.linalg as sla
from numba import jit, prange

import pycbc
import pycbc.detector as detector
import pycbc.conversions
import pycbc.waveform
from pycbc.waveform import Array

from astropy.cosmology import Planck15 as cosmo
from astropy.cosmology import z_at_value
from pycbc.cosmology import get_cosmology
import astropy.units as units
from astropy import coordinates
from astropy.units import Mpc as Mpc_unit


import time
import lal # this is for unit conversion
from astropy.time import Time

from scipy.optimize import fsolve

# @jit(nopython=True)
def ecliptic_to_equatorial(lambd, beta, **kargs):
    """
    Converts ecliptic (lambda, beta) into equatorial (alpha, delta) coordinates [in radians]
    """
    if kargs['earth-tilt']==True:
        epsilon = 23.4 * np.pi / 180.

        num = np.cos(beta) * np.sin(lambd) * np.cos(epsilon)
        num -= np.sin(beta) * np.sin(epsilon) 
        denom = np.cos(beta) * np.cos(lambd)
        alpha = np.arctan2(num, denom)

        delta = np.sin(beta) * np.cos(epsilon) + np.cos(beta) * np.sin(epsilon) * np.sin(lambd)
        delta = np.arcsin(delta)

        return alpha, delta
    else:
        return lambd, beta
    
@jit(nopython=True)
def L_to_iota_psi(lambd, beta, phiL, thetaL):
    
    cosIota = np.cos(thetaL) * np.sin(beta)
    cosIota += np.sin(thetaL) * np.cos(beta) * np.cos(phiL-lambd)
    iota = np.arccos(cosIota)
    
    tanPsiNum = np.cos(thetaL) + cosIota * np.sin(beta)
    tanPsiDen = np.cos(beta) * np.sin(thetaL) * np.sin(phiL - lambd)
    
    psi = np.arctan2(tanPsiNum, tanPsiDen)

    return iota, psi


def read_mag(freq, fileName):
    f_tf, mag_tf = np.loadtxt(fileName, unpack=True)
    
    idx = np.where(f_tf>0)
    mag_tf = mag_tf[idx]
    f_tf = f_tf[idx]

    mag_func = interp.interp1d(np.log(f_tf), np.log(mag_tf), kind='linear', bounds_error=False, fill_value=np.inf)
    mag_out = np.exp(mag_func(np.log(freq)))
    return mag_out

def get_u_ph(Mc, q, DL, freqs):
    """
    Gets numpy array of u_ph values at freqs
    """

    m1 = pycbc.conversions.mass1_from_mchirp_q(Mc,q)
    m2 = pycbc.conversions.mass2_from_mchirp_q(Mc,q)
    
    freqs_lal = Array(freqs)
    hp, _ = pycbc.waveform.get_fd_waveform_sequence(approximant = "IMRPhenomD", mass1 = m1, mass2 = m2, distance = DL, sample_points = freqs_lal) #dont specify iota or polarization angle
    u = np.array(hp)
    
    return u


def get_tau_a(det, lambd, beta, t_gps, **kargs):
    
    ra, dec = ecliptic_to_equatorial(lambd, beta, **kargs)
    detectorToEarth = det.time_delay_from_earth_center(ra, dec, t_gps) # t_det - t_earth
    
    if kargs['barycenter'] == 'earth':
        detectorToBary = detectorToEarth
    elif kargs['barycenter'] == 'sun':
        Rs = lal.AU_SI
        sl = lal.C_SI
        t_utc = Time(val=t_gps, format='gps', scale='utc')
        earth = coordinates.get_body('earth', t_utc, location=None).transform_to('barycentricmeanecliptic')
    
        earthToBary = - Rs / sl # t_earth - t_baryc
        earthToBary *= np.cos(beta)
        earthToBary *= np.cos(earth.lon.rad - lambd)
        detectorToBary = detectorToEarth + earthToBary
    else:
        print('error barycenter arg not set')

    return detectorToBary


def get_phi_d_a(det, freq, lambd, beta, t_gps, **kargs):
    phid = - 2 * np.pi * freq * get_tau_a(det, lambd, beta, t_gps, **kargs)
    return phid
    
# @jit(nopython=True)
def get_Lamb_amp(det, t_gps, lambd, beta, phiL, thetaL, **kargs): # this might have typo of 1/2 in it.
    """
    Finds Lambda amplitude of h
    """
    iota, psi = L_to_iota_psi(lambd, beta, phiL, thetaL)
    ra, dec = ecliptic_to_equatorial(lambd, beta, **kargs)
    fp, fc = det.antenna_pattern(ra, dec, psi, t_gps)
    cosiotasq = np.cos(iota) ** 2.

    
    lambAmp = ( ( 1. + cosiotasq ) ** 2. ) * ( fp ** 2. )
    lambAmp += ( 4. * ( cosiotasq  ) ) * ( fc ** 2. )
    lambAmp = np.sqrt(lambAmp)
    lambAmp = 0.5 * lambAmp
    
    return lambAmp


# @jit(nopython=True)
def get_phi_p(det, t_gps, lambd, beta, phiL, thetaL, **kargs):
    iota, psi = L_to_iota_psi(lambd, beta, phiL, thetaL)
    ra, dec = ecliptic_to_equatorial(lambd, beta, **kargs)
    fp, fc = det.antenna_pattern(ra, dec, psi, t_gps)
    cosiota = np.cos(iota)
    
    
    phiP = np.arctan2( 2* cosiota * fc, (1 + (cosiota ** 2.) ) * fp)
    return phiP
        
def get_h(freqs, det, par, **kargs):
    Mc = par["Mc"]; q = par["q"];  DL = par["Dl"]; tc = par["tc"]; phic = par["phic"]; 
    lambd = par["lambda"]; beta = par["beta"]; phiL = par["phiL"]; thetaL = par["thetaL"]; t_gps = par["gps_t"]
    
    u = get_u_ph(Mc, q, DL, freqs)

    amp = get_Lamb_amp(det, t_gps, lambd, beta, phiL, thetaL, **kargs)
    phiDs = get_phi_d_a(det, freqs, lambd, beta, t_gps, **kargs)
    phiP = get_phi_p(det, t_gps, lambd, beta, phiL, thetaL, **kargs)
    tcs = 2. * np.pi * tc * freqs
    
    hf = amp * u
    hf *= np.exp(1.j * (tcs - phiDs - phiP - phic))

    return hf


def get_h_alternate(freqs, det, par, **kargs):
    Mc = par["Mc"]; q = par["q"];  DL = par["Dl"]; tc = par["tc"]; phic = par["phic"]; 
    lambd = par["lambda"]; beta = par["beta"]; phiL = par["phiL"]; thetaL = par["thetaL"]; t_gps = par["gps_t"]

    m1 = pycbc.conversions.mass1_from_mchirp_q(Mc,q)
    m2 = pycbc.conversions.mass2_from_mchirp_q(Mc,q)
    
    freqs_lal = Array(freqs)
    hp, hc = pycbc.waveform.get_fd_waveform_sequence(approximant = "IMRPhenomD", mass1 = m1, mass2 = m2, distance = DL, sample_points = freqs_lal) #dont specify iota or polarization angle
    hp = np.array(hp); hc = np.array(hc)

    iota, psi = L_to_iota_psi(lambd, beta, phiL, thetaL)
    ra, dec = ecliptic_to_equatorial(lambd, beta)
    fp, fc = det.antenna_pattern(ra, dec, psi, t_gps)

    cosiota = np.cos(iota)

    phiDs = get_phi_d_a(det, freqs, lambd, beta, t_gps, **kargs)
    tcs = 2. * np.pi * tc * freqs


    Ap =  (1. + cosiota**2 )/2
    Ac = cosiota  

    hf = Ap*fp*hp + Ac* fc*hc
    hf *= np.exp(1.j * (tcs - phiDs - phic))
    

    return hf


def deriv_h_fd(freqs, idx, det, par, dpar, **kargs):
    """
    Function to differentate h with finite differences.
    """
    dh_idx = np.zeros([len(freqs)], dtype=np.complex128)

    if kargs['order']==2:
        par_u = par.copy(); par_u[idx] += dpar[idx]
        par_d = par.copy(); par_d[idx] -= dpar[idx]

        h_u = get_h(freqs, det, par_u, **kargs)
        h_d = get_h(freqs, det, par_d, **kargs)

        dh_idx = (1/2*h_u - 1/2*h_d) 
    elif kargs['order']==4:
        par_uu = par.copy(); par_uu[idx] += 2*dpar[idx]
        par_u = par.copy(); par_u[idx] += dpar[idx]
        par_d = par.copy(); par_d[idx] -= dpar[idx]
        par_dd = par.copy(); par_dd[idx] -= 2*dpar[idx]

        h_uu = get_h(freqs, det, par_uu, **kargs)
        h_u = get_h(freqs, det, par_u, **kargs)
        h_d = get_h(freqs, det, par_d, **kargs)
        h_dd = get_h(freqs, det, par_dd, **kargs)

        dh_idx = -1/12*h_uu+2/3*h_u-2/3*h_d+1/12*h_dd

    elif kargs['order']==6:
        par_uuu = par.copy(); par_uuu[idx] += 3*dpar[idx]
        par_uu = par.copy(); par_uu[idx] += 2*dpar[idx]
        par_u = par.copy(); par_u[idx] += dpar[idx]
        par_d = par.copy(); par_d[idx] -= dpar[idx]
        par_dd = par.copy(); par_dd[idx] -= 2*dpar[idx]
        par_ddd = par.copy(); par_ddd[idx] -= 3*dpar[idx]

        h_uuu = get_h(freqs, det, par_uuu, **kargs)
        h_uu = get_h(freqs, det, par_uu, **kargs)
        h_u = get_h(freqs, det, par_u, **kargs)
        h_d = get_h(freqs, det, par_d, **kargs)
        h_dd = get_h(freqs, det, par_dd, **kargs)
        h_ddd = get_h(freqs, det, par_ddd, **kargs)

        dh_idx = 1/60*h_uuu - 3/20*h_uu + 3/4*h_u - 3/4*h_d  + 3/20*h_dd - 1/60*h_ddd
    else:
        print('derivative order not set')

    dh_idx = dh_idx /  dpar[idx]
    
    return dh_idx

def innprod(hf1, hf2, psd, freqs):
    prod = 2. * integ.simps( (np.conj(hf1) * hf2 + hf1 * np.conj(hf2)) / psd , freqs)
    return prod

def snr(hf, psd, freqs):
    return np.real(np.sqrt(innprod(hf, hf, psd, freqs)))

def fish_single(freqs, det, par, dpar, idx_par, psd, log_flag, **kargs):

    nPt = len(freqs)
    nDof = len(idx_par)

    dh = np.zeros([nDof, nPt], dtype=np.complex128)

    for idx in idx_par:
        
        dh[idx_par[idx],:] = deriv_h_fd(freqs, idx, det, par, dpar, **kargs)

        if log_flag[idx]:
            dh[idx_par[idx], :] *= par[idx]
        
    gamma = np.zeros([nDof,nDof], dtype=np.float64)
        
    for i in range(nDof):
        for j in range(i, nDof):
            gamma[i, j] = np.real(innprod(dh[i, :], dh[j, :], psd, freqs))

    for i in range(nDof):
        for j in range(i):
            gamma[i, j] = np.conj(gamma[j, i])

    return gamma

def fish_multi(freqs, dets, par, dpar, idx_par, psd, log_flag, **kargs):
    nDof = len(idx_par)
    fish = np.zeros([nDof,nDof], dtype=np.float64)
    
    for det in dets:
        fish += fish_single(freqs, det, par, dpar, idx_par, psd, log_flag, **kargs)

    
    return fish

def get_delta_omega(beta, sigma_ll, sigma_bb, sigma_lb):
    
    deltaOmega = 2 * np.pi
    deltaOmega *= np.abs( np.cos(beta) )
    deltaOmega *= np.sqrt( sigma_ll * sigma_bb - sigma_lb**2.)
    
    return deltaOmega
    
def do_fisher_delta_omega(freqs, dets, par, dpar, psd, idx_par, log_flag, **kargs):
    
    fish = fish_multi(freqs, dets, par, dpar, idx_par, psd, log_flag, **kargs)
    
    sigma = sla.inv(fish)
    
    idx_l = idx_par["lambda"]
    idx_b = idx_par["beta"]
    sigma_ll = sigma[idx_l, idx_l] 
    sigma_bb = sigma[idx_b, idx_b] 
    sigma_lb = sigma[idx_l, idx_b]
    
    deltaOmega = get_delta_omega(par["beta"], sigma_ll, sigma_bb, sigma_lb)
    deltaOmega = deltaOmega*(180./np.pi)**2
    
    return deltaOmega

def sky_loc_plot_gen(freqs, dets, zval, par, dpar, psd, idx_par, log_flag):
    
    deltaOmegas = np.zeros(len(zval))
    
    for i in range(len(zval)):
        
        tic = time.perf_counter()
        par["z"] = zval[i]
        deltaOmegas[i] = do_fisher_delta_omega(freqs, dets, par, dpar, psd, idx_par, log_flag)
        toc = time.perf_counter()
        print(f"ran in {toc - tic:0.4f} seconds")
        
    return deltaOmegas

def L_to_cosiota(lambd, beta, phiL, thetaL):
    return np.cos(thetaL) * np.sin(beta) + np.sin(thetaL) * np.cos(beta) * np.cos(phiL-lambd)

def iota_solve(thL, *data):
    lambd, beta, phiL, iota = data  
    return L_to_cosiota(lambd, beta, phiL, thL) - np.cos(iota)

def find_thL(lamb, beta, phiL, iota):
    data = (lamb, beta, phiL, iota)
    thL0 = 0.
    thL = fsolve(iota_solve, thL0, args=data)[0]
    
    foundSol = np.isclose(iota_solve(thL, *data), 0.0)
    
    return thL, foundSol

def gen_random_angles(iota):
    while True:
        r = np.random.rand(3)
        lamb = np.arccos(2. * r[0] - 1)
        beta = np.pi / 2. - 2 * np.pi * r[1]
        phiL = 2. * np.pi * r[2]

        thL, foundSol = find_thL(lamb, beta, phiL, iota)
        foundSol = foundSol and (np.abs(thL)<3. *np.pi) # getting some random thL ~ 10^9 which satisfy eqn strangely.
        if foundSol:
            
            break
    
    return lamb, beta, phiL, (thL % 2 * np.pi)

def printA(a):
    """
    Useful to print matrices in readable way
    """
    for row in a:
        for col in row:
            print("{:.0e}".format(col), end="\t")
        print("")
        
def make_correlation_plot(sigma, parname, title):
    """
    Helper function for making a correlation function plot. It normalizes sigma so one can visually see correlations.
    sigma: nxn matrix, parname: string names of matrix, title:what to title it 
    """
    sigmanorm = sigma.copy()
    for idx, x in np.ndenumerate(sigma):
        if idx[0] <= idx[1]:
            sigmanorm[idx] = sigma[idx]/np.sqrt(sigma[idx[0],idx[0]])/np.sqrt(sigma[idx[1],idx[1]])
        else:
            sigmanorm[idx] = 0
    fig, ax = plt.subplots(figsize=(10,5))
    im = ax.imshow(sigmanorm)
    ax.set_xticks(np.arange(len(parname)))
    ax.set_yticks(np.arange(len(parname)))
    ax.set_xticklabels(parname)
    ax.set_yticklabels(parname)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    ax.set_title("Sigma Matrix terms: " + title)

    for i in range(len(parname)):
        for j in range(len(parname)):
            text = ax.text(j, i, np.round(sigmanorm[i, j],decimals = 2),
                           ha="center", va="center", color="w")


    fig.tight_layout()
    im.set_clim(-1., 1.)
    plt.colorbar(im)
    plt.show()
    
def test_fish(idx, epsvals, freqs, dets, par, dpar, idx_par, psd, log_flag): 
    
    dpar = dpar.copy()
    nEps = len(epsvals)
    nDof = len(idx_par)
    deltaParam = np.zeros([nDof, nEps], dtype=np.double)
    for i, eps in np.ndenumerate(epsvals):
        dpar[idx] = epsvals[i]
        fish = fish_multi(freqs, dets, par, dpar, idx_par, psd, log_flag)
        sigma = sla.inv(fish)
        deltaParam[:, i[0]] = np.sqrt(np.diagonal(sigma))
        
    return deltaParam
    
    
def make_test_fig(idx, epsvals, freqs, dets, par, dpar, idx_par, psd, log_flag):
    deltaParam = test_fish(idx, epsvals, freqs, dets, par, dpar, idx_par, psd, log_flag)
    
    for param in idx_par.keys():
        plt.loglog(epsvals, deltaParam[idx_par[param], :], label = param)
    
    plt.title('Errors in theta_i by changing: ' + idx)
    plt.legend()
    plt.show()