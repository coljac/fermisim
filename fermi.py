#!env python
import numpy as np
# import threading
from multiprocessing import Pool, Process, Lock
import astropy.units as u
from astropy.units import cds
import sys
import scipy.integrate as integrate
sys.path.append("/home/coljac/build/coltools")
import coltools as ct
cds.enable()
import time

SERSIC = 0.8

def density(point):
    central_density = 1e2
    radius = np.sqrt(point[0]**2. + point[1]**2.)
    return np.exp(radius * -SERSIC) * central_density * u.pc**-3


def stars_per_pc(density):
    return np.power(density, 1. / 3.)


def mean_distance_btw_stars(density):
    return np.power(density, -1. / 3.)


def distance_to_earth(point):
    x, y = point
    return np.sqrt(x**2 + (y - 8)**2) * u.kpc


def simulate_galaxy(v, r, e, cutoff=1e9):
    timestep = 1
    time = 0 # * u.year
    aliens = 0
    while True:
        if time > 10000:# * u.Gyr:
            break
        new_civs = np.random.binomial(timestep*int(1e6), r)
        for i in range(new_civs):
            if simulate_civ(v, e, 10000  - time):
                aliens += 1
            if aliens >= cutoff:
                break
        time += timestep # * u.year

    # print("Contacts: %d" % aliens)
    return aliens


def simulate_civ(v, e, time_remaining):
    earth = (0, 8)
    galaxy_size = 30  #  u.kpc

    x = np.random.normal(scale=galaxy_size / 2)
    y = np.random.normal(scale=galaxy_size / 2)
    from_ = (x, y)
    to = earth
    failed = False
    vector = np.array((to[0] - from_[0], to[1] - from_[1]))
    d = np.sqrt((vector * vector).sum())  # Distance to earth

    def mdensity(s):  # zero to 1
        return stars_per_pc(density(from_ + (vector * s))).value

    stops = integrate.quad(mdensity, 0, 1)[0] * d * 1000
    time_taken = (d * u.kpc / v).to(u.Myr)
    if time_taken > time_remaining*u.Myr:
        failed = True
    replication_chances = np.random.random((int(stops), 1))
    if (replication_chances <= e).sum() > 0:
        failed = True
    return failed

def run(r_index):
    np.random.seed()
    v = np.power(10., np.random.randint(-6, 0)) * cds.c
    if r_index is None:
        r = np.power(10., np.random.randint(-9, -5))
    else:
        r = np.power(10., r_index)
    f = 1 - np.power(10., np.random.randint(-7, 0))
    result = simulate_galaxy(v, r, f)
    return np.array((v.value, r, f, result))

def run_uniform(r_index):
    np.random.seed()
    v = np.random.random()*6 - 6
    if r_index is None:
        r = np.random.random()*4 - 9
    else:
        r = r_index
    e = np.random.random()*7 - 7
    result = simulate_galaxy(unlog(v)*cds.c, unlog(r), unlog(e), cutoff=100)
    return np.array((v, r, e, result))

def unlog(x):
    return np.power(10, x)


if __name__ == "__main__":
    threads = int(sys.argv[1])
    output = "results.npz"
    trials = 200
    r_index = None

    if len(sys.argv) > 2:
        output = sys.argv[2]
    if len(sys.argv) > 3:
        trials = int(sys.argv[3])
    if len(sys.argv) > 4:
        r_index = float(sys.argv[4])

    # v = 0.0001 * cds.c
    # r = 1e-8  # Chance of new civ per year
    # f = 1 - 1e-4

    lock = Lock()
    results = np.zeros((trials, 4))
    p = Pool(threads)

    for trial in range(int(trials/threads)):
        r = p.map(run_uniform, [r_index]*threads)
        results[trial*threads:(trial*threads)+threads] = r

    np.savez(output, results=results)
        
