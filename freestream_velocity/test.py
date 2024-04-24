#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

import argparse

import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy import integrate
from scipy import special
from nose.tools import assert_raises

import freestream


def assert_allclose(a, b, name, rtol=1e-3, atol=1e-6):
    """
    Check for accuracy with default tolerance of 0.1% plus numerical jitter.

    """
    assert np.allclose(a, b, rtol=rtol, atol=atol), \
        '{} does not agree:\n{}'.format(name, a/b)


def _check_shear_orthogonal(fs):
    """
    Check that the shear tensor is orthogonal to the flow velocity,
    pi^uv u_v = 0, or actually, pi^uv u_v < epsilon*sqrt(pi^uv pi_uv).

    """
    g = np.array([1, -1, -1])

    u = fs.flow_velocity()
    pi = fs.shear_tensor()

    upi = np.einsum('...uv,...v,v', pi, u, g)
    pipi = np.einsum('...uv,...uv,u,v', pi, pi, g, g)

    assert np.all(
        (np.abs(upi) < 1e-14) | (upi*upi < 1e-14*pipi[..., np.newaxis])
    ), (
        'Shear tensor is not orthogonal to flow velocity (min = {}, max = {}).'
        .format(upi.min(), upi.max())
    )


def test_gaussian():
    """
    check symmetric gaussian against analytic solution

    """
    # sample random thermalization time and Gaussian width
    t0 = np.random.uniform(0.01, 1.5)
    sigma = np.random.uniform(0.4, 1.0)
    sigma_sq = sigma*sigma
    
    vfs0 = np.random.uniform(0.4, 1.0)

    xymax = 5*sigma + 2*t0
    nsteps = 2*int(xymax/0.1) + 1
    s = np.s_[-xymax:xymax:nsteps*1j]
    Y, X = np.mgrid[s, s]
    grid_max = xymax/(1 - 1/nsteps)

    for var in ['t0', 'sigma', 'xymax', 'nsteps']:
        print(var, '=', eval(var))

    initial = np.exp(-(X*X + Y*Y)/(2*sigma_sq)) / (2*np.pi*sigma_sq)

    fs = freestream.FreeStreamer(initial, grid_max, t0, vfs0)

    initial_sum = initial.sum()
    final_sum = fs.Tuv(0, 0).sum() * t0
    assert abs(initial_sum/final_sum - 1) < 1e-6, \
        'particle number is not conserved: {} != {}'.format(initial_sum,
                                                            final_sum)

    # compare all quantities to analytic solution along the positive x-axis
    # skip the outermost cells where everything is tiny
    pos_x = np.s_[int(nsteps/2), int(nsteps/2)+1:int(0.95*nsteps)]
    x = X[pos_x]

    # T^uv can be computed analytically on the x-axis starting from a symmetric
    # Gaussian initial state.  After appropriate change of the variables the
    # integrals become Bessel functions.

    # This prefactor multiplies the entire matrix.
    prefactor = ( np.exp(-(x*x + vfs0*vfs0*t0*t0)/(2*sigma_sq)) 
               / (2*np.pi*sigma_sq*t0) )

    # Dimensionless variable that appears often.
    w = vfs0*x*t0/sigma_sq

    # Bessel functions I_0 and I_1.
    i0 = special.i0(w)
    i1 = special.i1(w)

    # Compare to exact results for T^uv.
    for (u, v, Tuv_exact) in [
            (0, 0, i0),
            (0, 1, vfs0*i1),
            (0, 2, 0),
            (1, 1, vfs0*vfs0*(i0 - i1/w) ),
            (1, 2, 0),
            (2, 2, vfs0*vfs0*i1/w ),
    ]:
        assert_allclose(fs.Tuv(u, v)[pos_x], prefactor*Tuv_exact,
                        'T{}{}'.format(u, v))

    #Compare exact reust for trace.
    trace_exact = prefactor*i0*( 1.0 - vfs0*vfs0 )
    
    assert_allclose(fs.trace()[pos_x], trace_exact,
                        'trace')

    # The Landau matching eigenvalue problem can also be solved analytically.
    # This discriminant "d" from the characteristic equation appears frequently
    # in the remaining expressions.
    d = np.sqrt(vfs0**4*i1**2 - 2*i0*i1*w*vfs0**2*( 1 + vfs0**2 ) 
       + ( i0*( 1 + vfs0**2 ) - 2*i1*vfs0 )
       *( i0*( 1 + vfs0**2 ) + 2*i1*vfs0 )*w**2)

    # Verify energy density (eigenvalue).
    energy_density = prefactor/(2*w) \
                    * ( i1*vfs0**2 + i0*w*( 1 - vfs0**2 ) + d )
    assert_allclose(fs.energy_density()[pos_x], energy_density,
                    'energy density')

    # Verify flow velocity (eigenvector).
    u0, u1, u2 = fs.flow_velocity().T
    assert np.allclose(u0*u0 - u1*u1 - u2*u2, 1), \
        'flow velocities are not normalized'

    v0 =  ( i0*( 1 + vfs0**2 )/(2*vfs0*i1) - vfs0/(2*w) + d/(2*i1*vfs0*w) )
    v1 = 1
    v2 = 0
    v_norm = np.sqrt(v0*v0 - v1*v1 - v2*v2)

    for i, v in enumerate([v0, v1, v2]):
        assert_allclose(fs.flow_velocity(i)[pos_x], v/v_norm,
                        'flow velocity u{}'.format(i))

    # The shear tensor pi^uv can also be computed analytically, although the
    # expressions are somewhat tedious...
    for (u, v, piuv_exact) in [
            (0, 0, 2*i1**2*vfs0**4*w*( i0*w*( 2*d 
                   - i1*( 5*vfs0**2 + 3 ) ) 
                   + 3*i1*( i1*vfs0**2 - d ) 
                   + 2*w**2*( i0**2*( vfs0**2 + 1 ) - 2*i1**2 ) ) \
                   /( 3*d*( d + i0*(vfs0**2 + 1 )*w - i1*vfs0**2 )**2 ) ),
                   
            (0, 1, i1*vfs0/3 
                   * ( 1 - ( 2*i1*vfs0**2 + i0*( 1 - vfs0**2 )*w )/d ) ),
            (0, 2, 0),
            (1, 1, - vfs0**2*( i0*w*( i1*( 5*vfs0**2 + 3 ) - 2*d ) 
                   + 3*i1*( d - i1*vfs0**2 ) 
                   - 2*w**2*( i0**2*( vfs0**2 + 1 ) - 2*i1**2 ) ) \
                   /(6*w*d) ),
            (1, 2, 0),
            (2, 2, ( 5*i1*vfs0**2 - d + i0*w*( 1 - vfs0**2 ) )/(6*w) ),
    ]:
        assert_allclose(fs.shear_tensor(u, v)[pos_x], prefactor*piuv_exact,
                        'shear tensor pi{}{}'.format(u, v))

    _check_shear_orthogonal(fs)

    # Check total pressure
    total_pressure_exact = prefactor*( d + i0*( vfs0**2 - 1 )*w + i1*vfs0**2 )\
                          /( 6*w )

    assert_allclose(fs.total_pressure()[pos_x], total_pressure_exact, 
                    'total pressure', rtol=1e-3, atol=1e-6)

    # Check bulk pressure
    bulk_pressure_exact = prefactor*i0*( vfs0**2 - 1  )/3

    assert_allclose(fs.bulk_pressure()[pos_x], bulk_pressure_exact, 
                    'ideal bulk pressure', rtol=1e-3, atol=1e-6)

    # ...and nonzero for any other eos. Consider P(e) = e/6.
    bulk_pressure_exact = prefactor\
                         *( d + 3*i0*w*( vfs0**2 - 1 )  + i1*vfs0**2 )/(12*w)
    
    assert_allclose(
        fs.bulk_pressure(eos=lambda e: e/6)[pos_x], bulk_pressure_exact,
        'nonideal bulk pressure', rtol=1e-3, atol=1e-6)
        
    # Check pi^\eta\eta.
    pi_eta_eta_exact = - total_pressure_exact/t0**2
    
    assert_allclose(
        fs.shear_tensor_eta_eta()[pos_x], pi_eta_eta_exact,
        'shear tensor eta eta component', rtol=1e-3, atol=1e-6)

def test_random():
    """
    check random initial condition against non-interpolated solution

    """
    for bad_shape in [10, (10, 12), (10, 10, 2)]:
        with assert_raises(ValueError):
            freestream.FreeStreamer(np.empty(bad_shape), 10, 1, 1.0)

    # Test the algorithm on a more interesting initial state that cannot be
    # solved analytically.

    # Construct a random initial state by sampling normally-distributed
    # positions for Gaussian blobs.
    t0 = np.random.uniform(0.01, 1.5)
    sigma = np.random.uniform(0.4, 0.8)

    vfs0 = np.random.uniform(0.4, 1.0)

    xy0 = np.random.standard_normal(50).reshape(-1, 2)

    # Define function that evaluates the initial density at (x, y) points.
    def f(x, y):
        z = np.zeros_like(x)
        for (x0, y0) in xy0:
            z += np.exp(-(np.square(x - x0) + np.square(y - y0))/(2*sigma**2))
        z /= 2*np.pi*sigma**2
        return z

    # Discretize the function onto a grid.
    xymax = np.max(xy0) + 5*sigma + 2*t0
    nsteps = int(2*xymax/0.1) + 2
    grid_max = xymax/(1 - 1/nsteps)
    s = np.s_[-xymax:xymax:nsteps*1j]
    Y, X = np.mgrid[s, s]

    for var in ['t0', 'sigma', 'xymax', 'grid_max', 'nsteps']:
        print(var, '=', eval(var))

    initial = f(X, Y)
    fs = freestream.FreeStreamer(initial, grid_max, t0, vfs0)

    initial_sum = initial.sum()
    final_sum = fs.Tuv(0, 0).sum() * t0
    assert abs(initial_sum/final_sum - 1) < 1e-6, \
        'particle number is not conserved: {} != {}'.format(initial_sum,
                                                            final_sum)

    # Pick some random grid points to check near the middle-ish of the grid.
    check_indices = np.random.randint(0.25*nsteps, 0.75*nsteps, (10, 2))
    check_Tuv = fs.Tuv()[check_indices.T[1], check_indices.T[0]]
    check_xy = (check_indices + 0.5)*2*grid_max/nsteps - grid_max

    print('check indices and points:')
    for (ix, iy), (x, y) in zip(check_indices, check_xy):
        print('{: 5d}{: 4d}{: 7.2f}{: 6.2f}'.format(ix, iy, x, y))

    # Check the components of T^uv against adaptive quadrature integration of
    # the actual continuous function.
    T_exact = 0*check_Tuv
    
    for u, v, w in [
            (0, 0, lambda phi: 1),
            (0, 1, lambda phi: vfs0*np.cos(phi)),
            (0, 2, lambda phi: vfs0*np.sin(phi)),
            (1, 1, lambda phi: np.square(vfs0*np.cos(phi))),
            (1, 2, lambda phi: np.square(vfs0)*np.cos(phi)*np.sin(phi)),
            (2, 2, lambda phi: np.square(vfs0*np.sin(phi))),
    ]:
        approx = check_Tuv[:, u, v] * t0
        exact = [
            integrate.quad(
                lambda phi: f(x0 - vfs0*t0*np.cos(phi), 
                              y0 - vfs0*t0*np.sin(phi))*w(phi),
                0, 2*np.pi
            )[0]/(2*np.pi) for (x0, y0) in check_xy
        ]
        
        T_exact[ :, u, v] = exact
        
        assert_allclose(approx, exact, 'T{}{}'.format(u, v))

    T_exact[:,1,0] = T_exact[:,0,1]
    T_exact[:,2,0] = T_exact[:,0,2]
    T_exact[:,2,1] = T_exact[:,1,2]    

    # check basic class functionality and properties

    # check energy-momentum tensor
    T00 = fs.Tuv(0, 0)
    assert all([
        T00.base is fs.Tuv(),
        T00.shape == fs.Tuv().shape[:2],
        np.all(T00 == fs.Tuv()[..., 0, 0]),
    ]), 'T00 is not a view of Tuv'

    assert_raises(ValueError, fs.Tuv, 0)

    # check flow velocity
    u0, u1, u2 = fs.flow_velocity().transpose(2, 0, 1)

    assert all([
        u1.base is fs.flow_velocity(),
        u1.shape == fs.flow_velocity().shape[:2],
        np.all(u1 == fs.flow_velocity()[..., 1]),
    ]), 'u1 is not a view of flow_velocity'

    assert np.allclose(u0*u0 - u1*u1 - u2*u2, 1), \
        'flow velocities are not normalized'

    # check shear tensor
    pi11 = fs.shear_tensor(1, 1)
    assert all([
        pi11.base is fs.shear_tensor(),
        pi11.shape == fs.shear_tensor().shape[:2],
        np.all(pi11 == fs.shear_tensor()[..., 1, 1]),
    ]), 'pi11 is not a view of shear_tensor'

    _check_shear_orthogonal(fs)

    assert_raises(ValueError, fs.shear_tensor, 2)

    # check definition of Tuv
    gg = np.diag([1., -1., -1.])
    e = fs.energy_density()[..., np.newaxis, np.newaxis]
    uu = np.einsum('...i,...j', fs.flow_velocity(), fs.flow_velocity())
    Delta = gg - uu
    Peff = e/3 + fs.bulk_pressure()[..., np.newaxis, np.newaxis]
    Tuv_calc = e*uu - Peff*Delta + fs.shear_tensor()
    assert np.allclose(fs.Tuv(), Tuv_calc), \
        'Tuv does not match the sum of its parts'

    # check total pressure
    check_e = e[check_indices.T[1], check_indices.T[0]]
    check_Delta = Delta[check_indices.T[1], check_indices.T[0]]
    total_pressure_exact = np.einsum('au,bv,...ab,...uv', gg, gg, \
                                     check_Delta, T_exact)
    total_pressure_exact /= -3
    total_pressure_exact = total_pressure_exact[..., np.newaxis, np.newaxis]

#    assert_allclose( \
#       fs.total_pressure()[check_indices.T[1], check_indices.T[0]], \
#       total_pressure_exact, 'total pressure',
#       rtol=1e-3, atol=1e-6)


    # check bulk pressure    
#    bulk_pressure_exact = total_pressure_exact - check_e/3
    
#    assert_allclose( \
#       fs.bulk_pressure()[check_indices.T[1], check_indices.T[0]], \
#       bulk_pressue_exact, 'ideal bulk pressure',
#       rtol=1e-3, atol=1e-6)

    #assert_allclose(
    #    fs.bulk_pressure(eos=lambda e: e/6), fs.energy_density()/6,
    #    'nonideal bulk pressure', rtol=1e-5, atol=1e-10
    #)

def main():
    parser = argparse.ArgumentParser(
        description='plot freestream test cases')

    parser.add_argument('plot', choices=['gaussian1', 'gaussian2', 'random', \
                                         'file'],
                        help='test case to plot')
    parser.add_argument('output', help='plot output file')
    parser.add_argument('--velocity', default=1.0, type=float, 				 help='freestreaming velocity')

    parser.add_argument('--input_file', help='hdf5 input file')
    parser.add_argument('--dataset', help='hdf5 dataset') 

    args = parser.parse_args()

    grid_max = 5.0
    nsteps = 100
    xymax = grid_max*(1 - 1/nsteps)
    s = np.s_[-xymax:xymax:nsteps*1j]
    Y, X = np.mgrid[s, s]
    vfs = args.velocity

    if args.plot == 'gaussian1':
        initial = np.exp(-(X*X + Y*Y)/(2*0.5**2))
    elif args.plot == 'gaussian2':
        initial = np.exp(-(np.square(X - 0.5) + 3*np.square(Y - 1)))
    elif args.plot == 'random':
        Y, X = np.mgrid[s, s]
        initial = np.zeros_like(X)
        sigmasq = .4**2
        # truncate gaussians at several widths (mimics typical IC models)
        rsqmax = 5**2 * sigmasq
        for x0, y0 in np.random.standard_normal((25, 2)):
            rsq = (X - x0)**2 + (Y - y0)**2
            cut = rsq < rsqmax
            initial[cut] += np.exp(-.5*rsq[cut]/sigmasq)
    elif args.plot == 'file':
        with h5py.File(args.input_file, 'r') as f:
            initial = np.array(f[args.dataset])

    fs = freestream.FreeStreamer(initial, grid_max, 1.0, vfs)

    plt.rcdefaults()

    def pcolorfast(arr, ax=None, title=None, vrange=None):
        if ax is None:
            ax = plt.gca()

        arrmax = np.abs(arr).max()

        try:
            vmin, vmax = vrange
        except TypeError:
            vmax = arrmax if vrange is None else vrange
            vmin = -vmax
        else:
            vmin = 2*vmin - vmax

        ax.set_aspect('equal')
        ax.pcolorfast((-grid_max, grid_max), (-grid_max, grid_max), arr,
                      vmin=vmin, vmax=vmax, cmap=plt.cm.RdBu_r)

        ax.text(-0.9*grid_max, 0.9*grid_max, 'max = {:g}'.format(arrmax),
                ha='left', va='top')

        ax.set_xlim(-grid_max, grid_max)
        ax.set_ylim(-grid_max, grid_max)

        if title is not None:
            ax.set_title(title)

    with PdfPages(args.output) as pdf:
        def finish():
            plt.tight_layout()
            pdf.savefig()
            plt.close()

        gamma_max = np.percentile(fs.flow_velocity(0), 90)

        bulk_max = np.percentile(np.abs(fs.bulk_pressure()), 90)
        trace_max = np.percentile(fs.trace(), 90)
        
        if bulk_max <= 1e-15:
           bulk_max = 1e-15

        if trace_max <= 1e-15:
           trace_max = 1e-15

        for arr, title, vrange in [
            (initial, 'initial state', None),
            (fs.energy_density(), r'energy density', None),
            (fs.flow_velocity(0), r'$u^0$', (1, gamma_max)),
            (fs.flow_velocity(1), r'$u^1$', gamma_max),
            (fs.flow_velocity(2), r'$u^2$', gamma_max),
            (fs.bulk_pressure(), r'$\Pi$', bulk_max), 
            (fs.trace(), r'$T^\mu_\mu$', trace_max),
        ]:
            plt.figure(figsize=(6, 6))

            pcolorfast(arr, title=title, vrange=vrange)

            plt.xlabel(r'$x$ [fm]')
            plt.ylabel(r'$y$ [fm]')

            finish()

        for (tensor, name) in [(fs.Tuv, 'T'), (fs.shear_tensor, r'\pi')]:
            fig, axes = plt.subplots(nrows=3, ncols=3,
                                     sharex='col', sharey='row',
                                     figsize=(12, 12))

            for (u, v), ax in np.ndenumerate(axes):
                if u < v:
                    ax.set_axis_off()
                else:
                    pcolorfast(tensor(u, v), ax,
                               r'${}^{{{}{}}}$'.format(name, u, v))

                if ax.is_last_row():
                    ax.set_xlabel(r'$x$ [fm]')
                if ax.is_first_col():
                    ax.set_ylabel(r'$y$ [fm]')

            finish()


if __name__ == "__main__":
    main()

