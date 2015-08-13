from __future__ import print_function
import os
import numpy
from numpy import diag, sqrt, newaxis, where, log10, ones, zeros, exp
from . import files

import esutil as eu
from esutil.numpy_util import between

MIN_ARATE=0.3
MAX_ARATE=0.6
MIN_S2N=10.0
MIN_TS2N=2.0

SN=0.22

SHEARS={0: [ 0.05,  0.  ],
        1: [ 0.  ,  0.05],
        2: [-0.05,  0.  ],
        3: [ 0.  , -0.05],
        4: [ 0.03536,  0.03536],
        5: [ 0.03536, -0.03536],
        6: [-0.03536, -0.03536],
        7: [-0.03536,  0.03536]}


def load_data(run,
              gnum=None,
              columns=None,
              keep_cols=None,
              select=True,
              combine=False,
              **keys):
    """
    load all g collated files into a list of structs

    parameters
    ----------
    run: string
        run name
    gnum: int, optional
        shear number.  If sent just return this one struct
    columns: list, optional
        Columns to read from the colllated file.  These are all available
        for selection
    keep_cols: list, optional
        Columns to finally keep from the collated file
    select: bool, optional
        If True, do selections.  Default True
    combine: bool, optional
        If True, do combine into a single struct.  Default False
    **keys:
        Other keywords for selections
    """
    import fitsio
    import esutil as eu 

    conf=files.read_config(run=run)

    if gnum is not None:
        gnums = [gnum]
    else:
        gnums = range(conf['ng'])

    dlist=[]
    for i in xrange(conf['ng']):
        fname=files.get_collated_file(run=run, gnum=i)

        print("reading:",fname)
        data=fitsio.read(fname, columns=columns)

        if select:
            print("    selecting")
            data=select_good(data, **keys)

        if keep_cols is not None:
            data=eu.numpy_util.extract_fields(data, keep_cols, strict=False)

        dlist.append(data)

    if gnum is not None:
        return dlist[0]

    if combine:
        data=eu.numpy_util.combine_arrlist(dlist)
        return data
    else:
        return dlist


def select_good(data,
                g_field='g',
                min_T=None,
                T_range=None,
                flux_range=None,
                min_arate=None,
                max_arate=None,
                min_s2n=None,
                min_s2n_r=None,
                min_Ts2n=None,
                min_Ts2n_r=None,
                min_efficiency=None,
                max_chi2per=None,
                s2n_r_range=None,
                mcal_s2n_r_range=None,
                Ts2n_r_range=None,
                fracdev_err_max=None,
                cut_fracdev_exact=False,
                fracdev_range=None,
                min_Trat=None,
                max_g=1.0):
    """
    apply standard selection
    """

    logic = ones(data.size, dtype=bool)

    elogic = (data['flags']==0)
    w,=where(elogic)
    if w.size != data.size:
        print("    kept %d/%d from flags" % (w.size, data.size))
        logic = logic & elogic


    if g_field in data.dtype.names:
        gv = data[g_field]
        g = numpy.sqrt( gv[:,0]**2 + gv[:,1]**2 )
        elogic = (g < max_g)
        w,=where(elogic)
        if w.size != data.size:
            print("    kept %d/%d from %s < %g" % (w.size, data.size, g_field, max_g))
            logic = logic & elogic

    if min_s2n is not None:
        elogic = (data['s2n_w'] > min_s2n)
        w,=where(elogic)
        if w.size != data.size:
            print("    kept %d/%d from s2n > %g" % (w.size, data.size, min_s2n))
            logic = logic & elogic

    if min_s2n_r is not None:
        if 's2n_r' in data.dtype.names:
            elogic = (data['s2n_r'] > min_s2n_r)
            w,=where(elogic)
            if w.size != data.size:
                print("    kept %d/%d from s2n_r > %g" % (w.size, data.size, min_s2n_r))
                logic = logic & elogic

    if s2n_r_range is not None:
        if 's2n_r' in data.dtype.names:
            rng=s2n_r_range
            elogic = (data['s2n_r'] > rng[0]) & (data['s2n_r'] < rng[1])
            w,=where(elogic)
            if w.size != data.size:
                print("    kept %d/%d from s2n_r in [%g,%g]" % (w.size, data.size, rng[0],rng[1]))
                logic = logic & elogic

    if mcal_s2n_r_range is not None:
        field='mcal_s2n_r'
        if field in data.dtype.names:
            rng=mcal_s2n_r_range
            elogic = between(data[field], rng[0], rng[1])
            w,=where(elogic)
            if w.size != data.size:
                print("    kept %d/%d from %s in [%g,%g]" % (w.size, data.size, field, rng[0],rng[1]))
                logic = logic & elogic


    if min_Ts2n is not None:
        elogic = (data['T_s2n'] > min_Ts2n)
        w,=where(elogic)
        if w.size != data.size:
            print("    kept %d/%d from Ts2n > %g" % (w.size, data.size, min_Ts2n))
            logic = logic & elogic

    if min_Ts2n_r is not None:
        if 'T_s2n_r' in data.dtype.names:
            elogic = (data['T_s2n_r'] > min_Ts2n_r)
            w,=where(elogic)
            if w.size != data.size:
                print("    kept %d/%d from Ts2n_r > %g" % (w.size, data.size, min_Ts2n_r))
                logic = logic & elogic

    if Ts2n_r_range is not None:
        if 'T_s2n_r' in data.dtype.names:
            rng=Ts2n_r_range
            elogic = (data['T_s2n_r'] > rng[0]) & (data['T_s2n_r'] < rng[1])
            w,=where(elogic)
            if w.size != data.size:
                print("    kept %d/%d from Ts2n_r in [%g,%g]" % (w.size, data.size, rng[0],rng[1]))
                logic = logic & elogic

    if min_T is not None:
        elogic = (data['log_T'] > min_T)
        w,=where(elogic)
        if w.size != data.size:
            print("    kept %d/%d from log_T > %g" % (w.size, data.size, min_T))
            logic = logic & elogic

    if T_range is not None:
        elogic = (data['log_T'] > T_range[0]) & (data['log_T'] < T_range[1])
        w,=where(elogic)
        if w.size != data.size:
            tup=(w.size, data.size, T_range[0], T_range[1])
            print("    kept %d/%d from log_T in [%g,%g]" % tup)
            logic = logic & elogic

    if flux_range is not None:
        elogic = (data['log_flux'] > flux_range[0]) & (data['log_flux'] < flux_range[1])
        w,=where(elogic)
        if w.size != data.size:
            tup=(w.size, data.size, flux_range[0], flux_range[1])
            print("    kept %d/%d from log_flux in [%g,%g]" % tup)
            logic = logic & elogic



    if 'arate' in data.dtype.names:
        elogic = (data['arate'] > min_arate) & (data['arate'] < max_arate)
        w,=where(elogic)
        if w.size != data.size:
            print("    kept %d/%d from arate [%g,%g]" % (w.size, data.size,
                                                         min_arate,max_arate))
            logic = logic & elogic

    if 'fracdev' in data.dtype.names:
        if fracdev_range is not None:
            elogic = (  (data['fracdev'] > fracdev_range[0])
                      & (data['fracdev'] < fracdev_range[1]) )
            w,=where(elogic)
            if w.size != data.size:
                mess="    kept %d/%d from fracdev [%g,%g]"
                print(mess % (w.size, data.size, fracdev_range[0],fracdev_range[1]))
                logic = logic & elogic


        if fracdev_err_max is not None:
            elogic = (data['fracdev_err'] < fracdev_err_max)
            w,=where(elogic)
            if w.size != data.size:
                print("    kept %d/%d from fracdev_err < %g" % (w.size, data.size, fracdev_err_max))
                logic = logic & elogic

        if cut_fracdev_exact:
            elogic = (data['fracdev'] != 0.0) & (data['fracdev'] != 1.0)
            w,=where(elogic)
            if w.size != data.size:
                print("    kept %d/%d from fracdev exact" % (w.size, data.size))
                logic = logic & elogic

    if min_efficiency is not None:
        elogic = (data['efficiency'] > min_efficiency)
        w,=where(elogic)
        if w.size != data.size:
            print("    kept %d/%d from eff > %g" % (w.size, data.size, min_efficiency))
            logic = logic & elogic

    if max_chi2per is not None:
        elogic = (data['chi2per'] > max_chi2per)
        w,=where(elogic)
        if w.size != data.size:
            print("    kept %d/%d from chi2per < %g" % (w.size, data.size, max_chi2per))
            logic = logic & elogic


    if min_Trat is not None:
        if 'log_T_r' in data.dtype.names:
            T_r = exp(data['log_T_r'])
        else:
            T_r = data['T_r']

        Trat = T_r/data['psf_T_r']

        elogic = (Trat > min_Trat)
        w,=where(elogic)
        if w.size != data.size:
            print("    kept %d/%d from Tg/Tpsf > %g" % (w.size, data.size, min_Trat))
            logic = logic & elogic



    w,=where(logic)
    print("    kept %d/%d" % (w.size, data.size))

    data=data[w]
    return data

def get_weights(data, wstyle='tracecov'):
    """
    a good set of weights
    """

    #print("wstyle:",wstyle)
    if wstyle=='tracecov':
        csum=data['g_cov'][:,0,0] + data['g_cov'][:,1,1]
        wts=1.0/(2*SN**2 + csum)
    elif wstyle=='s2n_r':
        wts=1.0/(2*SN**2 + 2/data['s2n_r']**2)
    elif wstyle is None:
        wts=ones(data.size)
    else:
        raise ValueError("bad weights style: '%s'" % wstyle)

    return wts

def fit_m_c(dlist, gfield='g', wstyle='tracecov', show=False, doprint=False, get_plt=False):
    """
    get m and c
    """
    import fitting

    ng = len(dlist)
    gtrue = numpy.zeros( (ng,2) )
    gdiff = gtrue.copy()
    gdiff_err = gtrue.copy()

    nobj=0
    for i,data in enumerate(dlist):
        gtrue[i,:],gmean,gcov=calc_gmean(data, wstyle=wstyle)
        gdiff[i,:] = gmean - gtrue[i,:]
        gdiff_err[i,:] = sqrt(diag(gcov))

        nobj += data.size

    lf1=fitting.fit_line(gtrue[:,0], gdiff[:,0], yerr=gdiff_err[:,0])
    lf2=fitting.fit_line(gtrue[:,1], gdiff[:,1], yerr=gdiff_err[:,1])
    fitters=[lf1,lf2]


    m1,c1 = lf1.pars
    m1err,c1err = lf1.perr
    m2,c2 = lf2.pars
    m2err,c2err = lf2.perr

    res={'nobj':nobj,
         'm1':m1,
         'm1err':m1err,
         'm2':m2,
         'm2err':m2err,
         'c1':c1,
         'c1err':c1err,
         'c2':c2,
         'c2err':c2err}

    if doprint:
        mess="""
nobj: %(nobj)d
m1: %(m1).3g +/- %(m1err).3g
m2: %(m2).3g +/- %(m2err).3g
c1: %(c1).3g +/- %(c1err).3g
c2: %(c2).3g +/- %(c2err).3g""".strip()
        mess = mess % res
        print(mess)

    if get_plt or show:
        plt=plot_gdiff_vs_gtrue(gtrue, gdiff, gdiff_err, fitters=[lf1,lf2])

        if show:
            plt.show()

    if get_plt:
        return res, plt
    else:
        return res

def fit_m_c_s2n_diff(dlist, s2n_edges, field='s2n_r', wstyle='tracecov'):
    """
    fit m and c vs s/n differential
    """
    
    def select_dlist(dlist, s2n_min, s2n_max, field):
        dlist2=[]
        s2n_sum=0.0
        n=0
        for d in dlist:
            w,=where( between(d[field], s2n_min, s2n_max) )

            n += w.size
            s2n_sum += d[field][w].sum()

            dlist2.append(d[w])

        s2n=s2n_sum/n
        return dlist2, s2n

    dtype=[('s2n','f8'),
           ('nobj','i8'),
           ('m1','f8'),
           ('m1err','f8'),
           ('m2','f8'),
           ('m2err','f8'),
           ('c1','f8'),
           ('c1err','f8'),
           ('c2','f8'),
           ('c2err','f8')]
           
    num=len(s2n_edges)
    out=zeros(num-1, dtype=dtype)

    for i in xrange(num-1):
        dlist2, s2n=select_dlist(dlist, s2n_edges[i], s2n_edges[i+1], field)

        res=fit_m_c(dlist2, doprint=True, wstyle=wstyle)
        out['s2n'][i] = s2n
        out['nobj'][i] = res['nobj']
        out['m1'][i] = res['m1']
        out['m1err'][i] = res['m1err']
        out['m2'][i] = res['m2']
        out['m2err'][i] = res['m2err']

        out['c1'][i] = res['c1']
        out['c1err'][i] = res['c1err']
        out['c2'][i] = res['c2']
        out['c2err'][i] = res['c2err']

    return out



def fit_m_c_s2n_min(dlist, s2n_minvals, field='s2n_r', wstyle='tracecov'):
    """
    fit m and c vs s/n
    """
    
    def select_dlist(dlist, s2n_min):
        dlist2=[]
        for d in dlist:
            w,=where(d[field] > s2n_min)
            dlist2.append(d[w])
        return dlist2

    dtype=[('s2n_min','f8'),
           ('nobj','i8'),
           ('m1','f8'),
           ('m1err','f8'),
           ('m2','f8'),
           ('m2err','f8'),
           ('c1','f8'),
           ('c1err','f8'),
           ('c2','f8'),
           ('c2err','f8')]
           
    out=zeros(len(s2n_minvals), dtype=dtype)

    for i,s2n_min in enumerate(s2n_minvals):
        print("%s > %.3g" % (field, s2n_min))
        dlist2=select_dlist(dlist, s2n_min)

        res=fit_m_c(dlist2, doprint=True, wstyle=wstyle)
        out['s2n_min'][i] = s2n_min
        out['nobj'][i] = res['nobj']
        out['m1'][i] = res['m1']
        out['m1err'][i] = res['m1err']
        out['m2'][i] = res['m2']
        out['m2err'][i] = res['m2err']

        out['c1'][i] = res['c1']
        out['c1err'][i] = res['c1err']
        out['c2'][i] = res['c2']
        out['c2err'][i] = res['c2err']

    return out

def plot_m_c_s2n(data, field='s2n_r', type='min', show=False, xlog=True):
    import biggles
    tab=biggles.Table(2,1)

    if type=='min':
        pfield='s2n_min'
        xlabel = 'min %s' % field
    else:
        pfield='s2n'
        xlabel = field

    if xlog:
        xrng=[0.75*data[pfield].min(), 1.5*data[pfield].max()]
    else:
        xrng=[0.9*data[pfield].min(), 1.1*data[pfield].max()]

    m1c=biggles.Points(data[pfield], data['m1'],
                      type='filled circle', color='blue')
    m1c.label='m1'
    m1errc=biggles.SymmetricErrorBarsY(data[pfield], data['m1'], data['m1err'],
                                       color='blue')

    m2c=biggles.Points(data[pfield], data['m2'],
                      type='filled circle', color='red')
    m2c.label='m2'
    m2errc=biggles.SymmetricErrorBarsY(data[pfield], data['m2'], data['m2err'],
                                       color='red')
    mkey=biggles.PlotKey(0.9,0.9,[m1c,m2c],halign='right')


    c1c=biggles.Points(data[pfield], data['c1'],
                      type='filled circle', color='blue')
    c1c.label='c1'
    c1errc=biggles.SymmetricErrorBarsY(data[pfield], data['c1'], data['c1err'],
                                       color='blue')

    c2c=biggles.Points(data[pfield], data['c2'],
                      type='filled circle', color='red')
    c2c.label='c2'
    c2errc=biggles.SymmetricErrorBarsY(data[pfield], data['c2'], data['c2err'],
                                       color='red')
    ckey=biggles.PlotKey(0.9,0.9,[c1c,c2c],halign='right')

    zc=biggles.Curve(data[pfield], data[pfield]*0)

    mplt=biggles.FramedPlot()
    mplt.xlabel=xlabel
    mplt.ylabel='m'
    mplt.xrange=xrng
    mplt.xlog=xlog
    mplt.add( zc, m1c, m1errc, m2c, m2errc, mkey )

    cplt=biggles.FramedPlot()
    cplt.xlabel=xlabel
    cplt.ylabel='c'
    cplt.xrange=xrng
    cplt.xlog=xlog
    cplt.add( zc, c1c, c1errc, c2c, c2errc, ckey )


    tab[0,0] = mplt
    tab[1,0] = cplt

    if show:
        tab.show()
    return tab


def plot_gdiff_vs_gtrue(gtrue, gdiff, gdiff_err, fitters=None):
    """
    plot gdiff vs gtrue, both components, return the plot
    """
    import biggles

    plt=biggles.FramedPlot()
    plt.xlabel='g_true'
    plt.ylabel=r'$\Delta g$'

    color1='blue'
    color2='red'
    pts1=biggles.Points(gtrue[:,0], gdiff[:,0], type='filled circle', color=color1)
    perr1=biggles.SymmetricErrorBarsY(gtrue[:,0], gdiff[:,0], gdiff_err[:,0], color=color1)
    pts2=biggles.Points(gtrue[:,1], gdiff[:,1], type='filled diamond', color=color2)
    perr2=biggles.SymmetricErrorBarsY(gtrue[:,1], gdiff[:,1], gdiff_err[:,1], color=color2)

    pts1.label=r'$g_1$'
    pts2.label=r'$g_2$'

    key=biggles.PlotKey(0.9, 0.9, [pts1,pts2], halign='right')
    
    z=biggles.Curve( [gtrue[:,0].min(), gtrue[:,0].max()], [0,0] )
    plt.add( z )
    plt.add( pts1, perr1, pts2, perr2, key )

    if fitters is not None:
        lf1,lf2=fitters

        ply1=lf1.get_poly()
        ply2=lf2.get_poly()
        c1=biggles.Curve(gtrue[:,0], ply1(gtrue[:,0]), color=color1)
        c2=biggles.Curve(gtrue[:,1], ply2(gtrue[:,1]), color=color2)
        plt.add(c1, c2)

    return plt


def calc_fracdev_s2n_diff(dlist, s2n_edges, field='s2n_r', wstyle='tracecov'):
    """
    fit m and c vs s/n differential
    """
    
    def select_fracdev(dlist, s2n_min, s2n_max, field):
        """
        get fracdev from all sub fields for the specified binning
        """
        s2n_sum=0.0
        n=0

        alist=[]
        for d in dlist:
            w,=where( between(d[field], s2n_min, s2n_max) )

            alist.append( d['fracdev'][w] )

            n += w.size
            s2n_sum += d[field][w].sum()

        fracdevs = numpy.concatenate( alist )
        s2n=s2n_sum/n
        return fracdevs, s2n, n

    dtype=[('s2n','f8'),
           ('nobj','i8'),
           ('fracdev','f8'),
           ('fracdev_err','f8')]
           
    num=len(s2n_edges)
    out=zeros(num-1, dtype=dtype)

    for i in xrange(num-1):
        fracdevs, s2n, nobj =select_fracdev(dlist, s2n_edges[i], s2n_edges[i+1], field)

        fracdev = fracdevs.mean()
        fracdev_err = fracdevs.std()/sqrt(fracdevs.size)

        out['s2n'][i] = s2n
        out['nobj'][i] = nobj
        out['fracdev'][i] = fracdev
        out['fracdev_err'][i] = fracdev_err

        print("    %.1f %d %.3g +/- %.3g" % (s2n, nobj, fracdev, fracdev_err))

    return out




def calc_gmean(data, wstyle='tracecov'):
    """
    get gtrue, gmeas, gcov
    """
    import jackknife
    gtrue = data['shear_true'].mean(axis=0)
    gmeas = numpy.zeros(2)

    wts=get_weights(data, wstyle=wstyle)

    wa=wts[:,newaxis]
    jdsum=data['g']*wa
    if 'g_sens' in data.dtype.names:
        jwsum=data['g_sens']*wa
    else:
        jwsum=numpy.ones( data['g'].shape )*wa

    gmeas,gcov=jackknife.wjackknife(vsum=jdsum, wsum=jwsum)

    return gtrue, gmeas, gcov

def quick_shear(data, ishear, g=None, nbin=11,
                s2n_field='s2n_true',
                w=None, getall=False, use_weights=True, min_s2n=10.0, max_s2n=300.0):
    import esutil as eu 

    if w is None:
        print("selecting")
        w,=where(data['flags']==0)

    if use_weights:
        print("getting weights")
        weights=get_weights(data[w])
    else:
        weights=None

    min_logs2n = log10(min_s2n)
    max_logs2n = log10(max_s2n)

    if g is None:
        g=data['g'][:,ishear]

    print("binner")
    b=eu.stat.Binner(log10( data[s2n_field][w] ), g[w], weights=weights)
    b.dohist(min=min_logs2n, max=max_logs2n, nbin=nbin)
    b.calc_stats()

    print("sens binner")
    bs=eu.stat.Binner(log10( data[s2n_field][w] ), data['g_sens'][w,ishear], weights=weights)
    bs.dohist(min=min_logs2n, max=max_logs2n, nbin=nbin)
    bs.calc_stats()

    if use_weights:
        shear=b['wymean']/bs['wymean']
        shear_err=shear*sqrt( (b['wyerr']/b['wymean'])**2 + (bs['wyerr']/bs['wymean'])**2 )
    else:
        shear=b['ymean']/bs['ymean']
        shear_err=shear*sqrt( (b['yerr']/b['ymean'])**2 + (bs['yerr']/bs['ymean'])**2 )

    s2n= b['xmean']

    if getall:
        return s2n, shear, shear_err, b, bs
    else:
        return s2n, shear, shear_err


def quick_pqr(data, nbin=11, s2n_field='s2n_true',
              w=None, getall=False, min_s2n=10.0, max_s2n=300.0):
    import ngmix
    import esutil as eu 

    if w is None:
        print("selecting")
        w,=numpy.where(data['flags']==0)

    min_logs2n = log10(min_s2n)
    max_logs2n = log10(max_s2n)

    shear     = numpy.zeros( (nbin,2) )
    shear_err = numpy.zeros( (nbin,2) )
    shear_cov = numpy.zeros( (nbin,2,2) )

    logs2n = log10( data[s2n_field][w] )
    hdict=eu.stat.histogram(logs2n, min=log10(min_s2n),max=log10(max_s2n),
                            nbin=nbin, more=True)

    rev=hdict['rev']
    for i in xrange(nbin):
        if rev[i] != rev[i+1]:
            ww=rev[ rev[i]:rev[i+1] ]

            wtmp=w[ww]
            tshear, tshear_cov = ngmix.pqr.calc_shear(data['P'][wtmp],
                                                      data['Q'][wtmp,:],
                                                      data['R'][wtmp,:,:])
            
            terr=sqrt(diag(tshear_cov))

            shear[i,:] = tshear
            shear_err[i,:] = terr
            shear_cov[i,:,:] = tshear_cov


    return hdict, shear, shear_err, shear_cov



class Analyzer(dict):
    """
    analyze m and c vs various parameters

    wstyles should be 'tracecov' or None

    If deep data are sent, sensitivity refers to fields in that data
    """
    def __init__(self,
                 dlist,
                 sens_style='metacal',
                 g_field='mcal_g',
                 gcov_field='mcal_pars_cov', # forgot to record mcal_g_cov separately
                 sens_field='mcal_g_sens',
                 wstyle=None,
                 deep_data=None):

        self.dlist=dlist
        self.g_field=g_field
        self.gcov_field=gcov_field
        self.sens_field=sens_field
        self.sens_style=sens_style
        self.wstyle=wstyle

        self.deep_data=deep_data

    def fit_m_c(self, show=False, doprint=False,
                get_plt=False, dlist=None, deep_data=None):
        """
        get m and c

        sens dlist= to specify different data to fit
        """
        import fitting

        if dlist is None:
            dlist=self.dlist

        if deep_data is None:
            # could still be None!
            deep_data=self.deep_data

        ng = len(dlist)
        gtrue = numpy.zeros( (ng,2) )
        gdiff = gtrue.copy()
        gdiff_err = gtrue.copy()

        nobj=0
        for i,data in enumerate(dlist):
            gtrue[i,:],gmean,gcov=self.calc_gmean(data, deep_data=deep_data)
            gdiff[i,:] = gmean - gtrue[i,:]
            gdiff_err[i,:] = sqrt(diag(gcov))

            nobj += data.size

        lf1=fitting.fit_line(gtrue[:,0], gdiff[:,0], yerr=gdiff_err[:,0])
        lf2=fitting.fit_line(gtrue[:,1], gdiff[:,1], yerr=gdiff_err[:,1])
        fitters=[lf1,lf2]


        m1,c1 = lf1.pars
        m1err,c1err = lf1.perr
        m2,c2 = lf2.pars
        m2err,c2err = lf2.perr

        res={'nobj':nobj,
             'm1':m1,
             'm1err':m1err,
             'm2':m2,
             'm2err':m2err,
             'c1':c1,
             'c1err':c1err,
             'c2':c2,
             'c2err':c2err}

        if doprint:
            mess="""nobj: %(nobj)d
m1: %(m1).3g +/- %(m1err).3g
m2: %(m2).3g +/- %(m2err).3g
c1: %(c1).3g +/- %(c1err).3g
c2: %(c2).3g +/- %(c2err).3g""".strip()
            mess = mess % res
            print(mess)

        if get_plt or show:
            plt=plot_gdiff_vs_gtrue(gtrue, gdiff, gdiff_err, fitters=[lf1,lf2])

            if show:
                plt.show()

        if get_plt:
            return res, plt
        else:
            return res


    def fit_m_c_vs(self, field, minval, maxval, nbin, dolog=True):
        """
        calculate m and c in bins of some variable

        returns
        -------
        means, m, merr, c, cerr
        """

        dlist=self.dlist

        revlist=self._do_hist_many(dlist, field, minval, maxval, nbin, dolog=dolog)

        # select deep data the same way
        deep_data=self.deep_data
        if deep_data is not None:
            dodeep=True
            drev = self._do_hist1(deep_data[field],
                                      minval, maxval, nbin, dolog=dolog)
        else:
            tdeep_data=None

        m1=numpy.zeros(nbin)
        m1err=numpy.zeros(nbin)
        m2=numpy.zeros(nbin)
        m2err=numpy.zeros(nbin)
        c1=numpy.zeros(nbin)
        c1err=numpy.zeros(nbin)
        c2=numpy.zeros(nbin)
        c2err=numpy.zeros(nbin)

        fmean=numpy.zeros(nbin)
        num=numpy.zeros(nbin,dtype='i8')

        for i in xrange(nbin):
            wlist=[ rev[ rev[i]:rev[i+1] ] for rev in revlist ]

            f_sum=0.0
            wsum=0.0
            cut_dlist=[]
            
            ssum = numpy.zeros( (2,2) )
            for d,w in zip(dlist,wlist):
                td = d[w].copy()


                wts = self.get_weights(td)
                f_sum += (wts*td[field]).sum()
                wsum += wts.sum()

                num[i] += td.size

                cut_dlist.append(td)

            if deep_data is not None:
                tdeep_data=deep_data[ drev[ drev[i]:drev[i+1] ] ]

            res = self.fit_m_c(dlist=cut_dlist,
                               deep_data=tdeep_data,
                               doprint=True)

            m1[i],m2[i],c1[i],c2[i] = res['m1'],res['m2'],res['c1'],res['c2']
            m1err[i],m2err[i]=res['m1err'],res['m2err']
            c1err[i],c2err[i] = res['c1err'],res['c2err']

            fmean[i] = f_sum/wsum

            print("%s mean: %g" % (field, fmean[i]))
            print()

        return {'mean':fmean,
                'num':num,
                'm1':m1,
                'm1err':m1err,
                'm2':m2,
                'm2err':m2err,
                'c1':c1,
                'c1err':c1err,
                'c2':c2,
                'c2err':c2err}

    def plot_m_c_vs(self, res, name, xlog=True, show=False):
        """
        result from fit_m_c_vs
        """

        import biggles
        tab=biggles.Table(2,1)

        xvals = res['mean']

        if xlog:
            xrng=[0.75*xvals.min(), 1.5*xvals.max()]
        else:
            xrng=[0.9*xvals.min(), 1.1*xvals.max()]


        m1c=biggles.Points(xvals, res['m1'],
                          type='filled circle', color='blue')
        m1c.label='m1'
        m1errc=biggles.SymmetricErrorBarsY(xvals, res['m1'], res['m1err'],
                                           color='blue')

        m2c=biggles.Points(xvals, res['m2'],
                          type='filled circle', color='red')
        m2c.label='m2'
        m2errc=biggles.SymmetricErrorBarsY(xvals, res['m2'], res['m2err'],
                                           color='red')
        mkey=biggles.PlotKey(0.9,0.9,[m1c,m2c],halign='right')


        c1c=biggles.Points(xvals, res['c1'],
                          type='filled circle', color='blue')
        c1c.label='c1'
        c1errc=biggles.SymmetricErrorBarsY(xvals, res['c1'], res['c1err'],
                                           color='blue')

        c2c=biggles.Points(xvals, res['c2'],
                          type='filled circle', color='red')
        c2c.label='c2'
        c2errc=biggles.SymmetricErrorBarsY(xvals, res['c2'], res['c2err'],
                                           color='red')
        ckey=biggles.PlotKey(0.9,0.9,[c1c,c2c],halign='right')

        zc=biggles.Curve(xvals, xvals*0)

        mplt=biggles.FramedPlot()
        mplt.xlabel=name
        mplt.ylabel='m'
        mplt.xrange=xrng
        mplt.xlog=xlog
        mplt.add( zc, m1c, m1errc, m2c, m2errc, mkey )

        cplt=biggles.FramedPlot()
        cplt.xlabel=name
        cplt.ylabel='c'
        cplt.xrange=xrng
        cplt.xlog=xlog
        cplt.add( zc, c1c, c1errc, c2c, c2errc, ckey )

        tab[0,0] = mplt
        tab[1,0] = cplt

        if show:
            tab.show()
        return tab




    def calc_gmean(self, data, deep_data=None):
        """
        get gtrue, gmeas, gcov
        """
        import ngmix

        gtrue = data['shear_true'].mean(axis=0)

        wts=self.get_weights(data)

        if deep_data is not None:
            print("getting deep data weights")
            deep_wts = self.get_weights(deep_data)

        #print("using sens:",self.sens_field)

        chunksize=1000
        if chunksize > data.size/10.0:
            chunksize = data.size/10
        #print("chunksize:",chunksize)

        if self.sens_style=='lensfit':
            raise RuntimeError("lensfit not working yet")
            if self.sens_field is None:
                sens = numpy.ones( (data.size, 2) )
            else:
                sens = data[self.sens_field].copy()

            gmeas, gcov = ngmix.lensfit.lensfit_jackknife(data[self.g_field],
                                                          sens,
                                                          weights=wts,
                                                          chunksize=chunksize)
        elif self.sens_style=='metacal':

            if self.sens_field is None:
                sens = numpy.ones( (data.size,2,2) )
            else:
                import images

                if deep_data is not None:
                    print("    using mean sensitivity from deep data")

                    wsum = deep_wts.sum()
                    wa = deep_wts[:,newaxis,newaxis]

                    s=deep_data[self.sens_field]
                    sens_mean = (s*wa).sum(axis=0)/wsum

                    sens = numpy.zeros( (data.size,2,2) )
                    sens[:,0,0] = sens_mean[0,0]
                    sens[:,0,1] = sens_mean[0,1]
                    sens[:,1,0] = sens_mean[1,0]
                    sens[:,1,1] = sens_mean[1,1]
                else:
                    print("    using sensitivity from this data")
                    sens = data[self.sens_field].copy()

            print("mean sens:")
            images.imprint(sens.mean(axis=0), fmt='%g')

            res = ngmix.metacal.jackknife_shear(data[self.g_field],
                                                sens,
                                                weights=wts,
                                                chunksize=chunksize)
            gmeas=res['shear']
            gcov=res['shear_cov']
        elif self.sens_style is None:
            gmeas = data[self.g_field].mean(axis=0)
            gcov=numpy.zeros( (2,2) )
            gcov[0,0] = data[self.g_field][:,0].std()/numpy.sqrt(data.size)
            gcov[1,1] = data[self.g_field][:,1].std()/numpy.sqrt(data.size)
        else:
            raise ValueError("bad sens_style: '%s'" % self.sens_style)

        return gtrue, gmeas, gcov

    def get_weights(self, data):
        """
        a good set of weights
        """

        if self.wstyle=='tracecov':

            if 'g_cov' in self.gcov_field:
                gcov = data[self.gcov_field]
            else:
                cov = data[self.gcov_field]
                gcov = cov[:,2:2+2, 2:2+2]

            csum=gcov[:,0,0] + gcov[:,1,1]

            wts=1.0/(2*SN**2 + csum)

        elif self.wstyle is None:

            wts=ones(data.size)

        else:
            raise ValueError("bad weights style: '%s'" % wstyle)

        return wts



    def doplot(self, show=False):
        """
        plot m,c vs s2n
        """
        import biggles

        xrng=[0.5*self.s2n.min(), 1.5*self.s2n.max()]
        mplt=biggles.FramedPlot()
        cplt=biggles.FramedPlot()

        mplt.xlabel='S/N'
        mplt.ylabel='m'
        mplt.xlog=True
        mplt.xrange=xrng

        cplt.xlabel='S/N'
        cplt.ylabel='c'
        cplt.xlog=True
        cplt.xrange=xrng

        color1='blue'
        color2='red'

        mcurve1=biggles.Curve(self.s2n, self.m[:,0], type='solid',
                             color=color1)
        merr1=biggles.SymmetricErrorBarsY(self.s2n, self.m[:,0], self.merr[:,0],
                                          color=color1)
        mcurve2=biggles.Curve(self.s2n, self.m[:,1], type='dashed',
                             color=color2)
        merr2=biggles.SymmetricErrorBarsY(self.s2n, self.m[:,1], self.merr[:,1],
                                          color=color2)
        
        ccurve1=biggles.Curve(self.s2n, self.c[:,0], type='solid',
                             color=color1)
        cerr1=biggles.SymmetricErrorBarsY(self.s2n, self.c[:,0], self.cerr[:,0],
                                          color=color1)
        ccurve2=biggles.Curve(self.s2n, self.c[:,1], type='dashed',
                             color=color2)
        cerr2=biggles.SymmetricErrorBarsY(self.s2n, self.c[:,1], self.cerr[:,1],
                                          color=color2)
 
        key=biggles.PlotKey(0.1,0.9,[mcurve1,mcurve2],halign='left')

        mcurve1.label=r'$g_1$'
        mcurve2.label=r'$g_2$'
        ccurve1.label=r'$g_1$'
        ccurve2.label=r'$g_2$'

        zc=biggles.Curve( self.s2n, self.s2n*0 )

        mplt.add( mcurve1, merr1, mcurve2, merr2, zc, key )
        cplt.add( ccurve1, cerr1, ccurve2, cerr2, zc, key )

        if show:
            mplt.show()
            cplt.show()
        return mplt, cplt

    def _do_hist_many(self, dlist, field, minval, maxval, nbin, dolog=True):
        # get reverse indices for our binning
        revlist=[]
        for d in dlist:
            rev=self._do_hist1(d[field], minval, maxval, nbin, dolog=dolog)
            revlist.append(rev)

        return revlist

    def _do_hist1(self, data, minval, maxval, nbin, dolog=True):
        import esutil as eu

        if dolog:
            data = numpy.log10( data )
            minval = numpy.log10( minval )
            maxval = numpy.log10( maxval )

        h,rev=eu.stat.histogram(data,
                                min=minval,
                                max=maxval,
                                nbin=nbin,
                                rev=True)

        return rev

def plot_e_vs_sigma(data,
                    use_true_fwhm=False,
                    yrange=None,
                    nbin=None,
                    nperbin=None,
                    min_lsigma=-1.,
                    max_lsigma=-0.1,
                    title=None,
                    show=False):
    import biggles
    import esutil as eu

    shear_true=data['shear_true'].mean(axis=0)

    colors=['blue','red']
    types=['filled circle','filled diamond']

    if use_true_fwhm:
        log_sigma = numpy.log10( data['fwhm']/2.35*0.27 )
    else:
        log_sigma = numpy.log10( numpy.sqrt( 0.5*data['T'] ) )

    weights = get_weights(data)

    bs1=eu.stat.Binner(log_sigma, data['g'][:,0], weights=weights)
    bs2=eu.stat.Binner(log_sigma, data['g'][:,1], weights=weights)

    bs1.dohist(min=min_lsigma,
               max=max_lsigma,
               nbin=nbin,
               nperbin=nperbin)
    bs2.dohist(min=min_lsigma,
               max=max_lsigma,
               nbin=nbin,
               nperbin=nperbin)

    bs1.calc_stats()
    bs2.calc_stats()

    plt=biggles.FramedPlot()
    plt.xrange=[0.9*min_lsigma, 1.1*max_lsigma]
    plt.aspect_ratio=1
    plt.title=title
    plt.xlabel=r'$log_{10}( \sigma [arcsec] )$'
    plt.ylabel='<e>'


    pts1=biggles.Points(bs1['xmean'],bs1['ymean'],type=types[0],color=colors[0])
    err1=biggles.SymmetricErrorBarsY(bs1['xmean'],bs1['ymean'],bs1['yerr'],
                                     color=colors[0])

    pts2=biggles.Points(bs2['xmean'],bs2['ymean'],type=types[1],color=colors[1])
    err2=biggles.SymmetricErrorBarsY(bs2['xmean'],bs2['ymean'],bs2['yerr'],
                                     color=colors[1])

    sc1 = biggles.Curve(bs1['xmean'], bs1['xmean']*0 + shear_true[0])
    sc2 = biggles.Curve(bs1['xmean'], bs1['xmean']*0 + shear_true[1])

    pts1.label=r'$e_1$'
    pts2.label=r'$e_2$'

    key=biggles.PlotKey(0.1,0.9,[pts1,pts2])

    plt.add(sc1,sc2,pts1,err1,pts2,err2,key)

    if yrange is not None:
        plt.yrange=yrange

    if show:
        plt.show()
    return plt


def add_cmodel_to_run(run, run_cm):
    """
    add the s2n_r and T_s2n_r from one run to another

    e.g. the first run might be sfit-eg01 and the second sfit-c05
    """
    import fitsio

    new_run = '%s-%s' % (run, run_cm)
    print("new run:",new_run)

    d=files.get_collated_dir(run=new_run, gnum=0)
    if not os.path.exists(d):
        os.makedirs(d)

    conf=files.read_config(run=run)
    for gnum in xrange(conf['ng']):
        print("gnum:",gnum)
        outfile=files.get_collated_file(run=new_run, gnum=gnum)
        print("outfile:",outfile)


        fname=files.get_collated_file(run=run, gnum=gnum)
        fname_cm=files.get_collated_file(run=run_cm, gnum=gnum)

        print("    reading:",fname)
        data=fitsio.read(fname)#, rows=range(1000))
        print("    reading:",fname_cm)
        data_cm=fitsio.read(fname_cm)#, rows=range(1000))

        data['flags_r']=1
        data['s2n_r'] = -9999.0
        data['T_s2n_r'] = -9999.0

        for fnum in xrange(conf['nf']):
            print("        fnum:",fnum,)

            w,=where(data['fnum']==fnum)
            w_cm,=where(data_cm['fnum']==fnum)

            if w.size > 0 and w_cm.size > 0:
                m,m_cm=eu.numpy_util.match(data['number'][w],
                                           data_cm['number'][w_cm])

                print("%d/%d" % (m.size, w.size))
                if m.size > 0:
                    ind = w[m]
                    ind_cm = w_cm[m_cm]
                    for field in ['flags_r','s2n_r','T_s2n_r']:
                        data[field][ind] = data_cm[field][ind_cm]
        
        print("writing:",outfile)
        fitsio.write(outfile, data, clobber=True)
        #m, m_cm = eu.numpy_util.match(...)

