from __future__ import print_function
import os
import numpy
from numpy import array, diag, sqrt, newaxis, where, log10, ones, zeros, exp
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

        names=data.dtype.names
        fields2add = []
        if 'mcal_T_r' in names and 'mcal_psf_T_r' in names:
            do_Trat=True
            fields2add += [('Trat_r','f4')]
        else:
            do_Trat=False

        if 'mcal_pars' in names:
            do_flux=True
            fields2add += [('mcal_log_flux','f4')]
        else:
            do_flux=True

        if len(fields2add) > 0:
            print("adding fields:",fields2add)
            data = eu.numpy_util.add_fields(data, fields2add)

        if do_Trat:
            Trat = data['mcal_T_r']/data['mcal_psf_T_r']
            data['Trat_r'] = Trat.astype('f4')

        if do_flux:
            log_flux = data['mcal_pars'][:,5]
            data['mcal_log_flux'] = log_flux.astype('f4')

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
                min_Trat_r=None,
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


    if min_Trat_r is not None:
        Trat = data['Trat_r']

        elogic = (Trat > min_Trat_r)
        w,=where(elogic)
        if w.size != data.size:
            print("    kept %d/%d from round Tg/Tpsf > %g" % (w.size, data.size, min_Trat_r))
            logic = logic & elogic



    w,=where(logic)
    print("    kept %d/%d" % (w.size, data.size))

    data=data[w]
    return data

def quick_analyze_mcal(run, deep_run, vs_field, wstyle=None,
                       s2n_range=None,
                       min_Trat=None,
                       dlist=None, deep_data=None):

    #kw={}
    #if vs_field == 'mcal_s2n_r':
    #    kw['s2n_range']=[10**0.95, 10**3.0]
    #elif vs_field == '

    if dlist is None:
        dlist, deep_data = quick_load_mcal(run, deep_run, wstyle=wstyle,
                                           s2n_range=s2n_range,
                                           min_Trat=min_Trat)

    a=Analyzer(dlist, deep_data=deep_data, wstyle=wstyle)

    return dlist, deep_data

def quick_load_mcal(run, deep_run, wstyle=None, s2n_range=None, min_Trat=None):
    #s2n_range=[10**0.95, 10**3.0]

    # for low s/n run
    cols = ['mcal_pars','mcal_g','mcal_s2n_r','shear_true','flags']
    kc   = ['mcal_g','mcal_s2n_r','shear_true','mcal_log_flux']

    # for deep data
    dcols=['mcal_pars','mcal_g','mcal_g_sens','mcal_s2n_r','flags']
    dkc  =['mcal_g_sens','mcal_s2n_r','mcal_log_flux']

    #if wstyle is not None:
    if True:
        cols  += ['mcal_pars_cov']
        kc    += ['mcal_pars_cov']

        dcols += ['mcal_pars_cov']
        dkc   += ['mcal_pars_cov']

    #if min_Trat is not None:
    if True:
        cols  += ['mcal_T_r','mcal_psf_T_r']
        kc    += ['Trat_r'] # this is created from above

        dcols += ['mcal_T_r','mcal_psf_T_r']
        dkc   += ['Trat_r']

    dlist=load_data(run,
                    columns=cols,keep_cols=kc,
                    mcal_s2n_r_range=s2n_range,
                    min_Trat_r=min_Trat,
                    g_field='mcal_g')

    deep_data=load_data(deep_run,
                        columns=dcols,keep_cols=dkc,
                        mcal_s2n_r_range=s2n_range,
                        min_Trat_r=min_Trat,
                        combine=True,
                        g_field='mcal_g')

    return dlist, deep_data

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

    def fit_m_c(self, show=False, doprint=True,
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

        nwalkers, burnin, nstep = 200, 2000, 2000
        lf1=fitting.fit_line(gtrue[:,0], gdiff[:,0], yerr=gdiff_err[:,0],
                             method='mcmc',
                             nwalkers=nwalkers,burnin=burnin,nstep=nstep)
        lf2=fitting.fit_line(gtrue[:,1], gdiff[:,1], yerr=gdiff_err[:,1],
                             method='mcmc',
                             nwalkers=nwalkers,burnin=burnin,nstep=nstep)
        fitters=[lf1,lf2]

        res1=lf1.get_result()
        res2=lf2.get_result()

        m1,c1 = res1['pars']
        m1err,c1err = res1['perr']
        m2,c2 = res2['pars']
        m2err,c2err = res2['perr']

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
                               deep_data=tdeep_data)

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

    def plot_m_c_vs(self, res, name, xlog=True, show=False, combine=False, xrng=None):
        """
        result from fit_m_c_vs

        parameters
        ----------
        res: dict
            result from running fit_m_c_vs
        name: string
            Name for the variable binned against, e.g. s/n or whatever
            This is used for the x axis
        xlog: bool, optional
            If True, use a log x axis
        show: bool, optional
            If True, show the plot on the screen

        returns
        -------
        biggles plot object
        """

        import biggles
        biggles.configure('default','fontsize_min',2.0)
        tab=biggles.Table(2,1)

        xvals = res['mean']

        if xrng is None:
            if xlog:
                xrng=[0.5*xvals.min(), 1.5*xvals.max()]
            else:
                xrng=[0.9*xvals.min(), 1.1*xvals.max()]

        mplt=biggles.FramedPlot()
        mplt.xlabel=name
        mplt.ylabel='m'
        mplt.xrange=xrng
        mplt.yrange=[-0.01,0.01]
        mplt.xlog=xlog

        cplt=biggles.FramedPlot()
        cplt.xlabel=name
        cplt.ylabel='c'
        cplt.xrange=xrng
        cplt.yrange=[-0.0015,0.0015]
        cplt.xlog=xlog

        if combine:

            m = 0.5*(res['m1'] + res['m2'])
            c = 0.5*(res['c1'] + res['c2'])

            merr = array([min(m1err,m2err) for m1err,m2err in zip(res['m1err'],res['m2err'])])
            cerr = array([min(c1err,c2err) for c1err,c2err in zip(res['c1err'],res['c2err'])])

            merr /= sqrt(2)
            cerr /= sqrt(2)

            mc=biggles.Points(xvals, m, type='filled circle', color='blue')
            merrc=biggles.SymmetricErrorBarsY(xvals, m, merr, color='blue')


            cc=biggles.Points(xvals, c, type='filled circle', color='blue')
            cerrc=biggles.SymmetricErrorBarsY(xvals, c, cerr, color='blue')


            zc=biggles.Curve(xrng, [0,0])

            mplt.add( zc, mc, merrc )
            cplt.add( zc, cc, cerrc )

        else:
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

            mplt.add( zc, m1c, m1errc, m2c, m2errc, mkey )

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

        print("getting weights")
        wts=self.get_weights(data)
        if wts is None:
            print("    weights are None")

        if deep_data is not None:
            print("getting deep data weights")
            deep_wts = self.get_weights(deep_data)
            if deep_wts is None:
                print("    weights are None")

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

        return get_weights(data, field=self.gcov_field, wstyle=self.wstyle)

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


def get_weights(data, field='g_cov', wstyle='tracecov'):
    """
    Get shear weights for the input data

    parameters
    ----------
    data: array
        The input data

    wstyle: string, optional
        Type of weights to use ['tracecov','s2n_r'].  Default 'tracecov'
        's2n_r' is not correct yet

    field: string, optional
        The field to use for covariance weights, default 'g_cov'
    """

    if wstyle=='tracecov':

        if 'g_cov' in field:
            gcov = data[field]
        else:
            cov = data[field]
            gcov = cov[:,2:2+2, 2:2+2]

        csum=gcov[:,0,0] + gcov[:,1,1]

        wts=1.0/(2*SN**2 + csum)

    elif wstyle == 's2n_r':
        raise NotImplementedError("tune s2n weighting")
        wts=1.0/(2*SN**2 + 2/data['s2n_r']**2)

    elif wstyle is None:

        wts=ones(data.size)

    else:
        raise ValueError("bad weights style: '%s'" % wstyle)

    return wts

