from __future__ import print_function
import os


def get_data_dir():
    """
    the main data directory
    """
    d=os.environ['GDES_DATA_DIR']
    return d

def get_config_dir():
    """
    the main data directory
    """
    d=os.environ['GDES_CONFIG_DIR']
    return d


def get_config_file(**keys):
    """
    get a config file path
    """
    if 'run' in keys:
        run=keys['run']
        fname='%s.yaml' % run
    elif 'name' in keys:
        name=keys['name']
        fname='%s.yaml' % name
    else:
        raise ValueError("config needs to have 'run' or 'name' in it")

    d=get_config_dir()
    fname=os.path.join(d, fname)
    return fname

def read_config(**keys):
    """
    read a config file
    """
    import yaml
    fname=get_config_file(**keys)
    print("reading:",fname)
    data= yaml.load( open(fname) )

    if 'run' in keys:
        d=data['run']
        check=keys['run']
    elif 'name' in keys:
        d=data['name']
        check=keys['name']
    else:
        raise ValueError("config needs to have 'run' or 'name' in it")
    if d != check:
        raise ValueError("run or name '%s' doesn't "
                         "match '%s'" % (d,check))
    return data

def get_input_dir(**keys):
    """
    parameters
    ----------
    gdrun: keyword
        The great des run e.g. nbc-sva1-001
    """
    d=get_data_dir()
    d=os.path.join(d, keys['gdrun'], 'data')
    #d=os.path.join(h, 'lensing','great-des', keys['gdrun'])
    return d

def get_replace_lsf_dir(**keys):
    """
    parameters
    ----------
    gdrun: keyword
        The great des run e.g. nbc-sva1-001
    """
    d=get_data_dir()
    d=os.path.join(d, keys['gdrun'], 'lsf')
    return d

def get_replace_lsf_file(**keys):
    """
    parameters
    ----------
    gdrun: keyword
        The great des run e.g. nbc-sva1-001
    gnum: keyword
        The shear number
    fnum: keyword
        The file number
    """

    d=get_replace_lsf_dir(**keys)

    noisefree=keys.get("noisefree",False)
    if noisefree and ftype=='meds':
        fname='rep.%(fnum)03i.g%(gnum)02i.noisefree.lsf'
    else:
        fname='rep.%(fnum)03i.g%(gnum)02i.lsf'
    fname=fname % keys
    fname=os.path.join(d, fname)
    return fname


def get_input_file(**keys):
    """
    parameters
    ----------
    gdrun: keyword
        The gdrun e.g. nbc-sva1-001
    ftype: keyword
        The file type, e.g. 'meds' 'truth'
    fnum: keyword
        The file number within given g set
    gnum: keyword
        The g (shear) number set

    noisefree: bool
        If true, return path to noisefree data; meds only.

    meds_ext: string, optional
        Extension, e.g. fits of fits.fz
    """
    d=get_input_dir(**keys)

    noisefree=keys.get("noisefree",False)
    ftype=keys['ftype']


    front=get_front(**keys)

    if noisefree and ftype=='meds':
        bname=front+'.%(ftype)s.%(fnum)03i.g%(gnum)02i.noisefree'
    else:
        bname=front+'.%(ftype)s.%(fnum)03i.g%(gnum)02i'

    bname=bname % keys

    if ftype=='meds':
        ext=keys.get('meds_ext','fits')
    else:
        ext='fits'

    bname='%s.%s' % (bname, ext)

    fname=os.path.join(d, bname)
    return fname

def get_front(**keys):
    gdrun=keys['gdrun']
    if gdrun=='nbc-sva1-001':
        front='nbc2'
    else:
        front='nbc'
    return front

def read_input_file(**keys):
    """
    parameters
    ----------
    gdrun: keyword
        The gdrun e.g. nbc-sva1-001
    ftype: keyword
        The file type, e.g. 'meds' 'truth'
    fnum: keyword
        The file number within given g set
    gnum: keyword
        The g (shear) number set

    noisefree: bool
        If true, return path to noisefree data; meds only.
    """
    import fitsio

    fname=get_input_file(**keys)
    return fitsio.read(fname)

def get_psf_file(**keys):
    """
    parameters
    ----------
    gdrun: keyword
        The gdrun e.g. nbc-sva1-001
    res: keyword
        The res, default 'lores'
    """

    d=get_input_dir(**keys)

    if 'res' not in keys:
        keys['res'] = 'lores'

    front=get_front(**keys)

    fname=front+'.psf.%(res)s.fits'
    fname=fname % keys

    return os.path.join(d,fname)

def count_fnums(**keys):
    """
    count the number of files per shear

    parameters
    ----------
    gdrun: keyword
        The gdrun e.g. nbc-sva1-001

    noisefree: bool
        If true, return path to noisefree data; meds only.
    """

    count=0
    while True:
        f=get_input_file(ftype='meds',
                         gnum=0,
                         fnum=count,
                         **keys)
        if os.path.exists(f):
            count += 1
        else:
            break

    return count

def count_gnums(**keys):
    """
    count the number of shears

    parameters
    ----------
    gdrun: keyword
        The gdrun e.g. nbc-sva1-001

    noisefree: bool
        If true, return path to noisefree data; meds only.
    """

    count=0
    while True:
        f=get_input_file(ftype='meds',
                         gnum=count,
                         fnum=0,
                         **keys)
        if os.path.exists(f):
            count += 1
        else:
            break

    return count


def get_output_dir(**keys):
    """
    parameters
    ----------
    run: keyword
        The processing run
    """
    d=get_data_dir()
    d=os.path.join(d, keys['run'], 'output')
    return d

def get_prior_dir(**keys):
    """
    parameters
    ----------
    run: keyword
        The processing run
    """
    d=get_data_dir()
    d=os.path.join(d, keys['run'], 'prior')
    return d


def get_collated_dir(**keys):
    """
    parameters
    ----------
    run: keyword
        The processing run
    """
    d=get_data_dir()
    d=os.path.join(d, keys['run'], 'collated')
    return d


def get_condor_dir(**keys):
    """
    parameters
    ----------
    run: keyword
        The processing run
    """
    d=get_data_dir()
    d=os.path.join(d, keys['run'],'condor')
    return d

def get_condor_master(**keys):
    """
    parameters
    ----------
    run
    """
    d=get_condor_dir(**keys)

    fname='master.sh'
    fname=os.path.join(d, fname)

    return fname


def get_condor_file(**keys):
    """
    parameters
    ----------
    run
    fnum
    gnum
    start
    end
    """
    d=get_condor_dir(**keys)

    missing=keys.get('missing',False)
    if missing:
        fname='%(run)s-%(filenum)05d-missing.condor'
    else:
        fname='%(run)s-%(filenum)05d.condor'
    fname=fname % keys

    fname=os.path.join(d, fname)

    return fname

def get_lsf_dir(**keys):
    """
    parameters
    ----------
    run: keyword
        The processing run
    """
    d=get_data_dir()
    d=os.path.join(d, keys['run'],'lsf')
    return d

def get_lsf_master(**keys):
    """
    parameters
    ----------
    run
    """
    d=get_lsf_dir(**keys)

    fname='master.sh'
    fname=os.path.join(d, fname)

    return fname


def get_lsf_file(**keys):
    """
    parameters
    ----------
    run
    fnum
    gnum
    start
    end
    missing
    """
    d=get_lsf_dir(**keys)

    missing=keys.get('missing',False)
    if missing:
        keys['back']='-missing.lsf'
    else:
        keys['back']='.lsf'

    fname='%(run)s-%(fnum)03d-g%(gnum)02d-%(start)05d-%(end)05d%(back)s'
    fname=fname % keys

    fname=os.path.join(d, fname)

    return fname




def get_wq_dir(**keys):
    """
    parameters
    ----------
    run: keyword
        The processing run
    """
    d=get_data_dir()
    d=os.path.join(d,keys['run'],'wq')
    return d

def get_wq_file(**keys):
    d=get_wq_dir(**keys)

    start=keys['start']
    end=keys['end']

    fname='%(run)s-%(fnum)03i-g%(gnum)02i-%(start)05d-%(end)05d.yaml'
    fname=fname % keys

    fname=os.path.join(d, fname)

    return fname

def get_collate_wq_file(**keys):
    d=get_wq_dir(**keys)

    fname='%(run)s-g%(gnum)02i-collate.yaml'
    fname=fname % keys

    fname=os.path.join(d, fname)

    return fname


def get_output_file(**keys):
    """
    parameters
    ----------
    run: string, keyword
        String representing the run, e.g. nfit-noisefree-04
    gnum: int, keyword
        Integer representing the shear number.
    fnum: int
        File number for given gnum
    start: int
        Integer representing starting object number
    end: int
        Integer representing ending object number
    """
    d=get_output_dir(**keys)

    fname='%(run)s-%(fnum)03i-g%(gnum)02i-%(start)05d-%(end)05d.fits'
    fname=fname % keys

    fname=os.path.join(d, fname)

    return fname

def get_prior_file(**keys):
    """
    parameters
    ----------
    run: string, keyword
        String representing the run, e.g. nfit-noisefree-04
    partype: string, keyword
        Something extra to identify this
    ext: string, keyword, optional
        Extension for file, default 'fits' 
    """
    d=get_prior_dir(**keys)

    fname='%(run)s-%(partype)s'
    fname=fname % keys

    ext=keys.get('ext','fits')
    fname = '%s.%s' % (fname, ext)

    fname=os.path.join(d, fname)

    return fname

def read_prior(**keys):
    """
    parameters
    ----------
    run: string, keyword
        String representing the run, e.g. sfit-noisefree-c01
    partype: string, keyword
        Something extra to identify this
    ext: string, keyword, optional
        Extension for file, default 'fits' 
    """
    import fitsio
    from ngmix.gmix import GMixND

    fname=get_prior_file(**keys)
    print("reading:",fname)
    data = fitsio.read(fname)

    prior = GMixND(data['weights'],
                   data['means'],
                   data['covars'])
    return prior
    

def get_collated_file(**keys):
    """
    parameters
    ----------
    run: string, keyword
        String representing the run, e.g. nfit-noisefree-04
    gnum: int, keyword
        Integer representing the shear number.
    """
    d=get_collated_dir(**keys)

    fname='%(run)s-g%(gnum)02i.fits'
    fname=fname % keys

    fname=os.path.join(d, fname)

    return fname

def read_collated(**keys):
    """
    parameters
    ----------
    run: string, keyword
        String representing the run, e.g. nfit-noisefree-04
    gnum: int, keyword
        Integer representing the shear number.
    """
    import fitsio

    fname=get_collated_file(**keys)
    print("reading:",fname)
    return fitsio.read(fname)

def read_output(**keys):
    """
    parameters
    ----------
    run: string, keyword
        String representing the run, e.g. nfit-noisefree-04
    gnum: int, keyword
        Integer representing the shear number.
    fnum: int, keyword
        File number for given gnum
    start: int, optional keyword
        Integer representing starting object number
    end: int, optional keyword
        Integer representing ending object number
    """

    import fitsio
    fname=get_output_file(**keys)
    print('reading:',fname)
    data = fitsio.read(fname)
    return data


def get_averaged_file(**keys):
    """
    parameters
    ----------
    run: string, keyword
        String representing the run, e.g. nfit-noisefree-04
    gnum: int, keyword
        Integer representing the shear number.
    """

    d=get_collated_dir(**keys)
    
    if 'gnum' in keys:
        fname='%(run)s-g%(gnum)02i-avg.fits'
    else:
        fname='%(run)s-avg.fits'
    fname=fname % keys

    fname=os.path.join(d, fname)

    return fname

def read_averaged(**keys):
    """
    parameters
    ----------
    run: string, keyword
        String representing the run, e.g. nfit-noisefree-04
    gnum: int, keyword
        Integer representing the shear number.
    """

    import fitsio
    fname=get_averaged_file(**keys)
    print('reading:',fname)
    data = fitsio.read(fname)
    return data


def get_chunk_ranges(**keys):
    import meds
    nper=int(keys['nper'])
    
    keys['ftype']='meds'
    meds_file=get_input_file(**keys)

    m=meds.MEDS(meds_file)

    ntotal=m.size

    nchunks = ntotal/nper
    nleft = ntotal % nper

    if nleft != 0:
        nchunks += 1

    low=[]
    high=[]

    for i in xrange(nchunks):

        low_i = i*nper

        # minus one becuase it is inclusive
        if i == (nchunks-1) and nleft != 0:
            high_i = low_i + nleft -1
        else:
            high_i = low_i + nper  - 1

        low.append( low_i )
        high.append( high_i )

    return low,high


def get_shear_name_dict(model=None):
    # add as many as you need here
    names=['nuse','shear','shear_cov','shear_err',
           'P','Q','R','flags']

    ndict={}
    if model is not None:
        for n in names:
            name='%s_%s' % (model,n)
            ndict[n] = name
    else:
        for n in names:
            ndict[n] = n
    return ndict


