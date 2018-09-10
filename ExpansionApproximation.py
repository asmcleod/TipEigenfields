import os
import numpy
import pickle
from matplotlib.pyplot import *
from common.log import Logger
from common import numerical_recipes as numrec
from common import numerics as num
from common.baseclasses import AWA
from numpy.linalg import linalg
from scipy.interpolate import RectBivariateSpline,interp2d
from scipy.integrate import simps,trapz
from scipy.signal import invres,unique_roots
from NearFieldOptics import TipModels as tip
from NearFieldOptics import Materials as mat

root_dir=os.path.dirname(__file__)

def get_tip_eigenbasis_expansion(z=.1,freq=1000,a=30,\
                                 smoothing=0,reload_signal=True,\
                                 *args,**kwargs):
    """Appears to render noise past the first 15 eigenvalues.  
    Smoothing by ~4 may be justified, removing zeros probably not..."""
    
    # Rely on Lightning Rod Model only to load tip data from file in its process of computing a signal #
    if reload_signal:
        tip.verbose=False
        signal=tip.LRM(freq,rp=mat.Au.reflection_p,zmin=z,amplitude=0,\
                       normalize_to=None,normalize_at=1000,Nzs=1,demodulate=False,\
                       *args,**kwargs)
        tip.verbose=True

    global L,g,M,alphas,P,Ls,Es

    #Get diagonal basis for the matrix
    L=tip.LRM.LambdaMatrix(tip.LRM.qxs)
    g=numpy.matrix(numpy.diag(-tip.LRM.qxs*numpy.exp(-2*tip.LRM.qxs*z/numpy.float(a))*tip.LRM.wqxs))
    M=AWA(L*g,axes=[tip.LRM.qxs]*2,axis_names=['q']*2)
    
    #Smooth along s-axis (first), this is where we truncated the integral xform
    if smoothing: M=numrec.smooth(M,axis=0,window_len=smoothing) 
        
    alphas,P=linalg.eig(numpy.matrix(M))
    P=numpy.matrix(P)
    
    Ls=numpy.array(P.getI()*tip.LRM.Lambda0Vector(tip.LRM.qxs)).squeeze()
    Es=numpy.array(numpy.matrix(tip.LRM.get_dipole_moments(tip.LRM.qxs))*g*P).squeeze()
    
    Rs=-Es*Ls/alphas**2
    Ps=1/alphas
    
    return {'Rs':Rs,'Ps':Ps,'Es':Es,'alphas':alphas,'Ls':Ls}

def VisualizeEigenfieldDistributions(ns=[1,2,3],zs=[.1,5,15,30],a=30,*args,**kwargs):
    
    zs.sort()
    eigs=dict([(n,[]) for n in ns])
    for i,z in enumerate(zs):
        if i==0: kwargs['reload_signal']=True
        else: kwargs['reload_signal']=False
        d=get_tip_eigenbasis_expansion(z,a=a,*args,**kwargs)
        for n in ns:
            eigs[n].append(AWA(P[:,n-1],axes=[tip.LRM.qxs],axis_names=['$q/a$']).squeeze())
            
    colors=list(zip(numpy.linspace(1,0,len(zs)),\
               [0]*len(zs),\
               numpy.linspace(0,1,len(zs))))
            
    #return eigs
            
    figure()
    for i,n in enumerate(ns):
        subplot(1,len(ns),i+1)
        for j in range(len(zs)):
            numpy.abs(eigs[n][j]).plot(plotter=semilogx,\
                                       color=colors[j],\
                                       label='$z/a=%1.1f$'%(zs[j]/numpy.float(a)))
        
        props = dict(boxstyle='round', facecolor='white', alpha=0.8)
        gca().text(0.67, 0.85, '$n=%i$'%n, transform=gca().transAxes, fontsize=24,bbox=props)
        
        if i==0: ylabel('$E_n(q)$')
        else:
            yticks(numpy.arange(5)*.05,['']*5)
            xts=xticks()
            xticks(xts[0][1:])
            
        xlim(1e-4,20)
        ylim(0,.2)
        grid()
            
    tight_layout()
    subplots_adjust(wspace=0)
    

def VisualizeEigenfield(n=3,z=30,scale=50,\
                        a=30,*args,**kwargs):
    
    zs=numpy.linspace(0,a*scale,200)
    rs=numpy.linspace(-a*scale/2.,a*scale/2.,200)
    from common.numerics import broadcast_items
    from scipy.special import j0,j1
    
    d=get_tip_eigenbasis_expansion(z,a=a,*args,**kwargs)
    field_contributions=AWA(P[:,n-1],axes=[tip.LRM.qxs],axis_names=['$q/a$']).squeeze()
    
    zs_norm=zs/float(a)
    rs_norm=rs/float(a)
    rs_norm,zs_norm=broadcast_items(rs_norm,zs_norm)
    
    total_zfield=0
    total_rfield=0
    potential=0
    for field_contrib,q,wq in zip(field_contributions,tip.LRM.qxs,tip.LRM.wqxs):
        
        zpos_dependence=numpy.exp(-q*z/numpy.float(a))*numpy.exp(-q*zs_norm)*j0(q*rs_norm)
        rpos_dependence=numpy.exp(-q*z/numpy.float(a))*numpy.exp(-q*zs_norm)*j1(q*rs_norm)
        
        total_zfield+=(-field_contrib)*zpos_dependence*wq
        total_rfield+=(-field_contrib)*rpos_dependence*wq
        potential+=field_contrib*zpos_dependence*wq/q
        
    total_zfield=AWA(total_zfield,axes=[rs,zs],axis_names=['r','z'])
    total_rfield=AWA(total_rfield,axes=[rs,zs],axis_names=['r','z'])
    potential=AWA(potential,axes=[rs,zs],axis_names=['r','z'])
    potential-=numpy.mean(potential.cslice[:,z])
    
    return total_zfield,total_rfield,potential

def GetExpansionApproachCurve(zs=numpy.linspace(.1,150,50),a=30,beta=1,Nterms=None,\
                              *args,**kwargs):
    
    if hasattr(beta,'__len__'):
        if not isinstance(beta,numpy.ndarray): beta=numpy.array(beta)
        beta=beta.reshape((len(beta),1,1))
    
    Rs=[]
    Ps=[]
    for i,z in enumerate(zs):
        if i==0: kwargs['reload_signal']=True
        else: kwargs['reload_signal']=False
        d=get_tip_eigenbasis_expansion(z,a=a,*args,**kwargs)
        Rs.append(d['Rs'])
        Ps.append(d['Ps'])
        
    Rs=numpy.array(Rs); Ps=numpy.array(Ps)
    if not Nterms: Nterms=Ps.shape[1]
    Rs=AWA(Rs[:,:Nterms],axes=[zs,None],axis_names=['Z','Term'])
    Ps=AWA(Ps[:,:Nterms],axes=[zs,None],axis_names=['Z','Term'])
    
    approach=numpy.sum(Rs/(beta-Ps)+Rs/Ps,axis=-1)
    #beta*Rs/(poles-beta)=sig
    #0=sig+beta*Rs/(beta-poles)
    #0=sig/beta+Rs/(beta-poles)
    
    axes=[zs]; axis_names=['Z']
    if hasattr(beta,'__len__'):
        approach=approach.transpose()
        if isinstance(beta,AWA):
            axes=axes+[beta.axes[0]]
            axis_names=axis_names+[beta.axis_names[0]]
        else:
            axes=axes+[None]
            axis_names=axis_names+[None]
        
    approach=AWA(approach,axes=axes,axis_names=axis_names)
    
    return {'Rs':Rs,'Ps':Ps,'signals':approach}


class _EigenfieldModel_(object):
    
    verbose=True
    
    def __init__(self,zs=numpy.logspace(-3,2,400),Nterms_max=20,sort=True,
                *args,**kwargs):
        """For some reason using Nqs>=244, getting higher q-resolution,
        only makes more terms relevant, requiring twice as many terms for
        stability and smoothness in approach curves...
        (although overall """
        
        self.zs=zs
        self.Nterms_max=Nterms_max
        
        #Take a look at current probe geometry
        #Adjust quadrature parameters that assist in yielding smooth residues/poles
        #Pre-determined as good values for Nterms=10
        geometry=tip.LRM.geometric_params['geometry']
        if self.verbose:
            Logger.write('Setting up eigenfield tip model for geometry "%s"...'\
                         %geometry)
        if geometry is 'cone': tip.LRM.quadrature_params['b']=.5
        elif geometry is 'hyperboloid': tip.LRM.quadrature_params['b']=.5
        
        global Rs,Ps
        Rs=[]
        Ps=[]
        for i,z in enumerate(zs):
            
            if i==0: kwargs['reload_signal']=True
            elif i==1: kwargs['reload_signal']=False
            
            d=get_tip_eigenbasis_expansion(z,a=1,*args,**kwargs)
            
            Rrow=d['Rs']
            Prow=d['Ps']
            Rrow[numpy.isnan(Rrow)]=Rrow[numpy.isfinite(Rrow)][-1]
            Prow[numpy.isnan(Prow)]=Prow[numpy.isfinite(Prow)][-1]
            
            Rrow=Rrow[Prow.real>0]
            Prow=Prow[Prow.real>0]
            
            where_unphys_Ps=(Prow.imag>0)
            Prow[where_unphys_Ps]-=1j*Prow[where_unphys_Ps].imag
            
            Prow=Prow[:50]
            Rrow=Rrow[:50]
            
            if i>1 and sort:
                #Ensure continuity with previous poles (could have done residues instead, but this works)
                sorting=numpy.array([numpy.argmin(
                                            numpy.abs((Prow-previous_P)-\
                                                      (previous_P-preprevious_P)))
                                     for previous_P,preprevious_P in zip(previous_Prow,preprevious_Prow)])
                Prow=Prow[sorting]
                Rrow=Rrow[sorting]
            
            Rs.append(Rrow[:Nterms_max])
            Ps.append(Prow[:Nterms_max])
            
            #Make sure to keep a reference to previous set of poles and residues
            if sort:
                previous_Prow=Prow
                previous_Rrow=Rrow
                if i>=1:
                    preprevious_Prow=previous_Prow
                    preprevious_Rrow=previous_Rrow
        
        terms=numpy.arange(Nterms_max)+1
        self.Rs=AWA(Rs,axes=[zs,terms],axis_names=['z/a','Term'])
        self.Ps=AWA(Ps,axes=[zs,terms],axis_names=['z/a','Term'])
        
        ##Remove `nan`s from poles and residues##
        #Best way to remove `nan`s (=huge values) is to replace with largest finite value
        #Largest such value in Rs will be found in ratio to that in Ps, implying that
        #beta in denominator of that term is irrelevant, so term+offset goes to zero anyway..
        
        for j in range(Nterms_max):
            
            Rrow=self.Rs[:,j]; Prow=self.Ps[:,j]
            Rrow[numpy.isnan(Rrow)]=Rrow[numpy.isfinite(Rrow)][-1] #Highest value will be for largest z (end of array)
            Prow[numpy.isnan(Prow)]=Prow[numpy.isfinite(Prow)][-1]
        
        ##Remove any positive imaginary part from poles##
        #These are unphysical and are perhaps a by-product of inaccurate eigenvalues
        #when diagonalizing an almost-singular matrix (i.e. g*Lambda)
        #Just put them on the real line, at least it doesn't seem to hurt anything, at most it's more physical.
        where_unphys_Ps=(self.Ps.imag>0)
        self.Ps[where_unphys_Ps]-=1j*self.Ps[where_unphys_Ps].imag
        
        if self.verbose: Logger.write('\tDone.')
        
    def polyfit_poles_residues(self,deg=6,zmax=10):
        
        Nterms=self.Ps.shape[1]
        Rs=self.Rs.cslice[:zmax]
        Ps=self.Ps.cslice[:zmax]
        zs=Rs.axes[0]
        
        if self.verbose:
            Logger.write('Finding complex polynomial approximations of degree %i '%deg+\
                         'to the first %i poles and residues, up to a value z/a=%s...'%(Nterms,zmax))
            
        self.Ppolys=[]
        for i in range(Nterms):
            
            Ppoly=numpy.polyfit(zs,Ps[:,i],deg=deg)
            self.Ppolys.append(Ppoly)
        
        self.Rpolys=[]
        for i in range(Nterms):
            
            Rpoly=numpy.polyfit(zs,Rs[:,i],deg=deg)
            self.Rpolys.append(Rpoly)
        
    def evaluate_poles(self,zs=None,Nterms=None,interpolation='linear'):
        
        if zs is None: zs=self.zs
        
        #Default to all terms...
        if not Nterms:
            Nterms=numpy.min((self.Rs.shape[1],\
                              self.Nterms_max))
        
        Ps=self.Ps[:,:Nterms].interpolate_axis(zs,axis=0,kind=interpolation,
                                               bounds_error=False,extrapolate=True)
        
        return Ps
        
    def evaluate_residues(self,zs=None,Nterms=None,interpolation='linear'):
        
        if zs is None: zs=self.zs
        
        #Default to all terms...
        if not Nterms:
            Nterms=numpy.min((self.Rs.shape[1],\
                              self.Nterms_max))
        
        Rs=self.Rs[:,:Nterms].interpolate_axis(zs,axis=0,kind=interpolation,
                                               bounds_error=False,extrapolate=True)
        
        return Rs
    
    def evaluate_poles_poly(self,zs=None,Nterms=None,*args):
        
        if zs is None: zs=self.zs
        
        #Default to all terms...
        if not Nterms:
            Nterms=numpy.min((self.Rs.shape[1],\
                              self.Nterms_max))
        
        Ps=numpy.array([numpy.polyval(Ppoly,zs) for Ppoly in self.Ppolys[:Nterms]])
        
        return Ps.T

    def evaluate_residues_poly(self,zs=None,Nterms=None,*args):
        
        if zs is None: zs=self.zs
        
        #Default to all terms...
        if not Nterms:
            Nterms=numpy.min((self.Rs.shape[1],\
                              self.Nterms_max))
        
        Rs=numpy.array([numpy.polyval(Rpoly,zs) for Rpoly in self.Rpolys[:Nterms]])
        
        return Rs.T
        
    #evaluate_poles=evaluate_poles_poly
    #evaluate_residues=evaluate_residues_poly
        
    def get_Erad(self,beta,zs=None,Nterms=None,interpolation='linear'):
        
        if hasattr(beta,'__len__'):
            if isinstance(beta,numpy.ndarray):
                if not beta.ndim: beta=beta.tolist()
            else: beta=numpy.array(beta)
        
        if isinstance(beta,numpy.ndarray): beta=beta.reshape((len(beta),1,1))
        
        if zs is None: zs=self.zs
        Rs=self.evaluate_residues(zs,Nterms)#,interpolation)
        Ps=self.evaluate_poles(zs,Nterms)#,interpolation)
        
        #Should broadcast over freqs if beta has an additional first axis
        #The offset term is absolutely critical, offsets false z-dependence arising from first terms
        approach=numpy.sum(Rs*(1/(beta-Ps)+1/Ps),axis=-1)
        
        axes=[zs]; axis_names=['z/a']
        if hasattr(beta,'__len__'):
            approach=approach.transpose()
            if isinstance(beta,AWA):
                axes=axes+[beta.axes[0]]
                axis_names=axis_names+[beta.axis_names[0]]
            else:
                axes=axes+[None]
                axis_names=axis_names+[None]
        
        signals=AWA(approach,axes=axes,axis_names=axis_names).squeeze()
        if not signals.ndim: signals=signals.tolist()
        
        return signals
    
    def get_demodulation_nodes(self,zmin=0,amplitude=2,quadrature=numrec.GL,\
                               harmonic=3,Nts=None):
        #GL quadrature is the best, can do even up to harmonic 3 with 6 points on e.g. SiO2
        #TS requires twice as many points
        #quadrature `None` needs replacing, this linear quadrature is terrible
        
        #max harmonic resolvable will by 1/dt=Nts
        if not Nts: Nts=2*harmonic
        if isinstance(quadrature,str) or hasattr(quadrature,'calc_nodes'):
            ts,wts=numrec.GetQuadrature(N=Nts,xmin=-.5,xmax=0,quadrature=quadrature)
            
        else:
            ts=numpy.linspace(-1+1/numpy.float(Nts),\
                              0-1/numpy.float(Nts),Nts)*.5
            wts=numpy.ones((Nts,))/numpy.float(Nts)*.5
        
        #This is what's necessary for fourier element
        #cos harmonic kernel, *2 for full period integration, *2 for coefficient
        wts*=4*numpy.cos(2*numpy.pi*harmonic*ts)
        zs=zmin+amplitude*(1+numpy.cos(2*numpy.pi*ts))
        
        return list(zip(zs,wts))
    
    def get_signal_from_nodes(self,beta,nodes=[(0,1)],Nterms=None,interpolation='linear'):
        
        if not hasattr(beta,'__len__'): beta=[beta]
        if not isinstance(beta,AWA): beta=AWA(beta)
        
        #`Frequency` axis will be first
        if isinstance(beta,numpy.ndarray): beta=beta.reshape((len(beta),1,1))
        
        #Weights apply across z-values
        zs,ws=list(zip(*nodes))
        ws_grid=numpy.array(ws).reshape((1,len(ws),1))
        zs=numpy.array(zs)
        
        #Evaluate at all nodal points
        Rs=self.evaluate_residues(zs,Nterms,interpolation)
        Ps=self.evaluate_poles(zs,Nterms,interpolation)
        
        #Should broadcast over freqs if beta has an additional first axis
        #The offset term is absolutely critical, offsets false z-dependence arising from first terms
        approach=numpy.sum(numpy.sum(Rs*(1/(beta-Ps)+1/Ps)*ws_grid,axis=-1),axis=-1)#+Rs/Ps,axis=-1)
        
        axes=[zs]; axis_names=['z/a']
        if hasattr(beta,'__len__'):
            approach=approach.transpose()
            if isinstance(beta,AWA):
                axes=axes+[beta.axes[0]]
                axis_names=axis_names+[beta.axis_names[0]]
            else:
                axes=axes+[None]
                axis_names=axis_names+[None]
        
        signals=AWA(approach); signals.adopt_axes(beta)
        signals=signals.squeeze()
        if not signals.ndim: signals=signals.tolist()
        
        return signals
    
    def __call__(self,*args,**kwargs): return self.get_Erad(*args,**kwargs)
    
    @staticmethod
    def demodulate(signals,amplitude=2,harmonics=list(range(4)),Nts=None,\
                   quadrature=numrec.GL):
        """Takes z-axis as first axis, frequency as final axis."""
    
        global ts,wts,weights,signals_vs_time
    
        #max harmonic resolvable will be frequency = 1/dt = Nts
        if not Nts: Nts=4*numpy.max(harmonics)
        if isinstance(quadrature,str) or hasattr(quadrature,'calc_nodes'):
            ts,wts=numrec.GetQuadrature(N=Nts,xmin=-.5,xmax=0,quadrature=quadrature)
            
        else:
            ts,wts=numpy.linspace(-.5,0,Nts),None
            if quadrature is None: quadrature=simps
        
        zs=amplitude*(1+numpy.cos(2*numpy.pi*ts))
        
        harmonics=numpy.array(harmonics).reshape((len(harmonics),1))
        weights=numpy.cos(2*numpy.pi*harmonics*ts)
        if wts is not None: weights*=wts
        weights_grid=weights.reshape(weights.shape+(1,)*(signals.ndim-1))
        
        signals_vs_time=signals.interpolate_axis(zs,axis=0,bounds_error=False,extrapolate=True)
        signals_vs_time.set_axes([ts],axis_names=['t'])
        integrand=signals_vs_time*weights_grid
        
        if wts is not None:
            demodulated=2*2*numpy.sum(integrand,axis=1) #perform quadrature
        else: demodulated=2*2*quadrature(integrand,x=ts,axis=1)
    
        axes=[harmonics]; axis_names=['harmonic']
        if isinstance(signals,AWA):
            axes+=signals.axes[1:]
            axis_names+=signals.axis_names[1:]
        demodulated=AWA(demodulated,axes=axes,axis_names=axis_names)
        
        return demodulated

    def invert_signal(self,signals,nodes=[(1,0)],Nterms=10,\
                      interpolation='linear',\
                      select_by='continuity',\
                      closest_pole=0,\
                      scaling=10):
        """The inversion is not unique, consequently the selected solution
        will probably be wrong if signal values correspond with 
        "beta" values that are too large (`|beta|~>min{|Poles|}`).
        This can be expected to break at around `|beta|>2`."""
        #Default is to invert signal in contact
        #~10 terms seem required to converge on e.g. SiO2 spectrum,
        #especially on the Re(beta)<0 (low signal) side of phonons
        
        global roots,poly,root_scaling
        
        #global betas,all_roots,pmin,rs,ps,As,Bs,roots,to_minimize
        if self.verbose:
            Logger.write('Inverting `signals` based on the provided `nodes` to obtain consistent beta values...')
        
        if not hasattr(signals,'__len__'): signals=[signals]
        if not isinstance(signals,AWA): signals=AWA(signals)
        
        zs,ws=list(zip(*nodes))
        ws_grid=numpy.array(ws).reshape((len(ws),1)) #last dimension is to broadcast over all `Nterms` equally
        zs=numpy.array(zs)
        
        Rs=self.Rs[:,:Nterms].interpolate_axis(zs,axis=0,kind=interpolation,
                                                       bounds_error=False,extrapolate=True)
        Ps=self.Ps[:,:Nterms].interpolate_axis(zs,axis=0,kind=interpolation,
                                                       bounds_error=False,extrapolate=True)
        
        #`rs` and `ps` can safely remain as arrays for `invres`
        rs=(Rs*ws_grid).flatten()
        ps=Ps.flatten()
        
        k0=numpy.sum(rs/ps).tolist()
        
        #Rescale units so their order of magnitude centers around 1
        rscaling=numpy.exp(-(numpy.log(numpy.abs(rs).max())+\
                            numpy.log(numpy.abs(rs).min()))/2.)
        pscaling=numpy.exp(-(numpy.log(numpy.abs(ps).max())+\
                             numpy.log(numpy.abs(ps).min()))/2.)
        root_scaling=1/pscaling
        #rscaling=1
        #pscaling=1
        if self.verbose:
            Logger.write('\tScaling residues by a factor %1.2e to reduce floating point overflow...'%rscaling)
            Logger.write('\tScaling poles by a factor %1.2e to reduce floating point overflow...'%pscaling)
        rs*=rscaling; ps*=pscaling
        k0*=rscaling/pscaling
        signals=signals*rscaling/pscaling
        
        #highest order first in `Ps` and `Qs`
        #VERY SLOW - about 100ms on practical inversions (~60 terms)
        As,Bs=invres(rs, ps, k=[k0], tol=1e-16, rtype='avg') #tol=1e-16 is the smallest allowable to `unique_roots`..
        
        dtype=numpy.complex128 #Double precision offers noticeable protection against overflow
        As=numpy.array(As,dtype=dtype)
        Bs=numpy.array(Bs,dtype=dtype)
        signals=signals.astype(dtype)
        
        #import time
        
        betas=[]
        for i,signal in enumerate(signals):
            #t1=time.time()
            
            #Root finding `roots` seems to give noisy results when `Bs` has degree >84, with dynamic range ~1e+/-30 in coefficients...
            #Pretty fast - 5-9 ms on practical inversions with rank ~60 companion matrices, <1 ms with ~36 terms
            #@TODO: Root finding chokes on `Nterms=9` (number of eigenfields) and `Nts=12` (number of nodes),
            #       necessary for truly converged S3 on resonant phonons, probably due to
            #       floating point overflow - leading term increases exponentially with
            #       number of terms, leading to huge dynamic range.
            #       Perhaps limited by the double precision of DGEEV.
            #       So, replace with faster / more reliable root finder?
            #       We need 1) speed, 2) ALL roots (or at least the first ~10 smallest)
            poly=As-signal*Bs
            roots=find_roots(poly,scaling=scaling)
            roots=roots[roots.imag>0]
            roots*=root_scaling #since all beta units scaled by `pscaling`, undo that here
            
            #print time.time()-t1
            
            #How should we select the most likely beta among the multiple solutions?
            #1. Avoids large changes in value of beta
            if select_by=='difference' and i>=1:
                if i==1 and self.verbose:
                    Logger.write('\tSelecting remaining roots by minimizing differences with prior...')
                to_minimize=numpy.abs(roots-betas[i-1])
                
            #2. Avoids large changes in slope of beta (best for spectroscopy)
            #Nearly guarantees good beta spectrum, with exception of very loosely sampled SiC spectrum
            #Loosely samples SiO2-magnitude phonons still perfectly fine
            elif select_by=='continuity' and i>=2:
                if i==2 and self.verbose:
                    Logger.write('\tSelecting remaining roots by ensuring continuity with prior...')
                earlier_diff=betas[i-1]-betas[i-2]
                current_diffs=roots-betas[i-1]
                to_minimize=numpy.abs(current_diffs-earlier_diff)
                
            #3. Select specifically which pole we want |beta| to be closest to
            else:
                if i==0 and self.verbose:
                    Logger.write('\tSeeding inversion closest to pole %i...'%closest_pole)
                reordering=numpy.argsort(numpy.abs(roots)) #Order the roots towards increasing beta
                roots=roots[reordering]
                to_minimize=numpy.abs(closest_pole-numpy.arange(len(roots)))
                
            beta=roots[to_minimize==to_minimize.min()].squeeze()
            betas.append(beta)
            if not i%5 and self.verbose:
                Logger.write('\tProgress: %1.2f%%  -  Inverted %i signals of %i.'%\
                                     (((i+1)/numpy.float(len(signals))*100),\
                                      (i+1),len(signals)))
        
        betas=AWA(betas); betas.adopt_axes(signals)
        betas=betas.squeeze()
        if not betas.ndim: betas=betas.tolist()
        
        return betas

EigenfieldModel=_EigenfieldModel_(zs=numpy.logspace(-3,2,100),Nqs=122)

def companion_matrix(p):
    """Assemble the companion matrix associated with a polynomial
    whose coefficients are given by `poly`, in order of decreasing
    degree.
    
    Currently unused, but might find a role in a custom polynomial
    root-finder.
    
    References:
    1) http://en.wikipedia.org/wiki/Companion_matrix
    """
    
    A = numpy.diag(numpy.ones((len(p)-2,), numpy.complex128), -1)
    A[0, :] = -p[1:] / p[0]

    return numpy.matrix(A)

def find_roots(p,scaling=10):
    """Find roots of a polynomial with coefficients `p` by
    computing eigenvalues of the associated companion matrix.
    
    Overflow can be avoided during the eigenvalue calculation
    by preconditioning the companion matrix with a similarity
    transformation defined by scaling factor `scaling` > 1.
    This transformation will render the eigenvectors in a new
    basis, but the eigenvalues will remain unchanged.
    
    Empirically, `scaling=10` appears to enable successful 
    root-finding on polynomials up to degree 192."""
    
    scaling=numpy.float64(scaling)
    M=companion_matrix(p); N=len(M)
    D=numpy.matrix(numpy.diag(scaling**numpy.arange(N)))
    Dinv=numpy.matrix(numpy.diag((1/scaling)**numpy.arange(N)))
    Mstar=Dinv*M*D # Similarity transform will leave the eigenvalues unchanged
    
    return linalg.eigvals(Mstar)

def uncoil_image(image):
    
    return numpy.hstack([row[::(-1)**i] for i,row in enumerate(image)])

def coil_image(uncoiled,row_len):
    
    Nrows=len(uncoiled)/row_len
    
    return numpy.vstack([uncoiled[i*row_len:(i+1)*row_len][::(-1)**i] for i in range(Nrows)])

def InvertImage(image,nodes,normalize_to=mat.Au,Nterms=8,freq=1000):
    
    row_len=image.shape[1]
    signals=uncoil_image(image)
    beta_norm=normalize_to.reflection_p(freq,q=1/(30e-7))
    norm_signal=EigenfieldModel.get_signal_from_nodes(beta=beta_norm,nodes=nodes,Nterms=Nterms)
    betas=EigenfieldModel.invert_signal(signals*norm_signal,nodes=nodes,Nterms=Nterms,select_by='pole')
    betas_image=coil_image(betas,row_len=row_len)
    
    if isinstance(image,AWA): betas_image=AWA(betas_image); betas_image.adopt_axes(image)
    
    return betas_image

quadratures=[('Trapezoid',(trapz,numpy.arange(2,200,2))),
             ('Simpson',(simps,numpy.arange(2,200,2))),
             ('Clenshaw Curtis',(numrec.CC,[6,8,10,12,24,48,96,192,192*2])),
             ('Gauss-Legendre',(numrec.GL,[6,12,24,48,96,192,192*2])),\
             ('Tanh-Sinh',(numrec.TS,[8,16,30,62,122,244,512]))]

def PlotHarmonicQuadratures(beta,amplitude=2,quadratures=quadratures,\
                               harmonic=3,**kwargs):
    
    global quad_harmonic,ref_harmonic,harmonic_value
    signal=EigenfieldModel(beta=beta)
    
    harmonics={}
    for quadrature_name,pair in quadratures:
        quadrature,Nts=pair
        harmonics_this_quad=[]
        for Nt in Nts:
            harmonics_this_Nt=EigenfieldModel.demodulate(signal,quadrature=quadrature,\
                                                           amplitude=amplitude,harmonics=[harmonic],Nts=Nt,**kwargs)
            harmonic_value=harmonics_this_Nt.cslice[harmonic].squeeze().tolist()
            harmonics_this_quad.append(numpy.complex(harmonic_value))
            
        harmonics_this_quad=numpy.array(harmonics_this_quad,dtype=numpy.complex)
        harmonics[quadrature_name]=AWA(harmonics_this_quad,axes=[Nts],axis_names=['$N_{quad}$'])
    
    import itertools
    
    figure()
    markers=[None,'o','s','^','+','^']
    markers = itertools.cycle(markers)
    
    ref_quad='Gauss-Legendre'
    ref_harmonic=harmonics[ref_quad][-1]
    
    for quadrature_name in zip(*quadratures)[0]:
        quad_harmonic=harmonics[quadrature_name]
        marker = next(markers)
        numpy.abs((quad_harmonic-ref_harmonic)/ref_harmonic)\
                    .plot(plotter=semilogy,marker=marker,label=quadrature_name,markersize=5)
        
    ylabel('$S_%i\,\mathrm{rel.\,error}$'%harmonic)
    leg=legend(loc='best',fancybox=True,shadow=True)
    leg.get_frame().set_linewidth(.1)
    xlim(0,200)
    
    tight_layout()
    
    return harmonics

def PlotPoleFits(Nterms=10,deg=6):
    
    Ps=EigenfieldModel.Ps
    
    figure()
    polys=[]
    for i in range(Nterms):
        Pterm=Ps[:,i].cslice[:10]
        zs=Pterm.axes[0]
        Pterm.real.plot(ls='-',label='Term %i'%(i+1))
        poly=numpy.polyfit(zs,Pterm,deg=deg)
        polys.append(poly)
        poly_vals=numpy.polyval(poly,zs)
        plot(zs,poly_vals.real,ls='--',color=gca().lines[-1].get_color())
        
    gca().set_yscale('symlog')
    gca().set_xscale('log')
    
    return polys
    
    
    
    #subplot(122)
    #for i in range(Nterms):
    #    Pterm=Ps[:,i].cslice[:10].imag
    #    zs=Pterm.axes[0]
    #    Pterm.plot(ls='-',label='Term %i'%(i+1))
    #    poly_real=numpy.polyfit(zs,Pterm,deg=deg)
    #    poly_vals=numpy.polyval(poly_real,zs)
    #    plot(zs,poly_vals,ls='--',color=gca().lines[-1].get_color())
    #    
    #gca().set_yscale('symlog')
    #gca().set_xscale('log')
    #tight_layout()
    
def PlotResidueFits(Nterms=10,deg=6):
    
    Rs=EigenfieldModel.Rs
    
    figure()
    subplot(121)
    for i in range(Nterms):
        Rterm=Rs[:,i].cslice[:10].real
        zs=Rterm.axes[0]
        Rterm.plot(ls='-',label='Term %i'%(i+1))
        poly_real=numpy.polyfit(zs,Rterm,deg=deg)
        poly_vals=numpy.polyval(poly_real,zs)
        plot(zs,poly_vals,ls='--',color=gca().lines[-1].get_color())
        
    gca().set_yscale('symlog')
    gca().set_xscale('log')
    
    subplot(122)
    for i in range(Nterms):
        Rterm=Rs[:,i].cslice[:10].imag
        zs=Rterm.axes[0]
        Rterm.plot(ls='-',label='Term %i'%(i+1))
        poly_real=numpy.polyfit(zs,Rterm,deg=deg)
        poly_vals=numpy.polyval(poly_real,zs)
        plot(zs,poly_vals,ls='--',color=gca().lines[-1].get_color())
        
    gca().set_yscale('symlog')
    gca().set_xscale('log')
    tight_layout()

def PlotExpansionOnApproach(zs=numpy.linspace(.1,300,100),a=30,beta=1,Nterms=5,\
                            *args,**kwargs):
    
    approach=GetExpansionApproachCurve(zs=zs,a=a,beta=1,Nterms=Nterms,*args,**kwargs)
    
    from matplotlib.collections import LineCollection
    from common import plotting #just for the BWR colormap
    
    figure()
    
    subplot(121)
    for i in range(Nterms):
        
        pole=approach['Ps'][:,i]
        x,y=pole.real,-pole.imag
        points = numpy.array([x, y]).T.reshape(-1, 1, 2)
        segments = numpy.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, cmap=get_cmap('jet_r'),
                            norm=Normalize(zs.min(),zs.max())) #use reversed colormap - lower values of freq are red
        lc.set_array(zs)
        lc.set_linewidth(2)
        gca().add_collection(lc)
        
    gca().set_xscale('log')
    gca().set_yscale('log')
    gca().set_aspect('equal')
    xlim(1e-1,1e4)
    xlabel(r'$\mathrm{Re}(\mathcal{P}_j)$')
    ylabel(r'$-\mathrm{Im}(\mathcal{P}_j)$')
    grid()
    
    subplot(122)
    for i in range(Nterms):
        
        pole=approach['Ps'][:,i]
        coeff=1j*approach['Rs'][:,i] #-1 is to put on same footing as dipole radiation, 1j was missing as prefactor to green's function
        #if i==0: p=numpy.angle(coeff[0])
        #coeff*=numpy.exp(-1j*p)
        
        #numpy.abs(coeff).plot(plotter=semilogx)
        #plot(coeff.real,coeff.imag,ls='',marker='')
        
        x,y=coeff.real,coeff.imag
        points = numpy.array([x, y]).T.reshape(-1, 1, 2)
        segments = numpy.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, cmap=get_cmap('jet_r'),
                            norm=Normalize(zs.min(),zs.max())) #use reversed colormap - lower values of freq are red
        lc.set_array(zs)
        lc.set_linewidth(2)
        gca().add_collection(lc)
        
    gca().set_aspect('equal')
    xlim(-.15e8,1.35e8)
    ylim(-.1e8,1.1e8)
    xticks(numpy.array([0,2.5,5,7.5,10,12.5])*1e7,\
           ['0','2.5','5','7.5','10','12.5'])
    yticks(numpy.array([0,2.5,5,7.5,10])*1e7,\
           ['0','2.5','5','7.5','10'])
    xlabel(r'$\mathrm{Re}(\mathcal{R}_j)\,[a.u.]$')
    ylabel(r'$\mathrm{Im}(\mathcal{R}_j)\,[a.u.]$')
    grid()
    
    gcf().set_size_inches([12.5,6],forward=True)
    subplots_adjust(wspace=.35)
    
    #Put colorbar for z-height colors#
    from matplotlib.colorbar import ColorbarBase
    cax=gcf().add_axes([.275,.265,.16,.25/7.])
    norm=Normalize(vmin=0,vmax=zs.max()/numpy.float(a))
    cbar=ColorbarBase(cax,cmap=cm.get_cmap('jet_r'),norm=norm,orientation='horizontal')
    gca().xaxis.set_label_position('top')
    gca().xaxis.set_ticks_position('top')
    cbar.set_ticks((0,2.5,5,7.5,10))
    xlabel('$d/a$',va='bottom',fontsize=18,labelpad=7) #labelpad to raise it a little, otherwise bbox patch will overlap xticks
    props=dict(boxstyle='round', facecolor='white', linewidth=0 , alpha=0.8)
    gca().xaxis.get_label().set_bbox(props)
    xticks(fontsize=13)
    for t in xticks()[1]: t.set_bbox(props)
    
    return approach

def PlotMaterialBetas():
    
    figure()
    
    beta_sic=mat.SiC_6H_Ellips.reflection_p(numpy.linspace(730,1150,200),q=1/(30e-7))
    beta_PMMA=mat.PMMA.reflection_p(numpy.linspace(1050,2000,300),q=1/(30e-7))
    
    axhline(0,ls='-',color='c',lw=3,label='Gold')
    axhline(1,ls='--',color='c',lw=3)
    
    beta_PMMA.imag.plot(ls='-',color='g',lw=3,label='PMMA')
    numpy.abs(beta_PMMA).cslice[1550:].plot(ls='--',color='g',lw=3)
    
    beta_sic.imag.plot(ls='-',color='r',lw=3,label='SiC')
    numpy.abs(beta_sic).plot(ls='--',color='r',lw=3)
    
    leg=legend(loc='upper right',shadow=True,fancybox=True)
    leg.get_frame().set_linewidth(.1)
    for t in leg.texts: t.set_fontsize(18)
    gca().set_yscale('log')
    
    ylim(1.5e-3,25)
    ylabel(r'$\beta$')
    xlabel('$\omega\,[cm^{-1}]$')
    
    twinx()
    axhline(-1,ls='-',color='r',lw=3,label=r'$\mathrm{Im}(\beta)$')
    axhline(-1,ls='--',color='r',lw=3,label=r'$|\beta|$')
    
    leg=legend(loc='upper center',shadow=True,fancybox=True)
    leg.get_frame().set_linewidth(.1)
    for t in leg.texts: t.set_fontsize(20)
    yticks([])
    
    ylim(0,1)
    xlim(750,1875)

def PlotApproachCurves(zmax=30,materials={'Gold':(mat.Au,1000),\
                                          'Carbonyl':(mat.PMMA,1735),\
                                          r'$\mathrm{SiC}(\omega_{SO})$':(mat.SiC_6H_Ellips,945),\
                                          r'$\mathrm{SiC}(\omega_{-})$':(mat.SiC_6H_Ellips,920),\
                                          r'$\mathrm{SiC}(\omega_{+})$':(mat.SiC_6H_Ellips,970)},\
                       Nterms=15):
    
    ordered=['Gold','Carbonyl',
             r'$\mathrm{SiC}(\omega_{-})$',
             r'$\mathrm{SiC}(\omega_{SO})$',\
             r'$\mathrm{SiC}(\omega_{+})$']
    colors=['c','g',(1,0,0),(.7,0,.7),(0,0,1)]
    
    global beta,LRM_signals,EFM_signals,material_name
    zs=numpy.logspace(numpy.log(.1/30.)/numpy.log(10.),numpy.log(zmax)/numpy.log(10.),100)
    zs2=numpy.logspace(numpy.log(.1/30.)/numpy.log(10.),numpy.log(15)/numpy.log(10.),100)
    
    LRM_signals={}
    EFM_signals={}
    for i,(material_name,pair) in enumerate(materials.items()):
        
        material,freq=pair
        beta=material.reflection_p(freq,q=1/(30e-7))
        LRM_signals[material_name]=tip.LRM(freq,rp=beta,a=30,zs=zs*30,\
                                           Nqs=244,demodulate=False,normalize_to=None)['signals']
        LRM_signals[material_name].set_axes([zs])
        EFM_signals[material_name]=EigenfieldModel(beta,zs=zs2,Nterms=Nterms)
        
        if i==0: tip.LRM.load_params['reload_model']=False
        
    tip.LRM.load_params['reload_model']=True
        
    figure()
    ref_signal=LRM_signals['Gold'].cslice[zmax]
    
    subplot(121)
    for material_name,color in zip(ordered,colors):
        numpy.abs(LRM_signals[material_name]/ref_signal).plot(label=material_name,plotter=loglog,color=color)
        numpy.abs(EFM_signals[material_name]/ref_signal).plot(color=color,marker='o',ls='')
        
    ylabel('$|E_\mathrm{rad}|\,[\mathrm{a.u.}]$')
    xlabel('$d/a$')
    ylim(3e-1,3e1)
    grid()
    
    leg=legend(loc='lower left',fancybox=True,shadow=True)
    leg.get_frame().set_linewidth(.1)
    for t in leg.texts: t.set_fontsize(18)
        
    subplot(122)
    for material_name,color in zip(ordered,colors):
        p_LRM=AWA(numpy.unwrap(numpy.angle(LRM_signals[material_name]/ref_signal)))
        p_LRM.adopt_axes(LRM_signals[material_name])
        p_EFM=AWA(numpy.unwrap(numpy.angle(EFM_signals[material_name]/ref_signal)))
        p_EFM.adopt_axes(EFM_signals[material_name])
        
        (p_LRM/(2*numpy.pi)).plot(label=material_name,plotter=semilogx,color=color)
        (p_EFM/(2*numpy.pi)).plot(color=color,marker='o',ls='')
        
    ylabel(r'$\mathrm{arg}(E_\mathrm{rad})\,/\,2\pi$')
    xlabel('$d/a$')
    ylim(-.05,.5)
    grid()
    
    gcf().set_size_inches([12.5,7],forward=True)
    tight_layout()
    subplots_adjust(wspace=.3)

def get_tip_expansion_coeffs(z=.1,freq=1000,a=30,\
                             N=10,inv_beta=False):
    
    signal=tip.LRM(freq,rp=mat.Au.reflection_p,a=a,zmin=z,amplitude=0,\
                   normalize_to=None,normalize_at=1000,Nzs=1,demodulate=False)
    global M
    
    L=tip.LRM.LambdaMatx
    g=-numpy.matrix(numpy.diag(tip.LRM.qxs*numpy.exp(-2*tip.LRM.qxs*z/numpy.float(a))*tip.LRM.wqxs))
    M=L*g
    if inv_beta: M=numrec.InvertIntegralOperator(M)
    
    eg_vec=numpy.matrix(tip.LRM.get_dipole_moments(tip.LRM.qxs))*g
    Lvec=tip.LRM.Lambda0Vecx
    if inv_beta: Lvec=-M*Lvec

    #Verified: gives correct signal, both relative and absolute
    return numpy.array([numpy.array(eg_vec*M**n*Lvec) for n in range(N)]).squeeze()
