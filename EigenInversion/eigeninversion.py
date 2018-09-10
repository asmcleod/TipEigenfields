import os
import time
import cmath
import numpy
from numpy.linalg import linalg
from scipy.signal import invres
from scipy.interpolate import interp1d
from .quadratures import CC,TS,GL,GetQuadrature

basedir=os.path.abspath(os.path.dirname(__file__))
datadir=os.path.join(basedir,'PolesResiduesData')
verbose=True

###################################################
#------Utilities for polynomial root-finding------#
###################################################

def companion_matrix(p):
    """Assemble the companion matrix associated with a polynomial
    whose coefficients are given by `poly`, in order of decreasing
    degree.
    
    References:
    1) http://en.wikipedia.org/wiki/Companion_matrix
    """
    
    A = numpy.diag(numpy.ones((len(p)-2,), numpy.complex128), -1)
    A[0, :] = -p[1:] / p[0]

    return numpy.matrix(A)

scaling=10

def find_roots(p):
    """Find roots of a polynomial with coefficients `p` by
    computing eigenvalues of the associated companion matrix.
    
    Overflow can be avoided during the eigenvalue calculation
    by preconditioning the companion matrix with a similarity
    transformation defined by scaling factor `scaling` > 1.
    This transformation will render the eigenvectors in a new
    basis, but the eigenvalues will remain unchanged.
    
    Empirically, `scaling=10` appears to enable successful 
    root-finding on polynomials up to degree 192."""
    
    global scaling,Mstar,q
    q=p
    
    scaling=numpy.float64(scaling)
    M=companion_matrix(p); N=len(M)
    D=numpy.matrix(numpy.diag(scaling**numpy.arange(N)))
    Dinv=numpy.matrix(numpy.diag((1/scaling)**numpy.arange(N)))
    Mstar=Dinv*M*D # Similarity transform will leave the eigenvalues unchanged
    
    return linalg.eigvals(Mstar)


##########################################
#------Output options for inversion------#
##########################################
#                                        #
# Pass any one to the constructor of     #
# `InverseFunction` as `output`          #
# and the inversion will output that     #
# desired quantity.                      #
#                                        #
##########################################
    
def passthrough(beta): return beta

Beta=passthrough #An alias

def ImBeta(beta): return beta.imag

def Epsilon(beta): return (1+beta)/(1-beta)

def Absorption(beta):
    
    epsilon=Epsilon(beta)
    
    return cmath.sqrt(epsilon).imag

class EigenfieldModel(object):
    
    def __init__(self,geometry='Cone',L=19,taper=20,interpolation='linear'):
        
        # ---Probe geometry determines which poles / residues should be loaded--- #
        poles_filename='Poles_%s_L=%imicrons_Taper=%ideg.txt'%(geometry.title(),L,taper)
        residues_filename='Residues_%s_L=%imicrons_Taper=%ideg.txt'%(geometry.title(),L,taper)
        
        poles_path=os.path.join(datadir,poles_filename)
        residues_path=os.path.join(datadir,residues_filename)
        
        # ---Load poles / residues data and format contents--- #
        Pdata=numpy.loadtxt(open(poles_path))
        Rdata=numpy.loadtxt(open(residues_path))
        
        ds=Pdata[:,0] # `ds` are log-spaced from zero to enable efficient sampling on tip-sample approach (units of tip radius "a")
        Parr=Pdata[:,1:]
        Rarr=Rdata[:,1:]
        
        Nj=Parr.shape[1]/2
        Ps=[Parr[:,2*i]+1j*Parr[:,2*i+1] for i in range(Nj)]
        Rs=[Rarr[:,2*i]+1j*Rarr[:,2*i+1] for i in range(Nj)]
        
        self.Ps=numpy.array(Ps).T #Each column of `self.Ps` will be a different pole, versus tip-sample distance
        self.Rs=numpy.array(Rs).T
        self.ds=ds
        
        # ---Store pole and residue interpolators--- #
        # This might be implemented with `scipy.interpolate.UnivariateSpline` instead, perhaps more accurate #
        self.PsInterp=interp1d(ds,self.Ps,axis=0,kind=interpolation,bounds_error=True)
        self.RsInterp=interp1d(ds,self.Rs,axis=0,kind=interpolation,bounds_error=True)
        self.Nj=Nj
        
    def __call__(self,*args,**kwargs): return self.get_Erad(*args,**kwargs)
        
    def evaluate_poles(self,ds=None,Nj=None):
        """Evaluate poles j=1 to `N_j` at tip-sample separations
        `ds` (in units of tip radius "a")."""
        
        # ---Default to all available terms--- #
        if not Nj: Nj=self.Nj
        if ds is None: ds=self.ds
        
        # ---Interpolate to arbitrary `ds` using linear interpolation--- #
        Ps=self.PsInterp(ds)[:,:Nj]
        
        return Ps
        
    def evaluate_residues(self,ds=None,Nj=None):
        """Evaluate residues j=1 to `N_j` at tip-sample separations
        `ds` (in units of tip radius "a")."""
        
        # ---Default to all available terms--- #
        if not Nj: Nj=self.Nj
        if ds is None: ds=self.ds
        
        # ---Interpolate to arbitrary `ds` using linear interpolation--- #
        Rs=self.RsInterp(ds)[:,:Nj]
        
        return Rs
    
    def get_Erad(self,beta,ds=None,Nj=None):
        """Evaluate tip-radiated field (up to a constant) for a given surface
        reflection coefficient `beta` (can be either a single [complex]
        number or an array, e.g., over frequency) and over tip-sample
        separations `ds` (in units of tip radius "a"), incorporating
        `N_j` eigenfield terms."""
        
        # ---In case `beta` is an array, ensure it has trailing dimensions for `ds` and j-terms--- #
        if hasattr(beta,'__len__'):
            if isinstance(beta,numpy.ndarray):
                if not beta.ndim: beta=beta.tolist()
            else: beta=numpy.array(beta)
        
        if isinstance(beta,numpy.ndarray): beta=beta.reshape((len(beta),1,1))
        
        # ---Acquire the necessary pole and residue values--- #
        if ds is None: ds=self.ds
        Ps=self.evaluate_poles(ds,Nj)
        Rs=self.evaluate_residues(ds,Nj)
        
        # ---Should broadcast if `beta` has an additional first axis (e.g. frequency)--- #
        # ---The offset term is absolutely critical. It offsets false z-dependence arising from first terms--- #
        Erad=numpy.sum(Rs*(1/(beta-Ps)+1/Ps),axis=-1)
        
        Erad=numpy.array(Erad).squeeze()
        if not Erad.ndim: Erad=Erad.tolist()
        
        return Erad
    
    def get_demodulation_nodes(self,lift=0,amplitude=2,quadrature=GL,\
                               harmonic=3,Nts=None):
        """Return quadrature nodes appropriate for modeling a signal
        demodulated at probe tapping `harmonic` with tapping `amplitude` 
        and measurement lift height `lift` (both in units of tip radius "a"). 
        `Nts` is the number of nodes, defaulting to twice `harmonic`.
        
        Available values for `quadrature` are:
        
            `GL`  --  Gauss-Legendre rule
            `TS`  --  Tanh-Sinh rule
            `CC`  --  Clenshaw-Curtis rule
            
            ... or any other quadrature rule class defined according to [1].
            
        [1] http://mpmath.googlecode.com/svn/trunk/doc/build/calculus/integration.html"""
        
        # ---Use minimum available lift height - even "in contact" is
        # a vanishing but non-zero minimum distance.
        if not lift>=self.ds.min(): lift=self.ds.min()
        
        # ---Get temporal points and integration weights from quadrature rule--- #
        if not Nts: Nts=2*harmonic # max resolvable harmonic will by 1/dt=Nts
        if hasattr(quadrature,'calc_nodes'):
            ts,wts=GetQuadrature(N=Nts,xmin=-.5,xmax=0,quadrature=quadrature)
            
        else:
            ts=numpy.linspace(-1+1/numpy.float(Nts),\
                              0-1/numpy.float(Nts),Nts)*.5
            wts=numpy.ones((Nts,))/numpy.float(Nts)*.5
            
        # ---cosine harmonic kernel, *2 for full period integration, *2 for coefficient--- #
        ws=wts*4*numpy.cos(2*numpy.pi*harmonic*ts)
        ds=lift+amplitude*(1+numpy.cos(2*numpy.pi*ts)) 
        
        return list(zip(ds,ws))
    
    def get_signal_from_nodes(self,beta,nodes=[(0,1)],Nj=None):
        
        if not hasattr(beta,'__len__'): beta=[beta]
        
        # ---"Frequency" axis will be first---
        if isinstance(beta,numpy.ndarray): beta=beta.reshape((len(beta),1,1))
        
        # ---Weights apply across z-values---
        ds,ws=list(zip(*nodes))
        ws=numpy.array(ws)[numpy.argsort(ds)]
        ds=numpy.array(ds)[numpy.argsort(ds)]
        
        ws_grid=numpy.array(ws).reshape((1,len(ws),1))
        
        # ---Evaluate at all nodal points---
        Rs=self.evaluate_residues(ds,Nj)
        Ps=self.evaluate_poles(ds,Nj)
        
        # ---Evaluate integral transform by quadrature sum---
        #Poles / residues hould broadcast over "frequencies" as beta has an additional first axis
        #The offset term is absolutely critical, offsets false z-dependence arising from first terms
        signals=numpy.sum(numpy.sum(Rs*(1/(beta-Ps)+1/Ps)*ws_grid,axis=-1),axis=-1)
        
        signals=numpy.array(signals).squeeze()
        if not signals.ndim: signals=signals.tolist()
        
        return signals
    
    def get_inverse(self,ref_beta=1,nodes=[(1,0)],Nj=None,output=passthrough):
        
        return InverseFunction(self,ref_beta,nodes,Nj,output=output)
    
    
###########################################
#------Inverse function of tip model------#
###########################################
    
class InverseFunction(object):
    
    def __init__(self,Model,ref_beta=1,
                 nodes=[(1,0)],Nj=None,\
                 output=passthrough):
    
        if not Nj: Nj=Model.Nj
        self.Nj=Nj
        self.nodes=nodes
        self.Model=Model
        self.set_ref_signal(ref_beta)
        
        ds,ws=list(zip(*nodes))
        ws_grid=numpy.array(ws).reshape((len(ws),1)) #last dimension is to broadcast over all `Nterms` equally
        ds=numpy.array(ds)
        
        # ---Acquire the necessary pole and residue values--- #
        Ps=Model.evaluate_poles(ds,Nj)
        Rs=Model.evaluate_residues(ds,Nj)
        
        #`rs` and `ps` can safely remain as arrays for `invres`
        ps=Ps.flatten()
        rs=(Rs*ws_grid).flatten()
        k0=numpy.sum(rs/ps).tolist()
        
        # ---Rescale units to center dynamic range of `rs` and `ps` around 1e0--- #
        rscaling=numpy.exp(-(numpy.log(numpy.abs(rs).max())+\
                            numpy.log(numpy.abs(rs).min()))/2.)
        pscaling=numpy.exp(-(numpy.log(numpy.abs(ps).max())+\
                             numpy.log(numpy.abs(ps).min()))/2.)
        
        self.rscaling=rscaling
        self.pscaling=pscaling
        
        ps*=pscaling
        rs*=rscaling
        k0*=rscaling/pscaling
        
        self.ps=ps
        self.rs=rs
        self.k0=k0
        
        # ---Inverse partial fraction expansion of Eigenfield Laurent series--- #
        #highest order first in `Ps` and `Qs`
        #VERY SLOW - about 100ms on practical inversions (~60 terms)
        #therefore, do it only once and store the polynomial coefficients `As` and `Bs`
        As,Bs=invres(rs, ps, k=[k0], tol=1e-16, rtype='avg') #tol=1e-16 is the smallest allowable to `unique_roots`..
        
        #Double precision offers noticeable protection against overflow
        dtype=numpy.complex128
        self.As=numpy.array(As,dtype=dtype)
        self.Bs=numpy.array(Bs,dtype=dtype)
        self.output=output
        
        # ---Initialize a counter so inverse function can have self-awareness of
        # beta continuity etc. for any series of inversions--- #
        self.i=0
        self.betas=[]
        # This might also be expanded into something like a signal <--> beta 
        # association for shortcut guessing at the inversion outcome
        
    def set_ref_signal(self,ref_beta):
        
        self.ref_signal=self.Model.get_signal_from_nodes(ref_beta,self.nodes,self.Nj)
        
    def __call__(self,norm_signal,select_by='continuity',closest_pole=0):
        
        signal=norm_signal*self.ref_signal
        signal*=self.rscaling/self.pscaling
        
        #Root finding `roots` seems to give noisy results when `Bs` has degree >84, with dynamic range ~1e+/-30 in coefficients...
        #Pretty fast, 1-2 ms on practical inversions with rank ~60 companion matrices, <1 ms with ~36 terms
        #@TODO: Root finding chokes on `Nj=8` (number of eigenfields) and `Nts=12` (number of nodes),
        #       necessary for truly converged S3 on resonant phonons.  Probably due to
        #       floating point overflow - leading terms of `As` and `Bs` increase exponentially with
        #       number of terms, leading to huge dynamic range.
        #       Perhaps limited by the double precision of DGEEV inside `numpy.roots`.
        #       So, replace with faster / more reliable root finder?
        #       We need 1) speed, 2) ALL roots (or at least the first ~10 smallest)
        
        roots=find_roots(self.As-signal*self.Bs)
        roots=roots[roots.imag>=0] #physical values of beta have `Im(beta)>0`
        roots/=self.pscaling #since all beta units scaled by `pscaling`, undo that here
        
        #How should we select the most likely beta among the multiple solutions?
        #There may be a better way to establish a filtering function that needs definition only once...
        if select_by=='minimum':
            to_minimize=numpy.abs(roots)
        
        #1. Avoids large changes in value of beta
        elif select_by=='difference' and self.i>=1:
            to_minimize=numpy.abs(roots-self.betas[self.i-1])
            
        #2. Avoids large changes in slope of beta (best for spectroscopy)
        #Nearly guarantees good beta spectrum, with exception of very loosely sampled SiC spectrum
        #Loosely samples SiO2-magnitude phonons still perfectly fine
        elif select_by=='continuity' and self.i>=2:
            earlier_diff=self.betas[self.i-1]-self.betas[self.i-2]
            current_diffs=roots-self.betas[self.i-1]
            to_minimize=numpy.abs(current_diffs-earlier_diff)
            
        #3. Select specifically which pole we want |beta| to be closest to
        else:
            reordering=numpy.argsort(numpy.abs(roots)) #Order the roots towards increasing beta
            roots=roots[reordering]
            to_minimize=numpy.abs(closest_pole-numpy.arange(len(roots)))
        
        beta=roots[to_minimize==to_minimize.min()].squeeze()
        self.betas.append(beta)
        self.i+=1
    
        output=self.output(beta)
        
        return output
    
    def invert_image(self,norm_image,*args,**kwargs):
        """Inverting an image consists of inverting each in a sequence
        of "uncoiled" signals from `norm_image` based on the criterion that
        the most likely "beta" value is that with the minimum magnitude among
        the eligible roots."""
        
        # ---Reset counter, since clearly this function call corresponds with a new experiment--- #
        self.i=0
        self.outputs=[]
        
        # ---Function to uncoil an 2-dimensional NxN array into a 1-dimension N*N array--- #
        uncoil_image = lambda image: \
                        numpy.hstack([row[::(-1)**i] for i,row in enumerate(image)])
        
        # ---Function to do the inverse--- #
        coil_image = lambda uncoiled,row_len: \
                        numpy.vstack([uncoiled[i*row_len:(i+1)*row_len][::(-1)**i] \
                                      for i in range(len(uncoiled)/row_len)])
        
        # ---Uncoil, invert each signal value in turn, and coil back up into corresponding image--- #
        t1=time.time()
        norm_signals=uncoil_image(norm_image)
        print(norm_signals.shape)
        
        outputs=[]; progress=True
        for i,norm_signal in enumerate(norm_signals):
            if progress and not i%len(norm_image):
                print('Progress:  %1.2f%%'%(i/numpy.float(len(norm_signals))*100))
            outputs.append(self(norm_signal,select_by=None,*args,**kwargs))
            
        if verbose: print('Average inversion time per pixel: %s s'%\
                           ((time.time()-t1)/float(len(norm_signals))))
                           
        outputs=numpy.array(outputs).squeeze()
        outputs_image=coil_image(outputs,row_len=norm_image.shape[1])
        
        return outputs_image
    
    def invert_spectrum(self,norm_signals,select_by='continuity',\
                        reverse=False,*args,**kwargs):
        """Inverting a spectrum consists of inverting each in a sequence
        of `norm_signals` based on the continuity criterion with the
        most previously inverted "beta" values."""
        
        # ---Reset counter, since clearly this function call corresponds with a new experiment--- #
        self.i=0
        self.outputs=[]
        
        # ---It can sometimes be desirable to do the inversion in reverse order--- #
        # This is true if the first signal values start far from minimum beta root,
        # whereas the last signal values are the ones closest to the minimum root.
        if reverse: norm_signals=norm_signals[::-1]
        
        # ---Invert each signal value in turn, using the continuity criterion with previous `beta`--- #
        t1=time.time()
        outputs=[self(norm_signal,select_by=select_by,*args,**kwargs) \
               for norm_signal in norm_signals]
        outputs=numpy.array(outputs).squeeze()
        
        if verbose: print('Average inversion time per pixel: %s s'%\
                           ((time.time()-t1)/float(len(norm_signals))))
        
        # ---Un-reverse the inverted values--- #
        if reverse: outputs=outputs[::-1]
        
        return outputs