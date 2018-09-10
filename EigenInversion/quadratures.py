import numpy
from scipy.fftpack import ifft
from mpmath import fp
from mpmath.calculus.quadrature import TanhSinh, GaussLegendre, QuadratureRule

class ClenshawCurtis(QuadratureRule):
    """This quadrature rule only implements `calc_nodes`, and as
    such is duck-typing as a more thoroughly implemented
    `mpmath.QuadratureRule` subclass.  Arbitrary precision is not
    implemented, nodes default to double precision."""
    
    buffer=1e-2
    
    @staticmethod
    def clencurt(N):
        """ Computes the Clenshaw Curtis nodes and weights """
        
        if N == 1:
            x = 0
            w = 2
        else:
            n = N - 1
            C = numpy.zeros((N,2))
            k = 2*(1+numpy.arange(numpy.floor(n/2)))
            C[::2,0] = 2/numpy.hstack((1, 1-k*k))
            C[1,1] = -n
            V = numpy.vstack((C,numpy.flipud(C[1:n,:])))
            F = numpy.real(ifft(V, n=None, axis=0))
            x = F[0:N,1]
            w = numpy.hstack((F[0,0],2*F[1:n,0],F[n,0]))
      
        return x,w
    
    def calc_nodes(self,degree,prec):
        
        x,w=self.clencurt(N=degree)
        smallest_diff=1-x[-2]
        x*=1-self.buffer*smallest_diff
        
        return list(zip(x,w))

CC=ClenshawCurtis(fp)
TS=TanhSinh(fp)
GL=GaussLegendre(fp)
prec=4
#GL quadrature seems to work A LOT better, 
#it's as though the middle values of the kernel
#are important, and TS under-samples them...

def GetQuadrature(N=72,xmin=1e-3,xmax=numpy.inf,\
                  quadrature=GL,**kwargs):
    
    global xs,weights
    
    if hasattr(quadrature,'calc_nodes'):
        
        #deg=int(numpy.floor(numpy.log(N)/numpy.log(2)))
        
        if quadrature is GL: deg=numpy.log(2/3.*N)/numpy.log(2)
        elif quadrature is TS: deg=numpy.log(N)/numpy.log(2)-1
        elif quadrature is CC: deg=N
        else: deg=N
        
        deg=int(numpy.ceil(deg))
        
        #The above formulas are just heuristics for the degree necessary for at least N samples
        #If it's not the case, might have to run once more...
        nodes=[]
        while len(nodes)<N:
            nodes=quadrature.calc_nodes(deg,prec); deg+=1
        
        nodes=quadrature.transform_nodes(nodes,a=xmin,b=xmax)
        
        xs,weights=list(zip(*nodes))
        xs=numpy.array(xs)
        weights=numpy.array(weights)
        #@bug: mpmath 0.17 has a bug whereby TS weights are overly large...
        if quadrature is TS: weights*=3.8/numpy.float(len(weights))
        
    # ---Just bail out and due linear quadrature otherwise (Riemann sum)--- #
    else:
        span=xmax-xmin
        xs=numpy.linspace(xmin+span/float(2*N),\
                          xmax-span/float(2*N),N)
        weights=numpy.array([span/float(N)]*int(N))
    
    return xs,weights