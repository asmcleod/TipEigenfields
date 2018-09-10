import os
import cmath
import numpy
from matplotlib import pyplot
from . import eigeninversion as EI

basedir=os.path.dirname(__file__)
demo_dir=os.path.join(basedir,'DemoData')

def PredictDemodulatedSiO2Spectrum(geometry='Cone',L=19,taper=20,\
                                   lift=0,amplitude=2,\
                                   Nj=8,Nts=6):
    
    eps_path=os.path.join(demo_dir,'SiO2_epsilon.txt')
    arr=numpy.loadtxt(open(eps_path))
    
    # ---Unpack epsilon and get surface reflectance `beta`--- #
    freqs,eps1,eps2=arr.T
    epsilon=eps1+1j*eps2
    beta_sio2=(epsilon-1)/(epsilon+1)
    
    # ---Build a tip model--- #
    Model=EI.EigenfieldModel(geometry=geometry,L=L,taper=taper)
    
    # ---Define the reference `beta` value known a priori--- #
    epsilon_Si=11.7
    beta_Si=(epsilon_Si-1)/(epsilon_Si+1)
    
    # ---Predict spectrum at each demodulation order--- #
    normalized_spectra={}
    for harmonic in [2,3]:
        
        # ---Nodes define the appropriate time-integral of approach curve--- #
        nodes=Model.get_demodulation_nodes(lift=lift,amplitude=amplitude,quadrature=EI.GL,\
                                           harmonic=harmonic,Nts=Nts)
        
        # ---Get reference signal (e.g., silicon)--- #
        Si_signal=Model.get_signal_from_nodes(beta_Si,nodes=nodes,Nj=Nj)
        
        # ---Get SiO2 spectrum--- #
        sio2_spectrum=Model.get_signal_from_nodes(beta_sio2,nodes=nodes,Nj=Nj)
        
        # ---Keep normalized spectrum--- #
        normalized_spectra[harmonic]=sio2_spectrum/Si_signal
        
    # ---Plot the result--- #
    pyplot.figure()
    
    pyplot.subplot(131)
    pyplot.plot(freqs,epsilon.real,\
                lw=2,color='b',\
                linestyle='-')
    pyplot.plot(freqs,epsilon.imag,\
                lw=2,color='b',\
                linestyle='--')
    
    pyplot.gca().lines[0].set_label(r'$\mathrm{Re}(\epsilon_1)$')
    pyplot.gca().lines[1].set_label(r'$\mathrm{Im}(\epsilon_1)$')
    
    pyplot.ylim(-6,13)
    pyplot.xticks((900,1000,1100,1200))
    leg=pyplot.legend(loc='best',shadow=True,fancybox=True,frameon=False,handletextpad=.1)
    pyplot.xlabel('$\omega\,[cm^{-1}]$',fontsize=25)
    pyplot.ylabel('$\epsilon$',fontsize=30)
    
    pyplot.subplot(132)
    for color,harmonic in zip(('b','r'),(2,3)):
        pyplot.plot(freqs,numpy.abs(normalized_spectra[harmonic]),\
                    lw=2,color=color,\
                    label=r'$S_%i$'%harmonic)
    
    pyplot.ylim(-.5,5)
    pyplot.xticks((900,1000,1100,1200))
    leg=pyplot.legend(loc='upper left',shadow=True,fancybox=True,frameon=False)
    for t in leg.texts: t.set_fontsize(20)
    pyplot.xlabel('$\omega\,[cm^{-1}]$',fontsize=25)
    pyplot.ylabel('$S_n\, (\mathrm{Si\,ref.})$',fontsize=25)
    
    pyplot.subplot(133)
    for color,harmonic in zip(('b','r'),(2,3)):
        phase=numpy.unwrap(numpy.angle(normalized_spectra[harmonic]))
        pyplot.plot(freqs,phase,\
                    lw=2,color=color,\
                    label=r'$\phi_%i$'%harmonic)
    
    pyplot.ylim(-.5,4.5)
    pyplot.yticks((0,numpy.pi/4,numpy.pi/2.,3*numpy.pi/4,numpy.pi,5*numpy.pi/4),\
                  (r'$0$',r'$\pi/4$',r'$\pi/2$',r'$3 \pi/4$',r'$\pi$',r'$\pi/4$'),\
                  fontsize=18)
    pyplot.xticks((900,1000,1100,1200))
    leg=pyplot.legend(loc='upper left',shadow=True,fancybox=True,frameon=False)
    for t in leg.texts: t.set_fontsize(20)
    pyplot.xlabel('$\omega\,[cm^{-1}]$',fontsize=25)
    pyplot.ylabel('$\phi_n\, (\mathrm{Si\,ref.})$',fontsize=25)
    
    pyplot.gcf().set_size_inches((14,5),forward=True)
    pyplot.tight_layout()
    pyplot.subplots_adjust(wspace=.4)
    
    return normalized_spectra


def InvertSiO2Spectrum(geometry='Cone',L=19,taper=20,\
                       amplitude=2,Nj=8,Nts=6): #Lift height was zero when demo data was acquired
    
    s2_path=os.path.join(demo_dir,'SiO2_S2_spectrum.txt')
    s2_arr=numpy.loadtxt(open(s2_path))
    s3_path=os.path.join(demo_dir,'SiO2_S3_spectrum.txt')
    s3_arr=numpy.loadtxt(open(s3_path))
    
    # ---Unpack signal data and get spectra at 2nd and 3rd harmonics--- #
    signals={}
    freqs,s2_real,s2_imag=s2_arr.T
    signals[2]=s2_real+1j*s2_imag
    freqs,s3_real,s3_imag=s3_arr.T
    signals[3]=s3_real+1j*s3_imag
    
    # ---Build a tip model--- #
    Model=EI.EigenfieldModel(geometry=geometry,L=L,taper=taper)
    
    # ---Define the reference `beta` value known a priori--- #
    epsilon_Si=11.7
    beta_Si=(epsilon_Si-1)/(epsilon_Si+1)
    
    # ---Proceed to invert the spectrum at each harmonic--- #
    epsilons={}
    absorptions={}
    for harmonic in [2,3]:
        
        # ---Nodes define the appropriate time-integral of approach curve--- #
        nodes=Model.get_demodulation_nodes(lift=0,amplitude=amplitude,quadrature=EI.GL,\
                                           harmonic=harmonic,Nts=Nts)
    
        # ---Build the associated inverse functions-- #
        Inv=Model.get_inverse(ref_beta=beta_Si,nodes=nodes,output=EI.Epsilon,Nj=Nj)

        # ---Invert spectrum to get dielectric constant and absorption coefficient of SiO2 --- #
        spectrum=signals[harmonic]
        epsilon=Inv.invert_spectrum(spectrum, select_by='continuity')
        
        epsilons[harmonic]=epsilon
        absorptions[harmonic]=numpy.sqrt(epsilon).imag #This is the formula for optical absorption
    
    pyplot.figure()
    
    # ---Plot the result, the loaded spectra first--- #
    pyplot.subplot(131)
    for color,harmonic in zip(('b','r'),(2,3)):
        pyplot.plot(freqs,numpy.abs(signals[harmonic]),\
                    lw=2,color=color,\
                    label=r'$S_%i$'%harmonic)
    
    pyplot.yticks((0,1,2,3,4))
    pyplot.xticks((900,1000,1100,1200))
    leg=pyplot.legend(loc='best',shadow=True,fancybox=True,frameon=False)
    for t in leg.texts: t.set_fontsize(20)
    pyplot.xlabel('$\omega\,[cm^{-1}]$',fontsize=25)
    pyplot.ylabel('$S_n\, (\mathrm{Si\,ref.})$',fontsize=25)
    
    # ---Plot the extracted dielectric functions--- #
    pyplot.subplot(132)
    for color,harmonic in zip(('r','b'),(2,3)):
        pyplot.plot(freqs,epsilons[harmonic].real,\
                    lw=2,color=color,\
                    linestyle='-')
        pyplot.plot(freqs,epsilons[harmonic].imag,\
                    lw=2,color=color,\
                    linestyle='--')
        
        if harmonic is 2:
            pyplot.gca().lines[0].set_label(r'$\mathrm{Re}(\epsilon)$')
            pyplot.gca().lines[1].set_label(r'$\mathrm{Im}(\epsilon)$')
    
    pyplot.ylim(-8,13)
    pyplot.xticks((900,1000,1100,1200))
    leg=pyplot.legend(loc='best',shadow=True,fancybox=True,frameon=False)
    for t in leg.texts: t.set_fontsize(20)
    pyplot.xlabel('$\omega\,[cm^{-1}]$',fontsize=25)
    pyplot.ylabel('$\epsilon$',fontsize=30)
    
    # ---Plot the extracted absorption coefficients--- #
    pyplot.subplot(133)
    for color,harmonic in zip(('r','b'),(2,3)):
        pyplot.plot(freqs,numpy.abs(absorptions[harmonic]),\
                    lw=2,color=color)
    
    pyplot.ylim(-.25,2.75)
    pyplot.xticks((900,1000,1100,1200))
    pyplot.xlabel('$\omega\,[cm^{-1}]$',fontsize=25)
    pyplot.ylabel('Absorption Coefficient $\kappa$',fontsize=22)
    
    pyplot.gcf().set_size_inches((14,5),forward=True)
    pyplot.tight_layout()
    pyplot.subplots_adjust(wspace=.4)
    
    return {'signals':signals,\
            'epsilons':epsilons,\
            'absorptions':absorptions}
    