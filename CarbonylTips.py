import os
import numpy
import pickle
from common.log import Logger
from scipy import integrate
from scipy import special
from NearFieldOptics import TipModels as tip
from NearFieldOptics import Materials as mat
from common.baseclasses import AWA
from matplotlib.pyplot import *

root_dir=os.path.dirname(__file__)

############
#---Carbonyl
############

Amps=[2*1.35*.55]
Freqs=[1731]
Widths=[20]
lps=[[a*f*w,f,w] for a,f,w in zip(Amps,Freqs,Widths)]
Carbonyl=mat.IsotropicMaterial(eps_lps=lps,eps_infinity=1.82)

def get_DCSD(signals,zmin=0,amplitude=60,lift=15):
    
    A=signals.interpolate_axis(zmin+2*amplitude+lift,axis=0,bounds_error=False,extrapolate=True)
    B=signals.interpolate_axis(zmin+amplitude+lift,axis=0,bounds_error=False,extrapolate=True)
    C=signals.interpolate_axis(zmin+lift,axis=0,bounds_error=False,extrapolate=True)
    D=signals.interpolate_axis(zmin+amplitude+lift,axis=0,bounds_error=False,extrapolate=True)
    
    return A+C-(B+D)

def GenerateCarbonylTips(Ls=numpy.linspace(30,20e3,100),a=30,\
                         wavelength=6e3,taper_angles=[10,20],\
                         geometries=['cone','hyperboloid']):
    
    skin_depth=.05
    
    if wavelength is None: freq=0
    else: freq=a/numpy.float(wavelength)
    Ls=Ls/numpy.float(a)
    
    for i,geometry in enumerate(geometries):
        for k,taper_angle in enumerate(taper_angles):
            for j,L in enumerate(Ls):
                
                Logger.write('Currently working on geometry "%s", L/a=%s...'%(geometry,L))
                tip.build_charge_distributions(Nqs=144,Nzs=144,L=L,taper_angle=taper_angle,quadrature='TS',\
                                               geometry=geometry,freq=freq,skin_depth=skin_depth)
                
                progress=((i*len(taper_angles)+k)*len(Ls)+j+1)/numpy.float(len(Ls)*len(geometries)*len(taper_angles))*100
                Logger.write('\tProgress: %1.1f%%'%progress)
            
    Logger.write('Done!')

##Wavelength of 6 microns as default, very close to carbonyl frequency##
def CompareCarbonylTips(Ls=numpy.linspace(30,20e3,100),a=30,\
                                  wavelength=6e3,amplitude=60,lift=30,\
                                  geometries=['cone','hyperboloid'],\
                                  taper_angles=[20],\
                                  load=True,save=True,demo_file='CarbonylTips.pickle',\
                                  enhancement_index=2):
    
    carbonyl_freqs=numpy.linspace(1730,1750,10)
    normalize_at_freq=numpy.mean(carbonyl_freqs)
    freqs=[a/numpy.float(wavelength)] #load frequency should be quite close to `sio2_freq`
    Ls=Ls/float(a)
    freq_labels=['ED']
    skin_depth=.05
    
    ##################################################
    ## Load or compute Carbonyl measurement metrics ##
    ##################################################
    global d
    d_keys=['max_s3','max_s2',\
            'max_rel_absorption','max_absorption','max_freq',\
            'charges0','charges_q0','Lambda0','enhancement',\
            'signals','norm_signals','DCSD_contact','DCSD_lift',\
            'psf_contact','psf_lift']
    
    demo_path=os.path.join(root_dir,demo_file)
    if load and os.path.isfile(demo_path):
        Logger.write('Loading lengths data...')
        d=pickle.load(open(demo_path))
        
    else:
        tip.LRM.load_params['reload_model']=True
        tip.LRM.geometric_params['skin_depth']=skin_depth
        
        d={}
        
        for i,geometry in enumerate(geometries):
          for n,taper_angle in enumerate(taper_angles):
              
            #If geometry is not conical, do not iterate to additional taper angles, these are all the same.
            if geometry not in ['PtSi','hyperboloid','cone'] and n>0: break
              
            tip.LRM.geometric_params['geometry']=geometry
            tip.LRM.geometric_params['taper_angle']=taper_angle
            signals_for_geometry={}
            
            for j,freq in enumerate(freqs):
                signals_for_freq=dict([(key,[]) for key in d_keys])
            
                for k,L in enumerate(Ls):
                    tip.LRM.geometric_params['L']=int(L)
                    Logger.write('Currently working on geometry "%s", freq=%s, L/a=%s...'%(geometry,freq,L))
                    
                    #Make sure to calculate on zs all the way up to zmax=2*amplitude+lift
                    carbonyl_vals=tip.LRM(carbonyl_freqs,rp=Carbonyl.reflection_p,Nqs=72,zmin=.1,a=a,amplitude=amplitude+lift/2.,\
                                          normalize_to=mat.Si.reflection_p,normalize_at=normalize_at_freq,load_freq=freq,Nzs=30)
                    
                    ##Get overall signal with approach curve##
                    global carbonyl_sigs
                    carbonyl_sigs=carbonyl_vals['signals']*numpy.exp(-1j*numpy.angle(carbonyl_vals['norm_signals'].cslice[0]))
                    carbonyl_rel_sigs=carbonyl_vals['signals']/carbonyl_vals['norm_signals'].cslice[0]
                    ind=numpy.argmax(carbonyl_rel_sigs.imag.cslice[0]) #Find maximum phase in contact
                    
                    signals_for_freq['max_s3'].append(carbonyl_vals['signal_3'][ind]) #Pick out the peak frequency
                    signals_for_freq['max_s2'].append(carbonyl_vals['signal_2'][ind]) #Pick out the peak frequency
                    signals_for_freq['max_rel_absorption'].append(carbonyl_rel_sigs[:,ind].imag.cslice[0]) #Relative to out-of-contact on reference
                    signals_for_freq['max_absorption'].append(carbonyl_sigs[:,ind].imag.cslice[0]) #Absolute signal
                    signals_for_freq['max_freq'].append(carbonyl_freqs[ind])
                    signals_for_freq['charges0'].append(tip.LRM.charges0/(2*numpy.pi*tip.LRM.charge_radii))
                    signals_for_freq['charges_q0'].append(tip.LRM.charges.cslice[0]/(2*numpy.pi*tip.LRM.charge_radii))
                    signals_for_freq['Lambda0'].append(tip.LRM.Lambda0)
                    signals_for_freq['enhancement'].append(numpy.abs(2*tip.LRM.charges0[enhancement_index]))
                    signals_for_freq['signals'].append(carbonyl_sigs[:,ind]) #Pick out the peak frequency
                    signals_for_freq['norm_signals'].append(carbonyl_vals['norm_signals'])
                    signals_for_freq['DCSD_contact'].append(get_DCSD(carbonyl_sigs[:,ind],zmin=0.1,amplitude=60,lift=0))
                    signals_for_freq['DCSD_lift'].append(get_DCSD(carbonyl_sigs[:,ind],zmin=0.1,amplitude=60,lift=lift))
                    
                    ##Isolate some point-spread functions##
                    rs=numpy.linspace(0,5,200).reshape((200,1))
                    integrand_contact=AWA(tip.LRM.qxs**2*\
                                          special.j0(tip.LRM.qxs*rs)*tip.LRM.get_dipole_moments(tip.LRM.qxs)*\
                                          tip.LRM.wqxs,axes=[rs.squeeze(),tip.LRM.qxs])
                    integrand_contact=integrand_contact.interpolate_axis(numpy.logspace(numpy.log(tip.LRM.qxs.min())/numpy.log(10),\
                                                                                        numpy.log(tip.LRM.qxs.max())/numpy.log(10),1000),\
                                                                   axis=1)
                    psf_contact=numpy.sum(integrand_contact,axis=1)
                    signals_for_freq['psf_contact'].append(AWA(psf_contact,axes=[rs.squeeze()],axis_names=['r/a']))
                    
                    integrand_lift=AWA(tip.LRM.qxs**2*numpy.exp(-tip.LRM.qxs*lift/numpy.float(a))*\
                                       special.j0(tip.LRM.qxs*rs)*tip.LRM.get_dipole_moments(tip.LRM.qxs)*\
                                       tip.LRM.wqxs,axes=[rs.squeeze(),tip.LRM.qxs])
                    integrand_lift=integrand_lift.interpolate_axis(numpy.logspace(numpy.log(tip.LRM.qxs.min())/numpy.log(10),\
                                                                                  numpy.log(tip.LRM.qxs.max())/numpy.log(10),1000),\
                                                                   axis=1)
                    psf_lift=numpy.sum(integrand_lift,axis=1)
                    signals_for_freq['psf_lift'].append(AWA(psf_lift,axes=[rs.squeeze()],axis_names=['r/a']))
                    
                    progress=(i*len(Ls)*len(freqs)+j*len(Ls)+k+1)/numpy.float(len(Ls)*len(freqs)*len(geometries))*100
                    Logger.write('\tProgress: %1.1f%%'%progress)
                    
                for key in list(signals_for_freq.keys()):
                    if numpy.array(signals_for_freq[key]).ndim==2: #Prepend probe length axis and expand these into AWA's
                        axes=[a*Ls/numpy.float(wavelength),signals_for_freq[key][0].axes[0]]
                        axis_names=['$L/\lambda_\mathrm{C=O}$',signals_for_freq[key][0].axis_names[0]]
                        signals_for_freq[key]=AWA(numpy.array(signals_for_freq[key],dtype=numpy.complex),\
                                                      axes=axes,axis_names=axis_names)
                    else:
                        signals_for_freq[key]=AWA(numpy.array(signals_for_freq[key],dtype=numpy.complex),\
                                                      axes=[a*Ls/numpy.float(wavelength)],\
                                                      axis_names=['$L/\lambda_\mathrm{C=O}$'])
                    
                signals_for_geometry[freq_labels[j]]=signals_for_freq
            
            geometry_name=geometry
            if geometry in ['PtSi','hyperboloid','cone']: geometry_name+=str(taper_angle)
            
            d[geometry_name]=signals_for_geometry
            
    if not load and save:
        Logger.write('Saving tip comparison data...')
        file=open(demo_path,'wb')
        pickle.dump(d,file)
        file.close()
        
    return d

def PlotCarbonylTipComparison(lengths=True,charges=True):
    
    d=CompareCarbonylTips(load=True,save=True)
    
    #First plot enhancement and carbonyl absorption signal
    if lengths:
        figure()
        numpy.abs(d['cone']['ED']['enhancement']).plot(color='r')
        ylabel('Field Enhancement',color='r')
        tick_params(color='r',axis='y')
        yticks((10,20,30,40,50),color='r')
        gca().spines['left'].set_color('r')
        gca().spines['right'].set_color('b')
        twinx();numpy.abs(d['cone']['ED']['max_absorption']/(4*numpy.pi)).plot(color='b')
        ylabel('Carbonyl Absorption Signal',rotation=270,color='b')
        tick_params(color='b',axis='y')
        yticks(color='b')
        ylim(0,500)
        
        xlim(0,3.33)
        sca(gcf().axes[0]);grid()
        tight_layout()
        subplots_adjust(right=.85)
    
    #Next, plot the comparative out-of-phase surface charge density
    if charges:
        from common import plotting
        figure()
        z_axis=numpy.linspace(0,1,500)
        charges=d['hyperboloid']['ED']['charges0'].interpolate_axis(z_axis,axis=1,extrapolate=True,bounds_error=False)
        charges.imag.plot(cmap=cm.get_cmap('BWR'),colorbar=False)
        ylabel('$z/L$')
        clim(-.3,.3)
        
        cbar=colorbar()
        cbar.set_ticks((-.3,0,.3))
        cbar.set_ticklabels(('$+Q$','0','$-Q$'))
        sca(gcf().axes[1])
        ylabel('Surface Charge [arb. units]',rotation=270,ha='right')

def PlotIdealProbePerformance(ideal_length=404,wavelength=6e3,lift=30,\
                              approach=False,compare_spectra=True):
    
    freq=30/wavelength
    tip.LRM.geometric_params['L']=ideal_length
    
    if approach:
        gold=tip.LRM(1180,rp=mat.Au.reflection_p,a=30,amplitude=60,\
                     normalize_to=mat.Si.reflection_p,Nqs=72,normalize_at=1738,Nzs=30,\
                     load_freq=freq)
        p=numpy.exp(-1j*numpy.angle(gold['norm_signals'].cslice[120]))
        
        figure()
        approach=gold['sample_signal_v_time']
        approach*=p
        peak=abs(approach).max()
        approach/=peak
        
        approach+=numpy.cos(2*numpy.pi*tip.LRM.ts)*10+50
        
        numpy.abs(approach).plot(label='Approach on $Au$')
        plot(-1-tip.LRM.ts,numpy.abs(approach),color='b')
        plot(-1+tip.LRM.ts,numpy.abs(approach),color='b')
        plot(-tip.LRM.ts,numpy.abs(approach),color='b')
        plot(1-tip.LRM.ts,numpy.abs(approach),color='b')
        plot(1+tip.LRM.ts,numpy.abs(approach),color='b')
        
        approximation=gold['sample_signal_0']/2.+gold['sample_signal_1']*numpy.cos(2*numpy.pi*tip.LRM.ts)
        approximation*=p
        approximation=AWA(approximation,axes=[tip.LRM.ts])
        approximation/=peak
        
        numpy.abs(approximation).plot(label='1st Harmonic')
        plot(-1-tip.LRM.ts,numpy.abs(approximation),color='g',ls='-')
        plot(-1+tip.LRM.ts,numpy.abs(approximation),color='g',ls='-')
        plot(-tip.LRM.ts,numpy.abs(approximation),color='g',ls='-')
        plot(1-tip.LRM.ts,numpy.abs(approximation),color='g',ls='-')
        plot(1+tip.LRM.ts,numpy.abs(approximation),color='g',ls='-')
        
        xlabel('t/T')
        ylabel('Near-field Signal [a.u.]')
        leg=legend(loc='upper right',fancybox=True,shadow=True)
        for t in leg.texts: t.set_fontsize(18)
        leg.get_frame().set_linewidth(.1)
        grid()
        tight_layout()
        ylim(.35,1.18)
    
    if compare_spectra:
        freqs=numpy.linspace(1100,1900,200)
        
        layer=mat.LayeredMedia((mat.PMMA,100e-7),exit=mat.Si)
        PMMA=tip.LRM(freqs,rp=layer.reflection_p,a=30,amplitude=60+lift/2.,\
                     normalize_to=mat.Si.reflection_p,Nqs=72,normalize_at=1000,Nzs=30,\
                     load_freq=freq)
        figure(); PMMA['signal_2'].imag.plot(color='b')
        ylabel(r'$\mathrm{Im}[\,S_2\,]\,[\mathrm{norm.}]$',color='b')
        tick_params(color='b',axis='y')
        yticks(color='b')
        xlabel(r'$\omega\,[cm^{-1}]$')
        
        gca().spines['left'].set_color('b')
        gca().spines['right'].set_color('r')
        
        p=numpy.angle(PMMA['norm_signals'].cslice[0])
        DCSD1=get_DCSD(PMMA['signals'])*numpy.exp(-1j*p)
        DCSD2=get_DCSD(PMMA['signals'],lift=lift)*numpy.exp(-1j*p)
        twinx();(DCSD1.imag/numpy.abs(DCSD2)).plot(color='r')
        ylabel(r'$\mathrm{DCSD}\,[\mathrm{norm.}]$',color='r',rotation=270)
        tick_params(color='r',axis='y')
        yticks(color='r')

        tight_layout()
        subplots_adjust(right=.85)
    
    
    