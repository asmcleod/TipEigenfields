#!/usr/bin/env python
try: from .EigenInversion import demo
except ImportError:
    print('\nError importing necessary packages!  Please ensure '+\
          'the package directory "EigenInversion" is located int he same '+\
          'directory as this demo script.  SciPy, NumPy, Matplotlib, '+\
          'and mpmath are all python packages required by the '+\
          'EigenInversion package.\n')
    raise

if __name__=='__main__':
    
    print()
    print("(1) Demonstrating EigenInversion for the prediction of demodulated "+\
            "near-field IR sSNOM spectra of silicon dioxide normalized to silicon....")
    demo.PredictDemodulatedSiO2Spectrum()
    print("Done.\n")
    
    print("(2) Demonstrating EigenInversion for the inversion of experimentally "+\
            "acquired near-field IR sSNOM spectra of silicon dioxide (nanoFTIR) to extract the "+\
            "dielectric constant and absorption coefficient of the bulk material...")
    demo.InvertSiO2Spectrum()
    print("Done.\n")
    
    input('Inspect the produced plots, save them if desired, and hit [Enter] to terminate. '+\
              'Plots may not display promptly unless the Matplotlib backend is set to Tkinter (on Mac).')
    print()