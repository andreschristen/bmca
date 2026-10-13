#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 22 15:19:47 2025

@author: jac

pybmca:

Python implementation of the Bayesian Microplastics Count Analyses

https://github.com/andreschristen/bmca 

Camparison with a negative binomial distribution.

"""
import sys
import pickle
from datetime import datetime

from numpy import ones, zeros, floor, array, var, linspace, floor, mean, column_stack
from numpy import arange, argmin, int_
from numpy import sum as np_sum
from scipy.stats import nbinom, gamma,  dirichlet, multinomial, poisson
from matplotlib.pylab import subplots, rcParams, close
rcParams['font.size'] = 14.0

from pytwalk import pytwalk
from pybmca import pybmca, AnaDesign, PlotDesignFigures

path = "/home/jac/sync/Gente/Marco/Microplasticos"


def Nb( la, phi, A):
    """The Negative Binomial distribution with our parametrization."""
    return nbinom( n=(A * la**2)/phi, p=la/(la + phi) )

class pybmca_nbinom(pybmca):
    
    def __init__(self, la0, nu0, r1, r2, c, A, a=1, b=1):
        super().__init__( la0=la0, nu0=nu0, r1=r1, r2=r2, c=c, A=A)
        
        self.a = a
        self.b = b
        
        self.par_names = [   r"$\Lambda$",    r"$\Phi$"]
        self.twalk = pytwalk( n=2, U=self.Energy, Supp=self.Supp)
    
    def SimNsNb( self, m, A, la_true=None, phi_true=None):
     """Simulate a sample of m quadrants assuming a 
        quadrant area of A.
        This is used to calculate the expected loss L1.
     """
     if la_true is None:
         la_true = self.la_prior.rvs()
     if phi_true is None:
         phi_true = gamma.rvs( a=self.a, scale=la_true/self.b)
     la = la_true*ones(m)
     phi = phi_true*ones(m)
     ### Simulate la and phi | la  *once*
     ### and then with those, simulte the m Ns
     return Nb( la, phi, A=A).rvs()

    def SimNs( self, T, a, la_true=None, phi_true=None):
        """Simulate the total sampled, i.e summing over all quadrants.
           This is only used to calculate the expected loss L2.
        """
        m = int(floor(a/self.A))
        if m == 0:
            return zeros(T)
        if la_true is None:
            la = self.la_prior.rvs(size=T)
        else:
            la = la_true*ones(T)
        if phi_true is None:
            phi = gamma.rvs( a=self.a, scale=la/self.b)
        else:
            phi = phi_true*ones(T)
        ### Simulate la and phi | la  *once* per iterattion t (in range(T), cols)
        ### and then with those, simulate the m Ns (rows)
        la = column_stack([la]*m)
        phi = column_stack([phi]*m)
        return np_sum(Nb( la, phi, A=self.A).rvs(), axis=1)
    
    def L1_star( self, a, N1=100, N2=2000):
        
        ms= (floor(a/self.A)).astype(int_) #number of quadarnts
        K = ms.shape[0]
        self.Vs = zeros((K,N1))
        for i,m in enumerate(ms):
            for t in range(N1):
               print("****** m=%d, %d of %d: Calculating L1_star, %d of %d" % (m, i+1, K, t+1, N1))
               ### Simulate data
               self.ns = self.SimNsNb( m=m, A=self.A)
               self.RunMCMC( N=N2, burn_in=0)
               self.Vs[i,t] = var(self.twalk.Output[100:,0])
        return (self.beta**2/self.alpha) * mean(self.Vs, axis=1)
             
    def SimNsPoisson( self, m, A, la_true=None):
     """Simulate a sample of m quadrants assuming a 
        quadrant area of A, using a the Poisson model.
     """
     if la_true is None:
         la_true = gamma.rvs( self.alpha, scale=1/self.beta)
     la = ones(m)*la_true
     return poisson.rvs( A*la )
    
    def SimData( self, m, A, la_true=None, phi_true=None, n_true=None, p_true=None):
        """Simulate m quadrants data assuming a disign using area A and proportion
           q of categorized microplastics.
           returns T x (k+1) matrix, the first column with the n's
             the rest of the columns with the s' (categories counts)
         
        """
        if n_true is None:
            ns = self.SimNsNb( m, A, la_true, phi_true)
        else:
            ns = n_true
        True_n_p = zeros(( m, self.k+1))
        True_n_p[ :, 0] = ns
        Data_s = zeros(( m, self.k), dtype=int)
        for i,n in enumerate(True_n_p[ :, 0]):
            n_bar_q = floor(n * self.q( A*m, n, vector=False))
            if p_true is None:
                p = dirichlet.rvs( self.ga )[0,:]
            else:
                p = p_true
            True_n_p[ i, 1:] = p 
            Data_s[ i, :] = multinomial.rvs( n_bar_q, p)                
        return { "la_true":la_true, "n_true":n_true, "p_true":p_true,
                "True_n_p":True_n_p, "Data_s":Data_s, "a":A*m, "alpha":self.alpha, "beta":self.beta, "ga":self.ga}
    
    def NsLogLikelihood( self, la, phi, a, ns):
        dist = Nb( la, phi, A=a)
        return np_sum( dist.logpmf(ns) )
    
    def LaPhiLogPrior( self, la, phi):
        return self.la_prior.logpdf(phi) + gamma.logpdf( la, a=self.alpha, scale=1/self.beta)    

    def Energy( self, x):
        la, phi = x
        return -1*(self.NsLogLikelihood( la, phi, a=self.A, ns=self.ns) + self.LaPhiLogPrior( la, phi))

    def Supp( self, x):
        return all(x > 0)
    
    def SimInit(self):
         la = self.la_prior.rvs()
         phi = gamma.rvs( a=self.a, scale=la/self.b)
         return array([ la, phi])
    
    def RunMCMC( self, N, burn_in=1000):
        self.twalk.Run( T=N, x0=self.SimInit(), xp0=self.SimInit())
        if burn_in > 0:
            self.twalk.PlotTs(burn_in=1000)

    def PlotMargLa(self, la_true=None, title=True, plot_prior=True, \
                   linestyle='-', marker='o', color='k', fill=False, ax=None, **kwargs):
        """Plot the posterior for Lambda."""
        if ax is None:
            _, ax = subplots()
        self.twalk.PlotMarg( par=0, burn_in=100, ax=ax,  
            linestyle=linestyle, color=color, fill=fill, **kwargs)
        xl = ax.get_xlim()
        x = linspace( xl[0], xl[1], num=100)
        if plot_prior:
            ax.plot( x, self.la_prior.pdf(x), '-', color="green")
        #ax.hlines( 0, xl[0], xl[1], color="grey")
        ax.set_xlabel(r"$\Lambda$ (MP m$^{-2}$)")
        ax.grid(which='both', color='grey', linestyle='dotted', linewidth=1)
        if not(la_true is None):
            # True value vertical marker
            ax.vlines( la_true, 0, 0.1*ax.get_ylim()[1], linestyle="-", color="black", lw=1)
        if title:
            ax.set_title(r"Marginal posterior for $\Lambda$")
        return ax

    def PlotMargPhi(self, phi_true=None, title=True, \
                   linestyle='-', marker='o', color='k', fill=False, ax=None, **kwargs):
        """Plot the posterior for Phi."""
        if ax is None:
            _, ax = subplots()
        self.twalk.PlotMarg( par=1, burn_in=100, ax=ax,  
            linestyle=linestyle, color=color, fill=fill, **kwargs)
        xl = ax.get_xlim()
        ax.set_xlabel(r"$\Phi$ (MP m$^{-2}$)")
        ax.grid(which='both', color='grey', linestyle='dotted', linewidth=1)
        if not(phi_true is None):
            # True value vertical marker
            ax.vlines( phi_true, 0, 0.1*ax.get_ylim()[1], linestyle="-", color="black", lw=1)
        if title:
            ax.set_title(r"Marginal posterior for $\Phi$")
        return ax

    def ExpVR( self,  a_max=None, N1=100, N2=2000, prn_ev=1):
        """Calculate the Expected Variance Reduction from 0.1 to a_max."""
        if a_max is None:
            a_max = 1/self.c 
        a = arange( 0, a_max+ 0.1*self.A, step=self.A) #sequance of area sampled per quadrants
        num = a.size
        EL2_star = zeros(num)
        mean_n = zeros(num)
        mean_q = zeros(num)
        print("Calculating ExpL2_star:")
        for i in range(num):
            if i % prn_ev == 0:
                print("%d of %d" % (i,num-1))
            EL2_star[i], mean_n[i], mean_q[i] = self.ExpL2_star( a[i] )
        L1_star = self.L1_star( a, N1=N1, N2=N2) ### This takes 
        loss = 0.5*L1_star + 0.5*EL2_star
        i_star = argmin(loss)
        return {'i_star':i_star, 'a':a, 'loss':loss, 'L1_star':L1_star, 'EL2_star':EL2_star, 'mean_q':mean_q}

def PlotExpVR( rt, loss_color='black', plot_L1=True, plot_EL2=True, plot_Eq=True, ax=None):
    if ax is None:
        fig, ax = subplots()
    i_star = rt['i_star'] 
    a = rt['a'] 
    loss = rt['loss'] 
    L1_star = rt['L1_star'] 
    EL2_star = rt['EL2_star'] 
    mean_q = rt['mean_q']                  
    ax.plot( a, loss, '-', color=loss_color, linewidth=1.5)
    ax.plot( a[i_star], loss[i_star], color="magenta", marker="*", markersize=10)
    ax.axvline( a[i_star], color='grey', linestyle='solid', alpha=0.5 )
    if plot_L1:
        ax.plot( a, L1_star, '-.', color='blue')
    if plot_EL2:
        ax.plot( a, EL2_star, '--', color='orange')
    if plot_Eq:
        ax.plot( a, mean_q, 'k--')
    ax.set_ylabel("")
    ax.set_xticks( a, labels=["%d" % (i,) for i in range(a.size)] )
    ax.set_yticks( arange( 0, 1.1, 0.1) )
    #ax.grid(which='both', color='grey', linestyle='dotted', linewidth=1)
    ax.set_xlabel(r"Quadrants $m$")
    
    ax2 = ax.twiny()
    ax2.plot( a, loss, '-', color=loss_color)
    ax2.set_xlabel(r"$a$ (m$^2$)")
    return ax
    


def CreateDesignNb( number, mode0, r1, r2, c, A, 
                   alpha=3, N1=100, N2=2000, Flog=None, path="."):
    alpha = 3
    mode0 = mode0
    be = (alpha-1)/mode0
    la0 = mode0 + 1/be
    mz = pybmca_nbinom(la0= la0, nu0= (la0**2)/alpha, r1 = r1, r2 = r2, c = c, A=A)
    if Flog is None:
        print("This will take approx. %.1f minutes." %(N1*(N2/1000) * 1/(mz.c * mz.A) / 60))    
    else:
        print("\nThis will take approx. %.1f minutes.\n" %(N1*(N2/1000) * 1/(mz.c * mz.A) / 60), flush=True, file=Flog)
    rt = mz.ExpVR(N1=N1, N2=N2) ### This takes approx. N1*(N2/1000) * 1/(mz.c * mz.A) seconds !!!
    rt['number'] = number
    rt['mode0'] = mode0
    rt['r1'] = r1
    rt['r2'] = r2
    rt['c'] = c
    rt['A'] = A
    rt['alpha'] = alpha
    rt['la0'] = la0
    rt['nu0'] = (la0**2)/alpha
    # Saving the object to a file
    with open('%s/Nb/NbDesign%d.pkl' % (path,number), 'wb') as file:
        pickle.dump(rt, file)
    return rt
    

def PlotDesignFiguresNb( rt, plot_L2=False, ana_ylim=None, inset='LaPrior', path="."):
    mz = pybmca_nbinom(la0= rt['la0'], nu0= rt['nu0'], 
                       r1 = rt['r1'], r2 = rt['r2'], c = rt['c'], A=rt['A'])
    number = rt['number']
    ax = PlotExpVR( rt, plot_Eq=False)
    i_star = rt['i_star']
    a_star = rt['a'][i_star]
    mean_q_star = rt['mean_q'][i_star]
    print("a_star = %f, mean_q_star = %f" % (a_star, mean_q_star))
    print("Expected number of microplastics to be categorize = %f" % (a_star * la0 * mz.q( a_star, a_star * la0, vector=False),))

    if inset == 'LaPrior':
        ax_inset = ax.inset_axes(bounds=[ 0.3, 0.4, 0.4, 0.5])
        mz.PlotLaPrior( color='black', ax=ax_inset)
        ax_inset.set_yticklabels([])
        ax_inset.set_xlim((0,4000))
        ax_inset.grid(visible=False)
        ax_inset.axhline( y=0, color='grey', linestyle='dotted')
    if inset == 'AnaDesign':
        ax_inset = ax.inset_axes(bounds=[ 0.3, 0.4, 0.4, 0.5])
        ax_ana = AnaDesign( inst=mz, a_star=a_star, plot_L2=plot_L2, ylim=ana_ylim, ax=ax_inset)
        tksx = array([ 100, 400, 700, 1100, 1500])
        ax_ana.set_xticks( tksx ) #, [r"%d" % (tk,) for tk in tksx])
    else:
        ax_ana = AnaDesign( inst=mz, a_star=a_star, plot_L2=plot_L2, ylim=ana_ylim)
        tksx = array([ 100, 250, 400, 600, 800, 1000, 1500])
        ax_ana.set_xticks( tksx ) #, [r"%d" % (tk,) for tk in tksx])
        ax_ana.get_figure().tight_layout() 
        ax_ana.get_figure().savefig("%s/Nb/AnaDesign%d.png" % (path,number))
        
    ax.get_figure().tight_layout()    
    ax.get_figure().savefig("%s/Nb/NbExpLoss%d.png" % (path,number))

    return mz, ax, ax_ana, ax_inset




    
if __name__ == "__main__":


    ### The area sampled per 1 m2 quadrant is:
    A = 0.25*0.25
    ### Cost of sampling 1 m2 quadrant: (25 + 128)/5 + 154 + 616 + 616 + 154 = 1570
    ### Then sampling A m2 costs 1570.
    ### Sampling 1 m2 costs 1570/A 
    CxArea = 1570/A
    ### addehttps://matplotlib.org/stable/gallery/subplots_axes_and_figures/zoom_inset_axes.html#sphx-glr-gallery-subplots-axes-and-figures-zoom-inset-axes-pyd cost of counting samples: 1.28 per particle
    r1 = 1.28/CxArea
    #r1 = 5e-5 # rounded toq( a=TTomasa_a2, n=TTomasa_n2, vector=False)
    
    ### categorization per particle: 70 local 1000 foreing    
    r2 = 70/CxArea
    #r2 = 3e-3
    
    ### Old stuff:
    ### Maximum number of particles to analyze: 250
    ### Cost: 250 * 70 
    ###         beach    Categorization
    ### budget: 1570*5 + 150*70         = 25350q( a=TTomasa_a2, n=TTomasa_n2, vector=False)
    #bt = 1570*5 + 200*70 #        
    ### c2: How many **m2 can we sample** if all budget is spent in the beach?
    ### Sampling 1 m2 costs CxArea
    #c = CxArea/bt #1.14, 1/(mz1.A * mz1.c) = 13.9
    
    c = 1/(10 * A) # Maximum number of quadrants
    
    ###       pristine   afected   strongly polluted
    ### la0 =     10< ,      100,  >1000 

    alpha = 3
    mode0 = 200
    be = (alpha-1)/mode0
    la0 = mode0 + 1/be
    mz_P= pybmca( la0=la0, nu0=(la0**2)/alpha, 
                 r1 = r1, r2 = r2, c = c, A=A)
    mz = pybmca_nbinom( la0=la0, nu0=(la0**2)/alpha, 
                 r1 = r1, r2 = r2, c = c, A=A)

    command_line = False
    design = False
    if command_line:
        design = True
    figures = True
    over_dispersion = True

    if design:    
        jobs = [[1,200,1,10],[2,800,1,10], ### Illustration moving a priori
                [3,200,2,10], ### Twice expensive FTIR
                [4,800,1000,10], ### Very expensive FTIR 
                [5,200,1,8], [6,800,1,8], ### Lower budget
                [7,800,1,16], ### Same as 8 setting but expecting a less contaminated beach
                [8,800,1,16]] ### Insisting in sampling 5 quadrants, even with a contaminated beach
        ### Run only one job, taken from the first option
        ### in the command line, useful for procesing in several processors
        if command_line:
            job = int(sys.argv[1])
            read_rt = False
            job_list = jobs[job:(job+1)]
        else:
            job_list = jobs[0:1]
            read_rt = False
        for number,mode0,r2_mul,m_max in job_list:
            print(number,mode0,r2_mul,m_max)
            if read_rt:
                print("Reading pkl of design %d: %s/Nb/NbDesign%d.pkl \n\n" % ( number,path,number))
                with open('%s/Nb/NbDesign%d.pkl' % (path,number), 'rb') as file:
                    rt = pickle.load(file)
            else:
                F = open("%s/Nb/NbRun%d.log" % ( path, number), "w")
                print("Analizing design %d, %s \n\n" % ( number, datetime.now()), file=F, flush=True)
                rt = CreateDesignNb( number=number, 
                       mode0=mode0, r1=r1, r2=r2*r2_mul, c=1/(m_max * A), A=A, alpha=3, 
                       N1=300, N2=2000, Flog=F, path=path)
                print("Done, %s \n\n" %datetime.now(), file=F)    
                F.close()
            mz_Nb, ax, ax_ana, ax_inset = PlotDesignFiguresNb( rt=rt, path=path )    
            mz_P= pybmca( la0=rt['la0'], nu0=rt['nu0'], r1 =rt['r1'], r2 =rt['r2'], c =rt['c'], A =rt['A'])
            mz_P.PlotExpVR(loss_color='grey', plot_L1=False, plot_EL2=False, plot_Eq=False, prn_ev=10, ax=ax)
            ax.get_figure().savefig("%s/Nb/NbExpLoss%d.png" % ( path, number))
        if command_line:
            close('all')
        

    if figures:
    
        alpha = 3
        mode0 = 200
        be = (alpha-1)/mode0
        la0 = mode0 + 1/be
        mz_P= pybmca( la0=la0, nu0=(la0**2)/alpha, 
                     r1 = r1, r2 = r2, c = c, A=A)
        mz = pybmca_nbinom( la0=la0, nu0=(la0**2)/alpha, 
                     r1 = r1, r2 = r2, c = c, A=A)
    
        ### Compare Nb with Poisson with an example
        m_star = 5 #Optimal number of quadrants
        a_star = mz.A*m_star #Optimal areaq( a=TTomasa_a2, n=TTomasa_n2, vector=False)
    
        ### Example using this design
        La = 500 #True Lambda
        ### Simulate data with Nb, with v = 2mu
        ns = mz.SimNsNb( m=5, A=mz.A, la_true=La, phi_true=2*La)
        ### And data from Poisson
        n_obs = mz_P.SimNs( T=1, a=a_star, la_true=La) 
        mz.ns = ns
        ### Plot the posterior for Lambda with the Poisson
        ax = mz_P.PlotPostLa( a=a_star, n=n_obs, la_true=La,   fill=True, color='blue', ax=None, lw=1.5)
        ax.set_title(r"$m=%d, n=%d$" % ( m_star, n_obs))
        mz.RunMCMC(N=30_000)
        mz.PlotMargLa( la_true=La, ax=ax)
        ax.get_figure().tight_layout()    
        ax.get_figure().savefig("%s/Nb/NbPostCompare1.png" %path )
        
        ### Now simulate data with Poisson in both cases
        ns = mz.SimNsPoisson( m=5, A=mz.A, la_true=La)
        mz.ns = ns
        ### Use the same data in both cases:
        n_obs = sum(ns) #mz_P.SimNs( T=1, a=a_star, la_true=La) 
        ### Plot the posterior for Lambda with the Poisson
        ax = mz_P.PlotPostLa( a=a_star, n=n_obs, la_true=La,   fill=True, color='blue', ax=None, lw=1.5)
        ax.set_title(r"$m=%d, n=%d$" % ( m_star, n_obs))
        mz.RunMCMC(N=30_000)
        mz.PlotMargLa( la_true=La, ax=ax)
        ax.get_figure().tight_layout()    
        ax.get_figure().savefig("%s/Nb/NbPostCompare2.png" %path )
    
    if over_dispersion:
    
        alpha = 3
        mode0 = 200
        be = (alpha-1)/mode0
        la0 = mode0 + 1/be
        mz_P= pybmca( la0=la0, nu0=(la0**2)/alpha, 
                     r1 = r1, r2 = r2, c = c, A=A)
        a = 3
        mz = pybmca_nbinom( la0=la0, nu0=(la0**2)/alpha, 
                     r1 = r1, r2 = r2, c = c, A=A, a=3, b=1) #Over dispersion
    
        ### Compare Nb with Poisson with an example
        m_star = 5 #Optimal number of quadrants
        a_star = mz.A*m_star #Optimal areaq( a=TTomasa_a2, n=TTomasa_n2, vector=False)
        ### Example using this design, with more overdispersion
        La = 500 #True Lambda
        ### Simulate data with Nb, with v = 2mu
        ns = mz.SimNsNb( m=5, A=mz.A, la_true=La, phi_true=2*La)
        ### And data from Poisson
        n_obs = mz_P.SimNs( T=1, a=a_star, la_true=La) 
        mz.ns = ns
        ### Plot the posterior for Lambda with the Poisson
        ax = mz_P.PlotPostLa( a=a_star, n=n_obs, la_true=La,   fill=True, color='blue', ax=None, lw=1.5)
        ax.set_title(r"$m=%d, n=%d$" % ( m_star, n_obs))
        mz.RunMCMC(N=30_000)
        mz.PlotMargLa( la_true=La, ax=ax)
        ax.get_figure().tight_layout()    
        ax.get_figure().savefig("%s/Nb/NbPostCompare_OD%d_1.png" % (path,a) )
        
        ### Now simulate data with Poisson in both cases
        ns = mz.SimNsPoisson( m=5, A=mz.A, la_true=La)
        mz.ns = ns
        ### Use the same data in both cases:
        n_obs = sum(ns) #mz_P.SimNs( T=1, a=a_star, la_true=La) 
        ### Plot the posterior for Lambda with the Poisson
        ax = mz_P.PlotPostLa( a=a_star, n=n_obs, la_true=La,   fill=True, color='blue', ax=None, lw=1.5)
        ax.set_title(r"$m=%d, n=%d$" % ( m_star, n_obs))
        mz.RunMCMC(N=30_000)
        mz.PlotMargLa( la_true=La, ax=ax)
        ax.get_figure().tight_layout()    
        ax.get_figure().savefig("%s/Nb/NbPostCompare_OD%d_2.png" % (path,a) )
    
