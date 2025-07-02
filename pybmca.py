#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 22 15:19:47 2025

@author: jac

pybmca:

Python implementation of the Bayesian Microplastics Count Analyses
"""

from numpy import floor, zeros, ones, mean, linspace, argmin, array, arange
from scipy.stats import poisson, gamma, multinomial, dirichlet, beta
from matplotlib.pylab import subplots

from plotfrozen import PlotFrozenDist

#from two_scales import TwoScales_x_Axes, TwoScales_y_Axes

def slide(x):
    return max( 0.0, min( 1.0, x))

def ExtractSimData( Sim, i):
    a = Sim['a']
    n_true = Sim['True_n_p'][i,0]
    p_true = Sim['True_n_p'][i,1:]
    s = Sim['Data_s'][i, :]
    return a, p_true, n_true, s 

class pybmca:
    """
       Utilities to analyze microplasticssampling data.
       see ... .
       
       la0, nu0: prior mean and variance for the Gamma prior of 
         lambda, (# of microplastics / m^2).
       r1: relative cost of counting one MP           
       r2: and relative cost of categorizing one microplastic vs 
         both vs analyzing one squere meter of sand.
       c: square meters that may be sampled if the whole budget
         is spent in sampling in the beach 
       A: area sampled per quadrant
       k: number of microplastics categories
       ga_i_0: \gamma vecotr of parameters for the Dirichlet prior for p
         is ones(k)*ga_i_0.  May be a scalar or vector.
    """
    def __init__( self, la0, nu0, r1, r2, c, A=0.25*0.25, ga_i_0=1):
        self.la0 = la0
        self.nu0 = nu0
        self.beta = self.la0/self.nu0 #Rate parameter, gamma prior for la
        self.alpha = self.la0*self.beta #shape
        self.la_prior = gamma(self.alpha, scale=1/self.beta)
        self.MP_names = ['PE', 'PP', 'PET', 'PS', 'PA', 'PVC', 'PU', 
                         'AC', 'PES', 'NPP']
        self.k = len(self.MP_names)
        self.ga = ones(self.k)*ga_i_0 #parameters for the Dirichlet prior
        self.ga0 = sum(self.ga)
        self.A = A
        self.c = c
        self.r1 = r1
        self.r2 = r2

    
    def PlotLaPrior( self, color='green', ax=None):
        ax = PlotFrozenDist( self.la_prior, color=color, ax=ax)
        ax.set_xlabel(r"MP m$^{-2}$")
        ax.set_title(r"Prior for $\Lambda$")
        ax.grid(which='both', color='grey', linestyle='dotted', linewidth=1)
        return ax
    
    def SimNs( self, T, a, la_true=None):
     """Simulate T n's assuming a disign area a."""
     if la_true is None:
         la = gamma.rvs( self.alpha, scale=1/self.beta, size=T)
     else:
         la = ones(T)*la_true
     return poisson.rvs( a*la )
 
    def SimData( self, T, a, la_true=None, n_true=None, p_true=None):
        """Simulate T data assuming a disign using area a and proportion
           q of categorized microplastics.
           returns T x (k+1) matrix, the first column with the n's
             the rest of the columns with the s' (categories counts)
         
        """
        if n_true is None:
            ns = self.SimNs( T, a, la_true)
        else:
            ns = n_true
        True_n_p = zeros(( T, self.k+1))
        True_n_p[ :, 0] = ns
        Data_s = zeros(( T, self.k), dtype=int)
        for i,n in enumerate(True_n_p[ :, 0]):
            n_bar_q = floor(n * self.q( a, n, vector=False))
            if p_true is None:
                p = dirichlet.rvs( self.ga )[0,:]
            else:
                p = p_true
            True_n_p[ i, 1:] = p 
            Data_s[ i, :] = multinomial.rvs( n_bar_q, p)                
        return { "la_true":la_true, "n_true":n_true, "p_true":p_true,
                "True_n_p":True_n_p, "Data_s":Data_s, "a":a, "alpha":self.alpha, "beta":self.beta, "ga":self.ga}

    def PostLa( self, a, n):
        """Define the posterior for lambda."""
        return gamma( self.alpha + n, scale=1/(self.beta + a))

    def PlotPostLa(self, a, n, la_true=None, title=True, plot_prior=True, \
                   linestyle='-', marker='o', color='k', fill=False, ax=None, **kwargs):
        """Plot the posterior for lambda."""
        ax = PlotFrozenDist(self.PostLa( a, n), 
            linestyle=linestyle, color=color, fill=fill, ax=ax, **kwargs)
        xl = ax.get_xlim()
        x = linspace( xl[0], xl[1], num=100)
        if plot_prior:
            ax.plot( x, self.la_prior.pdf(x), '-', color="green")
        #ax.hlines( 0, xl[0], xl[1], color="grey")
        ax.set_xlabel(r"MP m$^{-2}$")
        ax.grid(which='both', color='grey', linestyle='dotted', linewidth=1)
        if not(la_true is None):
            # True value vertical marker
            ax.vlines( la_true, 0, 0.1*ax.get_ylim()[1], linestyle="-", color="black", lw=1)
        if title:
            ax.set_title(r"Posterior for $\Lambda$, observed n= %d" % (n,))
        return ax
    
    def PostMargP( self, s, i):
        """i-th marginal of dirichlet( self.ga + m )."""
        a, b = self.ga[i] + s[i], sum(self.ga + s) - (self.ga[i] + s[i]) 
        return beta( a, b )

    def PlotMargsP( self, s, p_true=None, include_MPs=None, linestyle='-', title=True, ax=None):
        """Plot all marginal distributions of p.
             m: data
             p_true: *all* true probiblities, if simulated data, idependet of include_MPs
             include_MPs: include only MPs listed by name
        """
        if include_MPs == None:
            include_MPs = self.MP_names # all
        if ax is None:            
            fig, ax = subplots( ncols=1, nrows=len(include_MPs), sharex=True)
        x = linspace( 0, 1, num=500)
        for j,mp in enumerate(include_MPs):
            i = self.MP_names.index(mp)
            # Posterior parameters
            marg = self.PostMargP( s, i) # beta marginal for P_i
            y = marg.pdf(x)
            # Plot ridge
            ax[j].plot( x, y, color=f"C{i}", lw=1.5, linestyle=linestyle)
            ax[j].fill_between(x, 0, y, alpha=0.3, color=f"C{i}")
            if not(p_true is None):
                # True value vertical marker
                ax[j].plot( p_true[i], 0, marker='*', markersize=5, color=f"C{i}")
            ax[j].set_ylabel(self.MP_names[i])
        mx = 0.0
        for j in range(len(include_MPs)):
            mx = max( mx, ax[j].get_ylim()[1])
        for j in range(len(include_MPs)):
            ax[j].set_ylim((0,mx))
        if title:
            ax[0].set_title(r"$\bar{n}(q) = %d$" % (sum(s),))
        ax[-1].set_xlabel(r"$P$s")
        return ax

    def L1( self, a, n):
        return (self.beta**2/self.alpha) * (self.alpha + n)/((self.beta + a)**2) 

    def L1_star( self, a):
        return 1/(1 + a * self.nu0/self.la0 )
    
    def L2_star( self, n, q):
        nq = floor(n*q)
        return (self.ga0 + 1 - nq/(self.ga0 + nq))/(1 + self.ga0 + nq)

    def q( self, a, n, vector=True):
        if vector:
            rt = zeros(n.size)
            for j,nj in enumerate(n):
                if nj == 0:
                    rt[j] = 1.0 #define q=1 when n=0, not relevant but avoids de division by 0
                else:
                    rt[j] = slide( (1/self.c - (a + self.r1 * nj))/( self.r2 * nj) )
            return rt
        else:
            if n == 0:
                return 1.0 #define q=1 when n=0, not relevant but avoids de division by 0
            else:
                return slide( (1/self.c - (a + self.r1 * n))/( self.r2 * n) )  

    def ExpL2_star( self, a, T=100_000):
        """Calculate the expected of L2_star sample area a.
        """
        ns = self.SimNs( T, a)
        return mean(self.L2_star(  ns, self.q( a, ns))), mean(ns), mean(self.q( a, ns))

    def PlotExpVR( self,  a_max=None, plot_L1=True, plot_EL2=True, plot_Eq=True, ax=None):
        """Plot the Expected Variance Reduction from 0.1 to a_max."""
        if ax is None:
            fig, ax = subplots()            
        if a_max is None:
            a_max = 1/self.c 
        a = arange( 0, a_max+ 0.1*self.A, step=self.A) #sequance of area sampled per quadrants
        num = a.size
        EL2_star = zeros(num)
        mean_n = zeros(num)
        mean_q = zeros(num)
        for i in range(num):
            if i % 10 == 0:
                print("%d of %d" % (i,num-1))
            EL2_star[i], mean_n[i], mean_q[i] = self.ExpL2_star( a[i] )
        loss = 0.5*self.L1_star(a) + 0.5*EL2_star
        i_star = argmin(loss)
        ax.plot( a, loss, 'k-', linewidth=1.5)
        ax.plot( a[i_star], loss[i_star], color="magenta", marker="*", markersize=10)
        ax.axvline( a[i_star], color='grey', linestyle='solid', alpha=0.5 )
        if plot_L1:
            ax.plot( a, self.L1_star(a), '-.', color='blue')
        if plot_EL2:
            ax.plot( a, EL2_star, '--', color='orange')
        if plot_Eq:
            ax.plot( a, mean_q, 'k--')
        ax.set_ylabel("")
        ax.set_xticks( a, labels=["%d" % (i,) for i in range(a.size)] )
        ax.set_yticks( arange( 0, 1.1, 0.1) )
        #ax.grid(which='both', color='grey', linestyle='dotted', linewidth=1)
        ax.set_xlabel(r"quadrants $m$")
        
        ax2 = ax.twiny()
        ax2.plot( a, loss, 'k-')
        ax2.set_xlabel(r"$a$ (m$^2$)")
        return {'ax':ax, 'i_star':i_star, 'a':a, 'loss':loss, 'EL2_star':EL2_star, 'mean_q':mean_q}



def AnaDesign( inst, a_star, plot_L2=False, ylim=None, ax=None):
    if ax is None:
        fig, ax = subplots()

    La = linspace( 0, 1500, num=100)
    n = La * a_star
    ax.plot( La, inst.q( a=a_star, n=n), '--', color="green")
    if plot_L2:
        ax.plot( La, inst.L2_star( n=n, q=inst.q( a=a_star, n=n)), '--', color="orange")
    ax.plot( La, n*inst.q( a=a_star, n=n)/100, '-', color="red")
    #ax.set_ylim((0,2))
    ax.set_xlabel(r"True $\lambda$ (MP m$^{-2}$)")
    #q_max = ax.get_ylim()[1]
    tksx = array([ 10, 100, 200, 500, 700, 1000, 1500])
    ax.set_xticks( tksx ) #, [r"%d" % (tk,) for tk in tksx])
    if not(ylim is None):
        ax.set_ylim(ylim)
    tksy = ax.get_yticks()[1:-1]
    ax.set_yticks( tksy, ["%d" % (tk*100,) for tk in tksy])
    #ax.set_ylabel(r"$\bar{n}(q)$")
    ax.set_ylabel("")
    ax.grid(which='both', color='grey', linestyle='solid', alpha=0.5, linewidth=1)

    return ax    

def PlotDesignFigures( number, mode0, r1, r2, c, alpha=3, plot_L2=False, ana_ylim=None, inset='LaPrior'):
    alpha = 3
    mode0 = mode0
    be = (alpha-1)/mode0
    la0 = mode0 + 1/be
    mz = pybmca(la0= la0, nu0= (la0**2)/alpha, r1 = r1, r2 = r2, c = c)
    rt = mz.PlotExpVR(plot_Eq=False)
    i_star = rt['i_star']
    a_star = rt['a'][i_star]
    mean_q_star = rt['mean_q'][i_star]
    print("a_star = %f, mean_q_star = %f" % (a_star, mean_q_star))
    print("Expected number of microplastics to be categorize = %f" % (a_star * la0 * mz.q( a_star, a_star * la0, vector=False),))

    ax = rt['ax']
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
        ax_ana.get_figure().savefig("AnaDesign%d.png" % (number,))
        
    ax.get_figure().tight_layout()    
    ax.get_figure().savefig("ExpLoss%d.png" % (number,))

    return mz, rt, ax_ana, ax_inset

    
if __name__ == "__main__":
    ### The area sampled per 1 m2 quadrant is:
    A = 0.25*0.25 #=  0.0625
    ### Cost of sampling 1 m2 quadrant: (25 + 128)/5 + 154 + 616 + 616 + 154 = 1570
    ### Then sampling 0.0625 m2 costs 1570.
    ### Sampling 1 m2 costs 1570/0.0625 = 25120.0
    ### addehttps://matplotlib.org/stable/gallery/subplots_axes_and_figures/zoom_inset_axes.html#sphx-glr-gallery-subplots-axes-and-figures-zoom-inset-axes-pyd cost of counting samples: 1.28 per particle
    r1 = 1.28/25120 #aprox. 5e-5
    r1 = 5e-5 # rounded toq( a=TTomasa_a2, n=TTomasa_n2, vector=False)
    
    ### categorization per particle: 70 local 1000 foreing    
    r2 = 70/25120 #aprox 0.002 to 0.02
    r2 = 3e-3
    
    ### Maximum number of particles to analyze: 250
    ### Cost: 250 * 70 
    ###         beach    Categorization
    ### budget: 1570*5 + 150*70         = 25350q( a=TTomasa_a2, n=TTomasa_n2, vector=False)
    bt = 1570*5 + 200*70 #        
    ### c2: How many **m2 can we sample** if all budget is spent in the beach?
    ### Sampling 1 m2 costs 1570/0.0625 = 25120.0
    c = (1570/0.0625)/bt #1.14, 1/(mz1.A * mz1.c) = 13.9
    
    ###       pristine   afected   strongly polluted
    ### la0 =     10< ,      100,  >1000 
    
    from matplotlib.pylab import close    
    from matplotlib import rcParams
    rcParams.update({'font.size': 14})
    
    joan = False
    design = True
    example = True
    if joan:
        ###  Setting and calculate the design
        mz, rt, ax_ana, ax_inset = PlotDesignFigures( number=5,
                   mode0=200, r1=r1, r2=r2, c=1/(12 * A), alpha=3, plot_L2=True)
        ### Resulting design
        m_star = rt['i_star'] #Optimal number of quadrants
        a_star = mz.A*m_star #Optimal areaq( a=TTomasa_a2, n=TTomasa_n2, vector=False)

        ### Example using this design
        La = 500 #True Lambda
        n_obs = floor(La*a_star) #Expected number of observations
        ### PLot the posterior for Lambda
        ax = mz.PlotPostLa( a=a_star, n=n_obs, la_true=La,   fill=True, color='blue', ax=None, lw=1.5)
        ax.set_title(r"$m=%d, n=%d$" % ( m_star, n_obs)) 
        ### True polymer proportions %
        ### ['PE', 'PP', 'PET', 'q( a=TTomasa_a2, n=TTomasa_n2, vector=False)PS', 'PA', 'PVC', 'PU', 'AC', 'PES', 'NPP']
        P = \
            [  52,   34,     0,   13,    1,     0,    0,    0,     0,    0 ]
        P = array(P)/100
        ### Calculate q, only q*n_obs will be categorized
        q = mz.q( a=a_star, n=n_obs, vector=False)
        sq = P * n_obs * q #Expected number of counts for each polymer type
        ### Plot the posterior marginals for P
        ax_P = mz.PlotMargsP( s=sq, p_true=P,\
                           include_MPs=['PE', 'PP', 'PS', 'PA'])
        ### If ALL MPs are categorized
        s = P * n_obs #Expected number of counts
        ### Plot the posterior marginals for P
        ax_P = mz.PlotMargsP( s=s, p_true=P,\
                           include_MPs=['PE', 'PP', 'PS', 'PA'], linestyle='--', ax=ax_P)
        ax_P[0].set_title(r"$\bar{n}_2(q) = %d$ (solid) vs. $n = %d$ (dashed)" % ( sum(sq), sum(s)))
        ax_P[-1].set_xlim((0,0.65))

        
    if design:
        ### Illustration moving a priori
        mz1, rt1, ax_ana1, ax_inset1 = PlotDesignFigures( number=1,
                   mode0=200, r1=r1, r2=r2, c=1/(12 * A), alpha=3, plot_L2=True, ana_ylim=(0,1.7))
        mz2, rt2, ax_ana2, ax_inset2 = PlotDesignFigures( number=2,
                   mode0=800, r1=r1, r2=r2, c=1/(12 * A), alpha=3, plot_L2=True, ana_ylim=(0,1.7)) 
        ### Twice expensive FTIR 
        mz3, rt3, ax_ana3, ax_inset3 = PlotDesignFigures( number=3, 
                   mode0=200, r1=r1, r2=r2*2, c=1/(12 * A), alpha=3, inset='AnaDesign')
        ### Very expensive FTIR
        mz4, rt4, ax_ana4, ax_inset4 = PlotDesignFigures( number=4, 
                   mode0=800, r1=r1, r2=r2*1000, c=1/(12 * A), alpha=3, inset='AnaDesign')
    
        ### Lower budget
        mz5, rt5, ax_ana5, ax_inset5 = PlotDesignFigures( number=5, 
                   mode0=200, r1=r1, r2=r2, c=1/(8 * A), alpha=3, inset='AnaDesign')
        mz6, rt6, ax_ana6, ax_inset1 = PlotDesignFigures( number=6, 
                   mode0=800, r1=r1, r2=r2, c=1/(8 * A), alpha=3, inset='AnaDesign')
    
        ### Insisting in sampling 5 quadrants, even with a contaminated beach
        mz7, rt7, ax_ana7, ax_inset6 = PlotDesignFigures( number=8, 
                   mode0=800, r1=r1, r2=r2, c=1/(14 * A), alpha=3, inset='AnaDesign') 
        ### Same setting but expecting a less contaminated beach
        mz7, rt7, ax_ana7, ax_inset6 = PlotDesignFigures( number=7, 
                   mode0=200, r1=r1, r2=r2, c=1/(14 * A), alpha=3, inset='AnaDesign') 
    if example:
        ### Design, equal to T de la Tomasa
        TTomasa_A = 0.25 * 0.25
        ### ['PE', 'PP', 'PET', 'PS', 'PA', 'PVC', 'PU', 'AC', 'PES', 'NPP']
        TTomasa_P = \
            [  52,   34,     0,   13,    1,     0,    0,    0,     0,    0 ]
        TTomasa_P = array(TTomasa_P)/100
        TTomasa_n = 358
        
        TTomasa_La = TTomasa_n/(TTomasa_A * 15) # True Lambda
        TTomasa_n1 = floor(5 * TTomasa_n/(5*3)) # five quadrants    
        TTomasa_a1 = 5 * TTomasa_A
        TTomasa_n2 = floor(7 * TTomasa_n/(5*3)) # seven quadrants
        TTomasa_a2 = 7 * TTomasa_A 
        
        alpha = 3
        mode0 = 200
        be = (alpha-1)/mode0
        la0 = mode0 + 1/be
        TT = pybmca(la0= la0, nu0= (la0**2)/alpha, r1=r1, r2=r2, c=1/(12 * A))
        ax_ex1_la = TT.PlotPostLa( a=TTomasa_a1, n=TTomasa_n1, la_true=TTomasa_La, linestyle='-',  fill=True, color='blue', ax=None, lw=1.5)
        ax_ex1_la = TT.PlotPostLa( a=TTomasa_a2, n=TTomasa_n2, la_true=TTomasa_La, linestyle='--', fill=True, color='blue',  ax=ax_ex1_la, lw=1.5)
        #ax_ex1_la.set_title(r"$m_1=5, n_1=%d$ vs. $m_2=7, n_2=%d$" % (TTomasa_n1,TTomasa_n2)) 
        ax_ex1_la.set_title("")
        ax_ex1_la.get_figure().tight_layout()    
        ax_ex1_la.get_figure().savefig("ExaLaPost1.png")
        s1 = TTomasa_P * TTomasa_n1
        q = TT.q( a=TTomasa_a2, n=TTomasa_n2, vector=False)
        sq = TTomasa_P * TTomasa_n2 * q
        ax_ex1 = TT.PlotMargsP( s=s1, p_true=TTomasa_P,\
                           include_MPs=['PE', 'PP', 'PS', 'PA'])
        ax_ex1 = TT.PlotMargsP( s=sq, p_true=TTomasa_P,\
                           include_MPs=['PE', 'PP', 'PS', 'PA'], linestyle='--', ax=ax_ex1)
        #ax_ex1[0].set_title(r"$n_1 = %d$ vs. $\bar{n}_2(q) = %d$" % ( TTomasa_n1, sum(mq),))
        ax_ex1[0].set_title("")
        ax_ex1[-1].set_xlim((0,0.65))
        ax_ex1[0].get_figure().tight_layout()
        ax_ex1[0].get_figure().savefig("ExaPMargs1.png")
        

        ### Same but witn higher abundance
        TTomasa_n = 600
        TTomasa_La = TTomasa_n/(TTomasa_A * 15) # True Lambda
        TTomasa_n1 = floor(5 * TTomasa_n/(5*3)) # five quadrants    
        TTomasa_a1 = 5 * TTomasa_A
        TTomasa_n2 = floor(7 * TTomasa_n/(5*3)) # seven quadrants
        TTomasa_a2 = 7 * TTomasa_A 
        
        ax_ex2_la = TT.PlotPostLa( a=TTomasa_a1, n=TTomasa_n1, la_true=TTomasa_La, linestyle='-',  fill=True, color='blue', ax=None, lw=1.5)
        ax_ex2_la = TT.PlotPostLa( a=TTomasa_a2, n=TTomasa_n2, la_true=TTomasa_La, linestyle='--',  fill=True, color='blue', ax=ax_ex2_la, lw=1.5)
        #ax_ex2_la.set_title(r"$m_1=5, n_1=%d$ vs. $m_2=7, n_2=%d$" % (TTomasa_n1,TTomasa_n2)) 
        ax_ex2_la.set_title("")
        ax_ex2_la.get_figure().tight_layout()
        ax_ex2_la.get_figure().savefig("ExaLaPost2.png")
        s1 = TTomasa_P * TTomasa_n1
        q = TT.q( a=TTomasa_a2, n=TTomasa_n2, vector=False)
        sq = TTomasa_P * TTomasa_n2 * q
        ax_ex2 = TT.PlotMargsP( s=s1, p_true=TTomasa_P,\
                           include_MPs=['PE', 'PP', 'PS', 'PA'])
        ax_ex2 = TT.PlotMargsP( s=sq, p_true=TTomasa_P,\
                           include_MPs=['PE', 'PP', 'PS', 'PA'], linestyle='--', ax=ax_ex2)
        #ax_ex2[0].set_title(r"$n_1 = %d$ vs. $\bar{n}_2(q) = %d$" % ( TTomasa_n1, sum(mq),))
        ax_ex2[0].set_title("")
        ax_ex2[-1].set_xlim((0,0.65))
        ax_ex2[0].get_figure().tight_layout()
        ax_ex2[0].get_figure().savefig("ExaPMargs2.png")
        
        ### An example with low abudance
        La = 5
        ax_ex3 = TT.PlotPostLa( a=5*TT.A, n=floor(La*TT.A*5), la_true=La)
        ax_ex3 = TT.PlotPostLa( a=7*TT.A, n=floor(La*TT.A*7), la_true=La, linestyle='--', ax=ax_ex3)
        ax_ex3 = TT.PlotLaPrior( ax=ax_ex3)
        ax_ex3.set_xlim((-50,700))
        ax_ex3.grid(visible=False)
        #ax_ex3.set_title(r"$m_1=5, n_1=%d$ vs. $m_2=7, n_2=%d$" % (floor(La*TT.A*5),floor(La*TT.A*7))) 
        ax_ex3.set_title("")
        ax_ex3_inset = ax_ex3.inset_axes(bounds=[ 0.4, 0.3, 0.5, 0.6],\
                    xlim=( 0, 75), ylim=( -0.001, 0.09))
        ax_ex3_inset = TT.PlotPostLa( a=5*TT.A, n=floor(La*TT.A*5), la_true=La, linestyle='-',   fill=True, color='blue', ax=ax_ex3_inset, lw=1.5)
        ax_ex3_inset = TT.PlotPostLa( a=7*TT.A, n=floor(La*TT.A*7), la_true=La, linestyle='--',  fill=True, color='blue', ax=ax_ex3_inset, lw=1.5)
        ax_ex3_inset.set_title("")
        ax_ex3_inset.set_xlabel("")
        ax_ex3_inset.set_yticklabels([])
        ex3_ylim = ax_ex3.set_ylim()
        ax_ex3.indicate_inset_zoom( ax_ex3_inset, edgecolor="black")
        ax_ex3.get_figure().tight_layout()
        ax_ex3.get_figure().savefig("ExaLowLa.png")

        ### An example with midium abudance
        La = 80
        ax_ex4 = TT.PlotPostLa( a=5*TT.A, n=floor(La*TT.A*5), la_true=La,   fill=True, color='blue', ax=None, lw=1.5)
        ax_ex4 = TT.PlotPostLa( a=7*TT.A, n=floor(La*TT.A*7), la_true=La, linestyle='--',   fill=True, color='blue', ax=ax_ex4, lw=1.5)
        ax_ex4 = TT.PlotLaPrior( ax=ax_ex4)
        #ax_ex4.set_title(r"$m_1=5, n_1=%d$ vs. $m_2=7, n_2=%d$" % (floor(La*TT.A*5),floor(La*TT.A*7))) 
        ax_ex4.set_title("")
        ax_ex4.set_xlim((0,250))
        ax_ex4.set_ylim(ex3_ylim)
        ax_ex4.get_figure().tight_layout()
        ax_ex4.get_figure().savefig("ExaMidiumLa.png")
        
        
        