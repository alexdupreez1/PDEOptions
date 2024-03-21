import numpy as np
from scipy.integrate import quad
import scipy.stats as si
import matplotlib.pyplot as plt



class CosMethod:

    def __init__(self,r,sigma,S0,K,T,L,n_iterations,i = 1j):

        self.r = r
        self.sigma = sigma
        self.S0 = S0
        self.K = K
        self.T = T
        self.L = L
        self.i = i
        self.n_iterations = n_iterations

        #To be declared later
        self.u = None
        self.x = None
        self.y = None
        self.n = None
        self.a = None
        self.b = None
        self.cf = None
        self.F_n = None
        self.G_n = None


    def characteristic_function(self):

        """
        Description
        ----------------
        Computes the characteristic function
        """

        n = np.arange(0,self.n_iterations+1,1) #Creating a vector of n's for the iterations

        c1 = (self.r - 0.5 * self.sigma**2) * self.T
        c2 = self.sigma**2 * self.T
        a = c1 - self.L * np.sqrt(abs(c2))
        b = c1 + self.L * np.sqrt(abs(c2))

        # a = 0.0 - self.L * np.sqrt(self.T) #Computing a for the rest of the simulation
        # b = 0.0 + self.L * np.sqrt(self.T) #Computing b for the rest of the simulation
        
        self.n = n 
        self.a = a #Store n, a and b
        self.b = b


        u = (self.n * np.pi) / (self.b - self.a)
        x = np.log(self.S0/self.K)

        cf = np.exp(self.i*u*(x+(self.r-0.5*(self.sigma**2))*self.T) - 0.5*self.T*(self.sigma**2)*(u**2))

        #Store u and x in class
        self.u = u
        self.x = x
        self.cf = cf

    def compute_Fn(self):

        """
        Description
        ----------------
        Computes the F_n coefficient used in the approximation
        """

        F_n = np.real(self.cf*np.exp(-self.i*((self.n*np.pi*self.a)/(self.b-self.a))))

        self.F_n = F_n

    def compute_Gn(self):
       
        """
        Description
        ----------------
        Computes the G_n coefficients used in the approximation
        """

        G_n = np.zeros_like(self.n)

        
        for i, n_value in enumerate(self.n):# Compute G_n for each element in n
            integrand = lambda y: (self.K * np.maximum(np.exp(y) - 1, 0)) * np.cos(n_value * np.pi * ((y - self.a) / (self.b - self.a)))
            integral, _ = quad(integrand, self.a, self.b)
            G_n[i] = (2 / (self.b - self.a)) * integral

        self.G_n = G_n


    def compute_price_approximation(self):

        self.F_n[0]*=0.5 #Multiple first element by 1/2

        price = np.exp(-self.r*self.T) * (np.dot(self.F_n,self.G_n))

        return price

    def analytical_solution(self):

        '''
        Description
        -----------
        Calculates the value of a European call option 
        using the Black-Scholes formula.
        '''
        d1 = (np.log(self.S0/self.K) + (self.r + (self.sigma**2)/2)*self.T) / (self.sigma*np.sqrt(self.T))
        N_d1 = si.norm.cdf(d1)
        d2 = d1 - self.sigma*np.sqrt(self.T)
        N_d2 = si.norm.cdf(d2)

        return self.S0*N_d1 - np.exp(-self.r*self.T)*self.K*N_d2


if __name__ == "__main__":

    price_ests = []
    n_terms = np.arange(1,65,1)

    for n in n_terms:

        cos_object = CosMethod(
            r = 0.04,
            sigma = 0.30,
            S0 = 100,
            K = 110,
            T = 1,
            L = 11,
            n_iterations = n,
        )
        cos_object.characteristic_function()
        cos_object.compute_Fn()
        cos_object.compute_Gn()
        price_est = cos_object.compute_price_approximation()
        price_ests.append(price_est)

    price_analytical = cos_object.analytical_solution()

    plt.plot(n_terms,price_ests, color = "blue",label = "COS-Method Estimate")
    plt.axhline(price_analytical,linestyle = "--", color = "green", label = "BS Solution")
    plt.legend()
    plt.show()
    



