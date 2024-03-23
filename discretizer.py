import numpy as np
import matplotlib.pyplot as plt


class FTCS:

    def __init__(self, S_max, S_min, K, T, r, sigma, Nt, Ns):
        self.S_max = S_max
        self.S_min = S_min
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.Nt = Nt
        self.Ns = Ns
        self.u, self.x, self.t, self.dx, self.dt = self.initialize_grid()
        self.u = self.discretize()


    def initialize_grid(self):
        S_max = self.S_max
        S_min = self.S_min
        K = self.K
        T = self.T
        r = self.r
        Nt = self.Nt
        Ns = self.Ns

        # set up the grid        
        Nx = Ns
        X_min = np.log(S_min)
        X_max = np.log(S_max)
        dt = T/Nt
        dx = (X_max-X_min)/Nx

        # create the grid in space and time
        x = np.linspace(X_min, X_max, Nx+1)  # +1 to include the boundary
        t = np.linspace(0, T, Nt+1)
        u = np.zeros((Nx+1, Nt+1))

        # set initial condition
        u[:, 0] = np.maximum(np.exp(x) - K, 0)
        
        # set boundary conditions
        u[0, :] = 0
        u[Nx, :] = np.exp(X_max) - K*np.exp(-r*t)

        return u, x, t, dx, dt


    def discretize(self):
        sigma = self.sigma
        r = self.r
        u = self.u
        x = self.x
        t = self.t
        dx = self.dx
        dt = self.dt

        Nx = len(x) - 1
        Nt = len(t) - 1

        a_neg = -(r-0.5*(sigma**2))*dt/(2*dx) + 0.5*(sigma**2)*dt/(dx**2)
        a     = 1 - (sigma**2)*dt/(dx**2) - r*dt
        a_pos = (r-0.5*(sigma**2))*dt/(2*dx) + 0.5*(sigma**2)*dt/(dx**2)

        for n in range(0, Nt):
            for i in range(1, Nx):
                u[i, n+1] = a_neg*u[i-1, n] + a*u[i, n] + a_pos*u[i+1, n]

        return u

    # def plot_solution(self):
    #     plt.figure()
    #     plt.plot(np.exp(self.x), self.u[:, 0], label='$t=0$')
    #     plt.xlabel('$S$')
    #     plt.ylabel('Payoff')
    #     plt.legend()
    #     plt.show()

    # def plot_grid(self):
    #     plt.figure()
    #     plt.imshow(self.u, origin='lower', aspect='auto')
    #     plt.xlabel('$\\tau$')
    #     plt.ylabel('S')
    #     plt.show()

    def plot_solution(self):
        # Plotting
        plt.figure(figsize=(12, 6))

        # Plot the final option prices as a function of the stock prices
        plt.subplot(1, 2, 1)
        plt.plot(np.exp(self.x), self.u[:, 0], label='Option Value at t=0')
        # plt.plot(np.exp(self.x), self.u[:, -1], label='Option Value at t=T')
        plt.xlabel('Stock Price')
        plt.ylabel('Option Value')
        plt.legend()

        # Plot the grid (evolution of the option price over time for a range of stock prices)
        plt.subplot(1, 2, 2)
        S_grid, t_grid = np.meshgrid(np.exp(self.x), self.t, indexing='ij')
        plt.contourf(S_grid, t_grid, self.u, 50, cmap='viridis')
        plt.colorbar(label='Option Value')
        plt.xlabel('Stock Price')
        plt.ylabel('Time to Maturity')
        plt.tight_layout()

        plt.show()


def main():
    # set parameters
    params = {
        'S_max': 200,
        'S_min': 1,
        'K': 99,
        'T': 1,
        'r': 0.06,
        'sigma': 0.2,
        'Nt': 1000,
        'Ns': 1000
    }
    
    # create the FTCS object
    ftcs = FTCS(**params)

    # plot the solution at t=0
    ftcs.plot_solution()


if __name__ == '__main__':
    main()
