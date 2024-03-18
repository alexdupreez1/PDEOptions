import numpy as np
import matplotlib.pyplot as plt


def initialize_grid(S_max, S_min, K, T, r, sigma, Nt, Ns):
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


def discretize(u, x, t, dx, dt, r, sigma):
    Nx = len(x) - 1
    Nt = len(t) - 1

    a_neg = -(r-0.5*(sigma**2))*dt/(2*dx) + 0.5*(sigma**2)*dt/(dx**2)
    a     = 1 - (sigma**2)*dt/(dx**2) - r*dt
    a_pos = (r-0.5*(sigma**2))*dt/(2*dx) + 0.5*(sigma**2)*dt/(dx**2)

    for n in range(0, Nt):
        for i in range(1, Nx):
            u[i, n+1] = a_neg*u[i-1, n] + a*u[i, n] + a_pos*u[i+1, n]

    return u


def plot_solution(u, x, t):
    plt.figure()
    plt.plot(np.exp(x), u[:, 0], label='$t=0$')
    plt.xlabel('$S$')
    plt.ylabel('Payoff')
    plt.legend()
    plt.show()


def main():
    # set parameters
    S_max = 200
    S_min = 1
    K = 99
    T = 1
    r = 0.06
    sigma = 0.2
    Nt = 1000
    Ns = 1000
    
    # initialize grid
    u, x, t, dx, dt = initialize_grid(S_max, S_min, K, T, r, sigma, Nt, Ns)
    
    # discretize the PDE
    u = discretize(u, x, t, dx, dt, r, sigma)
    
    # plot the solution at t=0
    plot_solution(u, x, t)


if __name__ == '__main__':
    main()
