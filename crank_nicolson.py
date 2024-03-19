import numpy as np
import matplotlib.pyplot as plt


# params
S_max = 200
S_min = 1
X_max = np.log(S_max)
X_min = np.log(S_min)
T = 1
K = 99
r = 0.06
sigma = 0.2
Nt = 1000
Ns = 1000
dt = T/Nt
dx = (X_max - X_min)/Ns

# create the matrices
alpha = (r-0.5*(sigma**2))*(dt/(4*dx))
beta = 0.25*(sigma**2)*(dt/(dx**2))
gamma = (r*dt)/2

def create_matrix_A(n):
    main_diag_val = 1 + 2 * beta + gamma
    sub_diag_val = alpha - beta
    super_diag_val = -alpha - beta

    main_diag = np.full(n, main_diag_val)
    sub_diag = np.full(n - 1, sub_diag_val)
    super_diag = np.full(n - 1, super_diag_val)

    # create the matrix
    A = np.diag(main_diag) + np.diag(sub_diag, -1) + np.diag(super_diag, 1)
    return A

A = create_matrix_A(Ns)

def create_matrix_B(Ns):
    main_diag_val = 1 - 2 * beta - gamma
    sub_diag_val = -alpha + beta
    super_diag_val = alpha + beta

    main_diag = np.full(Ns, main_diag_val)
    sub_diag = np.full(Ns - 1, sub_diag_val)
    super_diag = np.full(Ns - 1, super_diag_val)

    # create the matrix
    B = np.diag(main_diag) + np.diag(sub_diag, -1) + np.diag(super_diag, 1)
    return B

B = create_matrix_B(Ns)

# create the grid
S = np.linspace(S_min, S_max, Ns)
t = np.linspace(0, T, Nt)
u = np.zeros((Ns, Nt))

# set initial condition
u[:, 0] = np.maximum(S - K, 0)

# set boundary conditions
u[0, :] = 0
u[Ns-1, :] = S_max - K*np.exp(-r*t)

# solve the system
for n in range(0, Nt-1):
    b = B @ u[:, n]
    u[:, n+1] = np.linalg.solve(A, b)

# plot the solution
plt.plot(S, u[:, 0], label='t=0')
plt.xlabel('S')
plt.ylabel('u')
plt.legend()
plt.show()
