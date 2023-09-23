import numpy as np
import scipy.linalg
import time

from .laplacian import solve_poisson, laplace
#from .laplacian import select_skewherm as select_skewherm_laplacian

# ----------------
# GLOBAL VARIABLES
# ----------------

# _SKEW_HERM_ = True  # Is the dynamics skew-Hermitian?
# _SKEW_HERM_PROJ_FREQ_ = -1  # How many steps before skew-Hermitian projection, negative = never


# ---------------------
# LOWER LEVEL FUNCTIONS
# ---------------------
def initial_values_U(N):
    X0 = np.zeros((N,N), dtype=np.complex128)
    A = np.random.rand(N,N) + np.random.rand(N,N) * 1j
    Y0,Z0 = np.linalg.qr(A);
    X0 = Y0/np.linalg.det(Y0)**(1/N);
    return X0
def LieBracket(W, P):
    """
    Commutator for arbitrary matrices.

    Parameters
    ----------
    W: ndarray
    P: ndarray

    Returns
    -------
    ndarray
    """
    return W@P - P@W


# def commutator_skewherm(W, P):
#     """
#     Efficient computations of commutator for skew-Hermitian matrices.

#     Parameters
#     ----------
#     W: ndarray
#     P: ndarray

#     Returns
#     -------
#     ndarray
#     """
#     VF = W@P
#     VF -= VF.conj().T
#     return VF


# Select default commutator
#commutator = commutator_skewherm


# Project to skewherm (to avoid drift)
# def project_skewherm(W):
#     W /= 2.0
#     W -= W.conj().T


# Function to update solver statistics
def update_stats(stats: dict, **kwargs):
    for arg, val in kwargs.items():
        if arg in stats and np.isscalar(val):
            stats[arg] += val
        else:
            stats[arg] = val


# ----------------------
# HIGHER LEVEL FUNCTIONS
# ----------------------

# def select_skewherm(flag):
#     """
#     Select whether integrators should work with
#     skew Hermitian matrices.

#     Parameters
#     ----------
#     flag: bool

#     Returns
#     -------
#     None
#     """
#     global _SKEW_HERM_
#     global commutator
#     if flag:
#         commutator = commutator_skewherm
#         _SKEW_HERM_ = True
#     else:
#         commutator = LieBracket
#         _SKEW_HERM_ = False
#     select_skewherm_laplacian(flag)


# -------------------------------------------------
# CLASSICAL (EXPLICIT, NON-ISOSPECTRAL) INTEGRATORS
# -------------------------------------------------

# def euler(W: np.ndarray,
#           stepsize: float = 0.1,
#           steps: int = 100,
#           hamiltonian=solve_poisson,
#           forcing=None,
#           stats: dict = None,
#           **kwargs) -> np.ndarray:
#     """
#     Time-stepping by Euler's explicit first order method.

#     Parameters
#     ----------
#     W: ndarray
#         Initial vorticity (overwritten and returned).
#     stepsize: float
#         Time step length.
#     steps: int
#         Number of steps to take.
#     hamiltonian: function(W)
#         The Hamiltonian returning a stream matrix.
#     forcing: None or function(P, W)
#         Extra force function (to allow non-isospectral perturbations).
#     stats: None or dict
#         Dictionary with statistics
#     **kwargs: dict
#         Extra keyword arguments

#     Returns
#     -------
#     W: ndarray
#     """
#     if forcing is None:
#         rhs = commutator
#     else:
#         def rhs(P, W):
#             return commutator(P, W) + forcing(P, W)

#     for k in range(steps):
#         P = hamiltonian(W)
#         VF = rhs(P, W)
#         W += stepsize*VF

#     if stats is not None:
#         update_stats(stats, steps=steps)

#     return W


# def heun(W, stepsize=0.1, steps=100, hamiltonian=solve_poisson, forcing=None):
#     """
#     Time-stepping by Heun's second order method.

#     Parameters
#     ----------
#     W: ndarray
#         Initial vorticity (overwritten and returned).
#     stepsize: float
#         Time step length.
#     steps: int
#         Number of steps to take.
#     hamiltonian: function(W)
#         The Hamiltonian returning a stream matrix.
#     forcing: function(P, W)
#         Extra force function (to allow non-isospectral perturbations).

#     Returns
#     -------
#     W: ndarray
#     """
#     if forcing is None:
#         rhs = commutator
#     else:
#         def rhs(P, W):
#             return commutator(P, W) + forcing(P, W)

#     for k in range(steps):

#         # Evaluate RHS at W
#         P = hamiltonian(W)
#         F0 = rhs(P, W)

#         # Compute Heun predictor
#         Wprime = W + stepsize*F0

#         # Evaluate RHS at predictor WP
#         P = hamiltonian(Wprime)
#         F = rhs(P, Wprime)

#         # Compute averaged RHS
#         F += F0
#         F *= stepsize/2.0

#         # Update W
#         W += F

#     return W


# def rk4(W, stepsize=0.1, steps=100, hamiltonian=solve_poisson, forcing=None):
#     """
#     Time-stepping by the classical Runge-Kutta fourth order method.

#     Parameters
#     ----------
#     W: ndarray
#         Initial vorticity (overwritten and returned).
#     stepsize: float
#         Time step length.
#     steps: int
#         Number of steps to take.
#     hamiltonian: function(W)
#         The Hamiltonian returning a stream matrix.
#     forcing: function(P, W) or None (default)
#         Extra force function (to allow non-isospectral perturbations).

#     Returns
#     -------
#     W: ndarray
#     """
#     if forcing is None:
#         rhs = commutator
#     else:
#         def rhs(P, W):
#             return commutator(P, W) + forcing(P, W)

#     for k in range(steps):
#         P = hamiltonian(W)
#         K1 = rhs(P, W)

#         Wprime = W + (stepsize/2.0)*K1
#         P = hamiltonian(Wprime)
#         K2 = rhs(P, Wprime)

#         Wprime = W + (stepsize/2.0)*K2
#         P = hamiltonian(Wprime)
#         K3 = rhs(P, Wprime)

#         Wprime = W + stepsize*K3
#         P = hamiltonian(Wprime)
#         K4 = rhs(P, Wprime)

#         W += (stepsize/6.0)*(K1+2*K2+2*K3+K4)

#     return W


# # Default classical integrator
# classical = rk4


# -------------------
# ISOSPECTRAL METHODS
# -------------------

# def isomp_quasinewton(W, stepsize=0.1, steps=100, hamiltonian=solve_poisson, forcing=None,
#                       tol=1e-8, maxit=10, verbatim=False):
#     """
#     Time-stepping by isospectral midpoint second order method using
#     a quasi-Newton iteration scheme. This scheme preserves the eigen-spectrum
#     of `W` up to machine epsilon.

#     Parameters
#     ----------
#     W: ndarray
#         Initial vorticity (overwritten and returned).
#     stepsize: float
#         Time step length.
#     steps: int
#         Number of steps to take.
#     hamiltonian: function
#         The Hamiltonian returning a stream matrix.
#     forcing: function(P, W) or None (default)
#         Extra force function (to allow non-isospectral perturbations).
#     tol: float
#         Tolerance for iterations.
#     maxit: int
#         Maximum number of iterations.
#     verbatim: bool
#         Print extra information if True. Default is False.

#     Returns
#     -------
#     W: ndarray
#     """
#     if forcing is not None:
#         assert NotImplementedError("Forcing for isomp_quasinewton is not implemented yet.")

#     if not _SKEW_HERM_:
#         assert NotImplementedError("isomp_quasinewton might not work for non-skewherm.")

#     Id = np.eye(W.shape[0])

#     Wtilde = W.copy()

#     total_iterations = 0

#     for k in range(steps):

#         # --- Beginning of step ---

#         for i in range(maxit):

#             # Update iterations
#             total_iterations += 1

#             # Update Ptilde
#             Ptilde = hamiltonian(Wtilde)

#             # Compute matrix A
#             A = Id - (stepsize/2.0)*Ptilde

#             # Compute LU of A
#             luA, piv = scipy.linalg.lu_factor(A)

#             # Solve first equation for B
#             B = scipy.linalg.lu_solve((luA, piv), W)

#             # Solve second equation for Wtilde
#             Wtilde_new = scipy.linalg.lu_solve((luA, piv), -B.conj().T)

#             # Compute error
#             resnorm = scipy.linalg.norm(Wtilde - Wtilde_new, np.inf)

#             # Update variables
#             Wtilde = Wtilde_new

#             # Check error
#             if resnorm < tol:
#                 break

#         else:
#             # We used maxit iterations
#             if verbatim:
#                 print("Max iterations {} reached at step {}.".format(maxit, k))

#         # Update W
#         W_new = A.conj().T @ Wtilde @ A
#         np.copyto(W, W_new)

#         # Make sure solution is Hermitian (this removes drift in rounding errors)
#         if _SKEW_HERM_ and k % _SKEW_HERM_PROJ_FREQ_ == _SKEW_HERM_PROJ_FREQ_ - 1:
#             project_skewherm(W)

#         # --- End of step ---

#     if verbatim:
#         print("Average number of iterations per step: {:.2f}".format(total_iterations/steps))

#     return W


# def isomp_simple(W, stepsize=0.1, steps=100, hamiltonian=solve_poisson, forcing=None,
#                  enforce_hermitian=True):
#     """
#     Time-stepping by the simplified isospectral midpoint method.
#     This is an explicit isospectral method but not symplectic. Nor is it reversible.

#     Parameters
#     ----------
#     W: ndarray
#         Initial vorticity (overwritten and returned).
#     stepsize: float
#         Time step length.
#     steps: int
#         Number of steps to take.
#     hamiltonian: function
#         The Hamiltonian returning a stream matrix.
#     forcing: function(P, W) or None (default)
#         Extra force function (to allow non-isospectral perturbations).
#     enforce_hermitian: bool
#         Enforce at every step that the vorticity matrix is hermitian.

#     Returns
#     -------
#     W: ndarray
#     """
#     Id = np.eye(W.shape[0])

#     Wtilde = W.copy()

#     if forcing is not None:
#         assert NotImplementedError("Forcing for isomp_simple is not implemented yet.")

#     for k in range(steps):

#         # --- Beginning of step ---

#         # Update Ptilde
#         Ptilde = hamiltonian(Wtilde)

#         # Compute matrix A
#         A = Id - (stepsize/2.0)*Ptilde

#         if _SKEW_HERM_:
#             # Compute LU of A
#             luA, piv = scipy.linalg.lu_factor(A)

#             # Solve first equation for X
#             X = scipy.linalg.lu_solve((luA, piv), W)

#             # Solve second equation for Wtilde
#             Wtilde = scipy.linalg.lu_solve((luA, piv), -X.conj().T)

#             # Update W
#             W_new = A.conj().T @ Wtilde @ A

#         else:

#             # Solve first equation for X
#             X = np.linalg.solve(A, W)

#             # Solve second equation for Wtilde
#             Aalt = Id + (stepsize/2.0)*Ptilde
#             Wtilde = np.linalg.solve(Aalt.conj().T, X.conj().T).conj().T
#             # The line above could probably be done faster without conj().T everywhere

#             # Update W
#             W_new = Aalt @ Wtilde @ A

#         np.copyto(W, W_new)

#         # Make sure solution is Hermitian (this removes drift in rounding errors)
#         if _SKEW_HERM_ and k % _SKEW_HERM_PROJ_FREQ_ == _SKEW_HERM_PROJ_FREQ_ - 1:
#             project_skewherm(W)

#         # --- End of step ---

#     return W


# def isomp_fixedpoint(W,theta,alpha,Q,P, stepsize=0.1, steps=100, hamiltonian=solve_poisson, forcing=None,
#                       tol=1e-8, maxit=5, verbatim=True):
#     """
#     Time-stepping by isospectral midpoint second order method for skew-Hermitian W
#     using fixed-point iterations.

#     Parameters
#     ----------
#     W: ndarray
#         Initial skew-Hermitian vorticity matrix (overwritten and returned).
#     stepsize: float
#         Time step length.
#     steps: int
#         Number of steps to take.
#     hamiltonian: function
#         The Hamiltonian returning a stream matrix.
#     forcing: function(P, W) or None (default)
#         Extra force function (to allow non-isospectral perturbations).
#     tol: float
#         Tolerance for iterations.
#     maxit: int
#         Maximum number of iterations.
#     verbatim: bool
#         Print extra information if True. Default is False.

#     Returns
#     -------
#     W: ndarray
#     """

#     assert maxit >= 1, "maxit must be at least 1."

#     total_iterations = 0

#     # Initialize
#     dW = np.zeros_like(W)
#     dW_old = np.zeros_like(W)
#     Whalf = np.zeros_like(W)
#     start_time = time.time()
#     for k in range(steps):

#         # --- Beginning of step ---

#         for i in range(maxit):

#             # Update iterations
#             total_iterations += 1

#             # Compute Wtilde
#             np.copyto(Whalf, W)
#             Whalf += dW

#             # Update Ptilde
#             Phalf = solve_poisson(Whalf)
#             Phalf *= stepsize/2.0

#             # Compute middle variables
#             PWcomm = Phalf @ Whalf
#             PWPhalf = PWcomm @ Phalf
#             if _SKEW_HERM_:
#                 PWcomm -= PWcomm.conj().T
#             else:
#                 PWcomm -= Whalf @ Phalf

#             # Update dW
#             np.copyto(dW_old, dW)
#             np.copyto(dW, PWcomm)
#             dW += PWPhalf

#             # Add forcing if needed
#             if forcing is not None:
#                 # Compute forcing if needed
#                 FW = forcing(Phalf/(stepsize/2.0), Whalf)
#                 FW *= stepsize/2.0
#                 dW += FW

#             # Compute error
#             resnorm = scipy.linalg.norm(dW - dW_old, np.inf)

#             # Check error
#             if resnorm < tol:
#                 break

#         else:
#             # We used maxit iterations
#             if verbatim:
#                 print("Max iterations {} reached at step {}.".format(maxit, k))

#         # Update W
#         W += 2.0*PWcomm

#         # Make sure solution is Hermitian (this removes drift in rounding errors)
#         if _SKEW_HERM_ and k % _SKEW_HERM_PROJ_FREQ_ == _SKEW_HERM_PROJ_FREQ_ - 1:
#             project_skewherm(W)

#         # --- End of step ---
#     end_time = time.time()
#     stime = end_time-start_time
#     if verbatim:
#         print("Average number of iterations per step: {:.2f}. Time of execution: {}".format(total_iterations/steps,stime))

#     return W,theta,P,Q


# def isomp_fixedpoint(W,theta,alpha,Q,P, stepsize=0.1, steps=100, hamiltonian=solve_poisson, forcing=None,
#                       tol=1e-8, maxit=5, verbatim=True):
#     """
#     Time-stepping by isospectral midpoint second order method for skew-Hermitian W
#     using fixed-point iterations.

#     Parameters
#     ----------
#     W: ndarray
#         Initial skew-Hermitian vorticity matrix (overwritten and returned).
#     stepsize: float
#         Time step length.
#     steps: int
#         Number of steps to take.
#     hamiltonian: function
#         The Hamiltonian returning a stream matrix.
#     forcing: function(P, W) or None (default)
#         Extra force function (to allow non-isospectral perturbations).
#     tol: float
#         Tolerance for iterations.
#     maxit: int
#         Maximum number of iterations.
#     verbatim: bool
#         Print extra information if True. Default is False.

#     Returns
#     -------
#     W: ndarray
#     """

#     assert maxit >= 1, "maxit must be at least 1."

#     total_iterations = 0

#     # Initialize
#     h = stepsize
#     N = W.shape[0]
#     # print(N)
#     start_time = time.time()
#     for k in range(steps):

#         # --- Beginning of step ---

#         #tildeP,tildeQ = fixed_point_iteration(Q,P,alpha,N,stepsize);
#         Q0 = np.eye(N, dtype=np.complex128)
#         P0 = np.eye(N, dtype=np.complex128)
        
#         for i in range(10):
#             W0 = Q0 @ P0.conj().T
#             W0 = (W0-W0.conj().T)/2
           
#             M1 = solve_poisson(W0);
#             rhsQ = -M1 @ Q0
#             Q1 = Q0-(-Q+Q0-h/2*rhsQ)
                                
#             theta0 = Q0 @ alpha @ np.linalg.inv(Q0)
#             theta0 = (theta0-theta0.conj().T)/2
#             M2 = laplace(theta0);
#             rhsP = (M1.conj().T @ P0+M2.conj().T @ np.linalg.inv(Q0.conj().T) @ alpha.conj().T-
#                     np.linalg.inv(Q0.conj().T) @ alpha.conj().T @ Q0.conj().T @ M2.conj().T @ 
#                     np.linalg.inv(Q0.conj().T));
#             #rhsP = M1.conj().T @ P0
#             P1 = P0-(-P+P0-h/2*rhsP)
            
            
#             Q0=Q1
#             P0=P1


#         tildeP = P1
#         tildeQ = Q1

        

#         # Update W
#         tildeW = tildeQ @ tildeP.conj().T;
#         tildeW = (tildeW-tildeW.conj().T)/2
#         tildeZ = tildeQ @ alpha @ np.linalg.inv(tildeQ);
#         tildeTheta = (tildeZ-tildeZ.conj().T)/2;
#         M1 = solve_poisson(tildeW);
#         rhsQ = -M1@tildeQ;
#         M2 = laplace(tildeTheta);
#         rhsP = (M1.conj().T @ tildeP+M2.conj().T @ np.linalg.inv(tildeQ.conj().T) @ alpha.conj().T-
#                 np.linalg.inv(tildeQ.conj().T) @ alpha.conj().T @ tildeQ.conj().T @ M2.conj().T @ np.linalg.inv(tildeQ.conj().T));
#         #rhsP = M1.conj().T @ tildeP

#         Q += h*rhsQ;
#         P += h*rhsP;

        
#     W = Q @ P.conj().T;
#     theta = (np.eye(N)-h/2*M1) @ tildeZ @ np.linalg.inv(np.eye(N)-h/2*M1);
#     W = (W-W.conj().T)/2;
#     theta = (theta-theta.conj().T)/2;
#     end_time = time.time()
#     stime = end_time-start_time
#     print("Iteration finished. Time of execution: {} s".format(stime))   

#     return W,theta,P,Q


def isomp_fixedpoint(W,theta, stepsize=0.1, steps=100, hamiltonian=solve_poisson, forcing=None,
                      tol=1e-8, maxit=5, verbatim=True):
    """
    Time-stepping by isospectral midpoint second order method for skew-Hermitian W
    using fixed-point iterations.

    Parameters
    ----------
    W: ndarray
        Initial skew-Hermitian vorticity matrix (overwritten and returned).
    stepsize: float
        Time step length.
    steps: int
        Number of steps to take.
    hamiltonian: function
        The Hamiltonian returning a stream matrix.
    forcing: function(P, W) or None (default)
        Extra force function (to allow non-isospectral perturbations).
    tol: float
        Tolerance for iterations.
    maxit: int
        Maximum number of iterations.
    verbatim: bool
        Print extra information if True. Default is False.

    Returns
    -------
    W: ndarray
    """

    assert maxit >= 1, "maxit must be at least 1."

    total_iterations = 0

    # Initialize
    h = stepsize
    N = W.shape[0]
    # print(N)
    start_time = time.time()
    W0 = np.zeros((N,N),dtype=np.complex128)
    theta0 = np.zeros((N,N),dtype=np.complex128)
    for k in range(steps):

        # --- Beginning of step ---

        
        
        for i in range(15):
           
            M1 = solve_poisson(W0)
            M2 = laplace(theta0);
            
            rhsW = h/2*LieBracket(W0,M1)+h/2*LieBracket(theta0, M2)+h**2/4*M1@W0@M1+h**2/4*M2@theta0@M1+h**2/4*M1@theta0@M2
            rhsTheta = h/2*LieBracket(theta0,M1)+h**2/4*M1@theta0@M1
            
            W1 = W+rhsW
            theta1 = theta+rhsTheta
            
            W0 = W1
            theta0 = theta1


        tildeW = W1
        tildeTheta = theta1
        tildeW = (tildeW-tildeW.conj().T)/2
        tildeTheta = (tildeTheta-tildeTheta.conj().T)/2
        M1 = solve_poisson(tildeW)
        M2 = laplace(tildeTheta)
        
        W = W+h*LieBracket(tildeW, M1)+h*LieBracket(tildeTheta,M2)
        theta = theta+h*LieBracket(tildeTheta,M1)


    end_time = time.time()
    stime = (end_time-start_time)/60
    print("Time of execution: {} min".format(stime))   

    return W,theta


# Default isospectral method
isomp = isomp_fixedpoint
