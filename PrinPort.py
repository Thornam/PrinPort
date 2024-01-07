

#########################################################################
# Readme:                                                               #
# This file contains the raw code for the mathematical equation         #
# proposed in the research paper 'Principal Portfolio' by B. Kelly,     #
# S. Malamud and L. Pedersen (2020).                                    #
# Link:                                                                 #
# https://onlinelibrary.wiley.com/doi/full/10.1111/jofi.13199           #
#                                                                       #
#########################################################################

# IMPORT LIBRARIES
#########################################################################

import numpy as np
import pandas as pd
import gc
import torch

# Using the GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#########################################################################
# PREDICTION MATRIX                                            
#########################################################################

def true_prediction_matrix(returns, signals):
    """
    Calculate the prediction matrix Π using asset returns and a signal.
    
    Parameters:
        returns (numpy array or pandas DataFrame): Historical asset returns (shape: (N))
        signals (numpy array or pandas DataFrame): Signals for each asset (shape: (N))
    
    Returns:
        numpy array: The prediction matrix Π (shape: (N, N))
    """
    if len(returns) != len(signals):
        raise ValueError("The number of time periods in returns and signals must be the same.")
        
    prediction_matrix = np.outer(returns, signals)
        
    return prediction_matrix


def estimated_prediction_matrix(returns, signals, D=120, eff=False):
    """
    Calculate the estimated Prediction Matrix based on a backward-looking window of length D
    
    Parameters:
        returns (numpy array or pandas DataFrame): Historical asset returns (shape: (T, N))
        signals (numpy array or pandas DataFrame): Signals for each asset (shape: (T, N))
        D (int): Backward-looking periods to estimate the Prediction Matrix from
        eff (True/False): Efficiency parameter. If True the prediction matrix is created in a more efficient way.
    
    Returns:
        numpy array: The prediction matrix Π (shape: (N, N))
    """
    if len(returns) != len(signals):
        raise ValueError("The number of time periods in returns and signals must be the same.")

    # If the backward-looking window is longer than the asset returns length we set D to simply be the length of asset returns length (Mostly for when we are testing the code)
    if D > returns.shape[0]:
        D = returns.shape[0]

    if eff == True:

        r = torch.tensor(returns[-D:, :]).to(device)
        s = torch.tensor(signals[-D:, :]).to(device)

        del returns, signals
        gc.collect()

        # Calculate the individual predictions matrices over the backward-looking period and taking the mean
        prediction_matrix_tensor = torch.einsum('li,lj-> lij', r, s).mean(0)

        prediction_matrix = prediction_matrix_tensor.cpu().detach().numpy()
    
    else:

        # Calculate the individual predictions matrices over the P backward-looking period
        prediction_matrices = []
        for t in range(1 , D + 1):
            prediction_matrices.append(np.outer(returns[-t], signals[-t])) # Remember the individual Prediction Matrices are appended in reverse order compared to the time periods (-t)
            
        # Calculates the average elementvise across the P Prediction Matrices
        prediction_matrix = np.asarray(prediction_matrices).mean(0)

    return prediction_matrix



#########################################################################
# PRINCIPAL PORTFOLIO                                           
#########################################################################

def principal_portfolio(returns, signals, K=-1, p=2, D=120, eff=False):
    """
    This function calculates the (K) individual Principal Portfolios (PPs) alongside the optimal weights based on the individual PPs, 
    given the asset returns, signals, and the two tunning parameters (K, p)

    Parameters:
        returns (numpy array): Historical asset returns (shape: (T, N))
        signals (numpy array): Signals for each asset (shape: (T, N))
        K (int): Number of largest singular values used (K= -1: uses all singular values).
        p (float): Tunning parameter for the schatten matrix norm used for the calculations (values: [1:∞]). Default=2: Frobenius-norm
        D (int): Backward-looking periods to estimate the Prediction Matrix from
        eff (True/False): Efficiency parameter for the calculation of the Prediction matrix. If True the code is more efficient for large N and D but can give slightly less accurate results

    Returns:
        tuple: A tuple containing:
            - L (numpy.ndarray): The optimal position matrix (shape: (N, N))
            - S (numpy.ndarray): The sorted K Singular Values (shape: (K))
            - L_k (numpy.ndarray): Array of K matrices of each PPs positon matrix order by the the singular value (shape: (K, N, N))
    """

    # Sets q based on the p chosen
    if p == 1:
        q = 0
    else:
        q = 1 / (1 - (1 / p))

    #######################################################################################################
    # Step 1:                                                                                             #
    # Estimate the Prediction Matrix based on the D-period backwards-looking window                       #
    #######################################################################################################

    # Uses the pre-defined Prediction Matrix function
    prediction_matrix = estimated_prediction_matrix(returns, signals, D=D, eff=eff)

    del returns, signals
    gc.collect()

    #######################################################################################################
    # Step 2:                                                                                             #
    # Construct the (K) individual PPs using Singular-Value Decomposition (SVD) of the Prediction Matrix  #
    #######################################################################################################

    # Uses the Numpy linear algebra function for SVD:
    U, S, Vt = np.linalg.svd(prediction_matrix)

    # Transposes the Vt to just being V
    V = Vt.T

    del Vt, prediction_matrix
    gc.collect()

    # The number of singular values used
    if K == -1:
        K = len(S)

    ################################################################################    
    # Restrict Matrices to only include the K largest Singular Values              #
    ################################################################################

    # First we create a dataframe for the singular values to be able to use indexes which can also track the corresponding Singular-vectors
    df = pd.DataFrame(S, columns= ['Singularvalues'])
    df = df.reindex(df['Singularvalues'].sort_values(ascending=False).index) # We do not use the absolute values!

    df = df.iloc[:K, :] # Choose the K largest Singularvalues

    # Next we sort the corresponding singularvectors by the sorted index of df_singular
    U_sort = []
    V_sort = []
    S_sort = []
    for n in df.index:
         U_sort.append(U[:, n]) # Keep in mind that when we append like this the columns becomes the rows, which is why we transpose further down
         V_sort.append(V[:, n]) # Keep in mind that when we append like this the columns becomes the rows, which is why we transpose further down
         S_sort.append(S[n])
    U_sort = np.asarray(U_sort).T # Transpose
    V_sort = np.asarray(V_sort).T # Transpose
    S_sort = np.asarray(S_sort)

    S_sort = S_sort.reshape(S_sort.shape[0], -1)

    del U, S, V
    gc.collect()
    
    ################################################################################

    # Calculating position matrix of the individual PPs
    L_k = np.einsum('il,jl-> lij', V_sort, U_sort)

    del U_sort, V_sort
    gc.collect()

    #######################################################################################################    
    # Step 3:                                                                                             #
    # Calculate the optimal position matrix                                                               #
    #######################################################################################################

    # Calculate c
    df['Singular_c'] = df['Singularvalues'] ** q
    c = df['Singular_c'].sum() ** (-1 / p)    

    # Calculate the optimal position matrix
    L = c * np.einsum('lij, lp -> lij', L_k,  S_sort**(q-1)).sum(axis=0)

    return (L, S_sort, L_k)




#########################################################################
# PRINCIPAL EXPOSURE PORTFOLIO                                           
#########################################################################

def principal_exposure_portfolio(returns, signals, K=-1, p=2, D=120, eff=False):
    """
    This function calculates the (K) individual Principal Exposure Portfolios (PEPs) alongside the optimal weights based on the individual PEPs, 
    given the asset returns, signals, and the two tunning parameters (K, p)

    Parameters:
        returns (numpy array): Historical asset returns (shape: (T, N))
        signals (numpy array): Signals for each asset (shape: (T, N))
        K (int): Number of largest absolute eigenvalues used (K= -1: uses all eigenvalues).
        p (float): Tunning parameter for the schatten matrix norm used for the calculations (values: [1:∞]). Default=2: Frobenius-norm
        D (int): Backward-looking periods to estimate the Prediction Matrix from
        eff (True/False): Efficiency parameter for the calculation of the Prediction matrix. If True the code is more efficient for large N and D but can give slightly less accurate results

    Returns:
        tuple: A tuple containing:
            - L (numpy.ndarray): The optimal position matrix (shape: (N, N))
            - E (numpy.ndarray): The sorted K largest absolute Eigenvalues (shape: (K))
            - L_k (numpy.ndarray): Array of K matrices of each PEPs positon matrix ordered by the the absolute eigenvalues (shape: (K, N, N))
    """

    # Sets q based on the p chosen
    if p == 1:
        q = 0
    else:
        q = 1 / (1 - (1 / p))

    #######################################################################################################
    # Step 1:                                                                                             #
    # Estimate the Prediction Matrix based on the D-period backwards-looking window                       #
    # And calculate the symmetric part of the Prediction Matrix                                           #
    #######################################################################################################

    # Uses the pre-defined Prediction Matrix function
    prediction_matrix = estimated_prediction_matrix(returns, signals, D=D, eff=eff)

    # Calculate the symmetric part
    symmetric_matrix = (1/2) * (prediction_matrix + prediction_matrix.T)

    del prediction_matrix, returns, signals
    gc.collect()

    #######################################################################################################
    # Step 2:                                                                                             #
    # Construct the (K) individual PEPs using Eigenvalue decomposition of the symmetric Prediction Matrix #
    #######################################################################################################

    # Uses the Numpy linear algebra function for eigenvalue decomposition:
    E, V = np.linalg.eig(symmetric_matrix)

    del symmetric_matrix
    gc.collect()

    # The number of eigenvalues used
    if K == -1:
        K = len(E)

    ################################################################################    
    # Restrict Matrices to only include the K largest absolute Eigenvalues         #
    ################################################################################

    # First we create a dataframe for the eigenvalues to be able to use indexes which can also track the corresponding eigenvectors
    df = pd.DataFrame(E, columns= ['Eigenvalues'])
    df = df.reindex(df['Eigenvalues'].sort_values(ascending=False).index) # Sort by largest absolute value

    df = df.iloc[:K, :] # Choose the K largest absolute Eigenvalues

    # Next we sort the corresponding eigenvectors by the sorted index of df_eigen
    V_sort = []
    E_sort = []
    for n in df.index:
         V_sort.append(V[:, n]) # Keep in mind that when we append like this the columns becomes the rows, which is why we transpose further down
         E_sort.append(E[n])
    V_sort = np.asarray(V_sort).T # Transpose
    E_sort = np.asarray(E_sort)

    del V, E
    gc.collect()

    E_sort = E_sort.reshape(E_sort.shape[0], -1)
    
    ################################################################################

    # Calculating position matrix of the individual PEPs
    L_k = np.einsum('il,jl-> lij', V_sort, V_sort)

    del V_sort
    gc.collect()

    #######################################################################################################    
    # Step 3:                                                                                             #
    # Calculate the optimal position matrix                                                               #
    #######################################################################################################

    # Calculate c
    df['Eigen_c'] = df['Eigenvalues'].abs() ** q # Should we also take the absolute value of the eigenvalue in the calculation of c? This is not completely clear from the paper so we need to test this
    c = df['Eigen_c'].sum() ** (-1 / p)    

    # Calculate the optimal position matrix
    L = c * np.einsum('lij, lp -> lij', L_k, np.sign(E_sort)*abs(E_sort)**(q-1)).sum(axis=0)

    return (L, E_sort, L_k)



#########################################################################
# PRINCIPAL ALPHA PORTFOLIO                                           
#########################################################################

def principal_alpha_portfolio(returns, signals, K=-1, p=2, D=120, eff=False):
    """
    This function calculates the (K) individual Principal Alpha Portfolios (PAPs) alongside the optimal weights based on the individual PAPs, 
    given the asset returns, signals, and the two tunning parameters (K, p)

    Parameters:
        returns (numpy array): Historical asset returns (shape: (T, N))
        signals (numpy array): Signals for each asset (shape: (T, N))
        K (int): Number of largest complex eigenvalues used (K= -1: uses all eigenvalues). For the PAP it can max be N/2.
        p (float): Tunning parameter for the schatten matrix norm used for the calculations (values: [1:∞]). Default=2: Frobenius-norm
        D (int): Backward-looking periods to estimate the Prediction Matrix from
        eff (True/False): Efficiency parameter for the calculation of the Prediction matrix. If True the code is more efficient for large N and D but can give slightly less accurate results

    Returns:
        tuple: A tuple containing:
            - L (numpy.ndarray): The optimal position matrix (shape: (N, N))
            - E (numpy.ndarray): The K largest complex Eigenvalues (shape: (K))
            - L_k (numpy.ndarray): Array of K matrices of each PAPs positon matrix ordered by the the complex eigenvalues (shape: (K, N, N))
    """

    # Sets q based on the p chosen
    if p == 1:
        q = 0
    else:
        q = 1 / (1 - (1 / p))

    #######################################################################################################
    # Step 1:                                                                                             #
    # Estimate the Prediction Matrix based on the D-period backwards-looking window                       #
    # And calculate the antisymmetric part of the Prediction Matrix                                       #
    #######################################################################################################

    # Uses the pre-defined Prediction Matrix function
    prediction_matrix = estimated_prediction_matrix(returns, signals, D=D, eff=eff)

    # Calculate the antisymmetric part
    antisymmetric_matrix = (1/2) *(prediction_matrix - prediction_matrix.T)

    del prediction_matrix, returns, signals
    gc.collect()

    #######################################################################################################
    # Step 2:                                                                                             #
    # Construct the (K) individual PAPs using Eigenvalue decomposition of the symmetric Prediction Matrix #
    #######################################################################################################

    # Uses the Numpy linear algebra function for eigenvalue decomposition:
    E, V = np.linalg.eig(antisymmetric_matrix)

    del antisymmetric_matrix
    gc.collect()

    # The number of eigenvalues used
    if K == -1:
        K = int(len(E)/2)

    ################################################################################    
    # Restrict Matrices to only include the K largest complex Eigenvalues          #
    ################################################################################

    # First we create a dataframe for the eigenvalues to be able to use indexes which can also track the corresponding eigenvectors
    df = pd.DataFrame(E, columns= ['Eigenvalues'])
    df['imag'] = np.imag(df['Eigenvalues'])
    df_all = df.reindex(df['imag'].sort_values(ascending=False).index) # Sort by largest imaginary part og the eigenvalues 

    df_top = df_all.iloc[: K, :] # Choose the K largest imaginary Eigenvalues 
    df_bottom = df_all.iloc[ -(K):, :] # Choose the K smallest imaginary Eigenvalues

    df = pd.concat([df_top, df_bottom]) # Collect the top and bottom eigenvalues

    # Next we sort the corresponding eigenvectors by the sorted index of df_all
    V_sort = []
    E_sort = []
    for n in df.index:
            V_sort.append(V[:, n]) # Keep in mind that when we append like this the columns becomes the rows, which is why we transpose further down
            E_sort.append(E[n])
    V_sort = np.asarray(V_sort).T # Transpose
    E_sort = np.asarray(E_sort)

    # Separating the real and imaginary part of the vectors for the K largest eigenvalues
    V_imag = np.imag(V_sort[:, :K])
    V_real = np.real(V_sort[:, :K])

    del V, E
    gc.collect()

    E_sort = E_sort.reshape(E_sort.shape[0], -1)
    
    ################################################################################

    # Method one:
    #L_k = np.einsum('il,jl-> lij', V_real, V_imag) - np.einsum('il,jl-> lij', V_imag, V_real)

    del V_real, V_imag
    gc.collect()

    # Method two:
    V_flip = np.flip(V_sort, axis=1)
    l_1 = np.einsum('il,jl-> lij', V_sort, V_flip)
    l_2 = np.einsum('il,jl-> lij', V_flip, V_sort)
    L_k = l_1 - l_2

    del V_flip, l_1, l_2
    gc.collect()

    #######################################################################################################    
    # Step 3:                                                                                             #
    # Calculate the optimal position matrix                                                               #
    #######################################################################################################

    # Calculate c
    df['Eigen_c'] = df['Eigenvalues'] ** q 
    c = df['Eigen_c'].sum() ** (-1 / p)    

    # Calculate the optimal position matrix
    L = c * np.einsum('lij, lp -> lij', L_k, E_sort[:, :]**(q-1)).sum(axis=0)

    return (L, E_sort, L_k)
