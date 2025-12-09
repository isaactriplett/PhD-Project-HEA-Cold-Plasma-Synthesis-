# -*- coding: utf-8 -*-
"""
Created on Sat Nov 15 06:29:48 2025

@author: isaac
"""

import numpy as np
from scipy.stats import norm

def hodges_lehmann_estimator(x):
    """
    Compute the Hodges-Lehmann estimator, which is the median of all Walsh averages (x_i + x_j)/2 for i <= j.
    """
    x = np.sort(x)
    n = len(x)
    walsh_averages = []
    for i in range(n):
        for j in range(i, n):
            walsh_averages.append((x[i] + x[j]) / 2.0)
    return np.median(walsh_averages)

def legendre_p3(u):
    """
    The third Legendre polynomial score function: p3(u) = (sqrt(7)/2) * (6u^2 - 6u + 1)
    """
    return (np.sqrt(7) / 2) * (6 * u**2 - 6 * u + 1)

def symmetry_test(x):
    """
    Compute the p-value for the test of symmetry about an unknown median using the linear rank procedure
    with Hodges-Lehmann estimator and Legendre polynomial score functions.
    
    Parameters:
    x (array-like): The input data array.
    
    Returns:
    float: The p-value for the two-sided test.
    """
    x = np.array(x)
    n = len(x)
    if n < 2:
        raise ValueError("Sample size must be at least 2.")
    
    # Step 1: Compute Hodges-Lehmann estimator for theta
    theta_hat = hodges_lehmann_estimator(x)
    
    # Step 2: Compute absolute deviations
    abs_dev = np.abs(x - theta_hat)
    
    # Step 3: Compute ranks of absolute deviations (1 to n, assuming no ties)
    # Use argsort twice to get ranks starting from 1
    ranks = np.argsort(np.argsort(abs_dev)) + 1
    
    # Step 4: Compute scores using p3
    u = (ranks / (2*(n + 1.0)))+0.5
    scores = legendre_p3(u)
    
    # Step 5: Compute signs
    signs = np.sign(x - theta_hat)
    # If any exactly zero, sign is zero, but unlikely for continuous data
    
    # Step 6: Compute the statistic a
    a = np.sum(scores * signs) / n
    
    # Step 7: Compute finite sample variance V_p,n under known median
    k = np.arange(1, n + 1)
    uk = (k / (2*(n + 1.0)))+0.5
    v_pn = (1.0 / n) * np.sum(legendre_p3(uk)**2)
    
    # Step 8: For Legendre, d_p = 0.000
    d_p = 0.000
    sigma2 = v_pn + d_p
    sigma = np.sqrt(sigma2)
    
    # Step 9: Compute test statistic R
    if sigma == 0:
        raise ValueError("Computed sigma is zero.")
    R = a / sigma
    
    # Step 10: Compute two-sided p-value using standard normal
    p_value = 2 * (1 - norm.cdf(np.abs(R)))
    
    #return p_value
    #return a
    print("p value:"+str(p_value)+"a:"+str(a))
    #print("a:"+a)

# Example usage (commented out):
# if __name__ == "__main__":
#     # Sample data
#     data = np.random.normal(0, 1, 30)
#     p_val = symmetry_test(data)
#     print(f"P-value: {p_val}")

#create sample array with the same value as was dealt with in R
Sample = [-1.2,-0.8,-0.2,0,0.2,0.5,1.1,1.4,2]

symmetry_test(Sample)