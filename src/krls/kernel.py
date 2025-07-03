# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 17:17:41 2025

@author: Kaike Sa Teles Rocha Alves
@email_1: kaikerochaalves@outlook.com
@email_2: kaike.alves@estudante.ufjf.br

As a Data Science Manager at PGE-PR and a Ph.D. in Computational Modeling
at the Federal University of Juiz de Fora (UFJF), I specialize in artificial
intelligence, focusing on the study, development, and application of machine learning
models, under the supervision of Prof. Eduardo Pestana de Aguiar. My academic journey 
includes a scholarship that allowed me to pursue a year of my Ph.D. at the 
University of Nottingham/UK, where I was a member of the LUCID
(Laboratory for Uncertainty in Data and Decision Making) under the 
supervision of Professor Christian Wagner. My background in Industrial
Engineering provides me with a holistic view of organizational processes,
enabling me to propose efficient solutions and reduce waste.
"""

# Import libraries

import numpy as np

class Kernel:
    
    def __init__(self, kernel_type="Gaussian", validate_array = False, **kwargs):
        
        """
        Kernel class for computing various kernel functions.
        
        Parameters
        ----------
        kernel_type : str, optional (default="Gaussian")
            Specifies the type of kernel function to use. Must be one of:
            "Linear", "Polynomial", "RBF", "Gaussian", "Sigmoid", "Powered", "Log", 
            "GeneralizedGaussian", "Hybrid".
            
            The type of kernel function to use. Available options are:
            - "Linear" (no hyperparameters required)
            - "Polynomial" (requires 'a', 'b', and 'd')
            - "RBF" (requires 'sigma')
            - "Gaussian" (requires 'sigma')
            - "Sigmoid" (requires 'sigma' and 'r')
            - "Powered" (requires 'beta')
            - "Log" (requires 'beta')
            - "GeneralizedGaussian" (requires 'A', a symmetric positive definite matrix)
            - "Hybrid" (requires 'sigma', 'tau', and 'd')
            - "additive_chi2" (no hyperparameters required)
            - "Cosine" (no hyperparameters required)
        
        validate_array : bool, optional (default=False)
            If True, validates input arrays to ensure they are numeric, finite, and properly formatted.
        
        **kwargs : dict, optional
            Additional parameters specific to the chosen kernel function.
        
        Raises
        ------
        ValueError
            If an invalid kernel type is provided.
        
        Examples
        --------
        >>> kernel = Kernel(kernel_type="Polynomial", a=2, b=1, d=3)
        >>> X1, X2 = np.array([1, 2, 3]), np.array([4, 5, 6])
        >>> kernel.compute(X1, X2)
        """
        
        # Define valid kernel types and their required parameters
        self.valid_kernels = {
            "Linear": self.Linear,
            "Polynomial": self.Polynomial,
            "RBF": self.RBF,
            "Gaussian": self.Gaussian,
            "Sigmoid": self.Sigmoid,
            "Powered": self.Powered,
            "Log": self.Log,
            "GeneralizedGaussian": self.GeneralizedGaussian,
            "Hybrid": self.Hybrid,
            "additive_chi2": self.additive_chi2,
            "Cosine": self.Cosine
        }
        
        # Check if the kernel type is valid
        if kernel_type not in self.valid_kernels:
            raise ValueError(f"Invalid kernel type: {kernel_type}. Choose from {list(self.valid_kernels.keys())}.")
        
        # Validate the array
        self.validate_array = validate_array
        # Store kernel type and parameters
        self.kernel_type = kernel_type
        self.params = kwargs
        
    def is_numeric_and_finite(self, array):
        return np.isfinite(array).all() and np.issubdtype(np.array(array).dtype, np.number) and isinstance(array, np.ndarray)
        
    def x_format(self, x1, x2):
        
        # Check if is numeric, finit, and array
        if not self.is_numeric_and_finite(x1) or not self.is_numeric_and_finite(x2):
            raise ValueError(
                "X contains incompatible values."
                " Check X for non-numeric or infinity values"
                )
            
        # Check the number of dimensions
        if len(x1.shape) == 1 and len(x2.shape) == 1:
            # Check the dimension
            if x1.shape[0] == x2.shape[0]:
                return x1, x2
            else:
                raise ValueError(
                    "X1 and X2 contains incompatible shape."
                    )
        
        elif len(x1.shape) == 2 and len(x2.shape) == 2:
            # Check if one of the dimensions is 1
            if x1.shape[0] == 1 or x1.shape[1] == 1:
                x1 = x1.ravel()
            else:
                raise ValueError(
                    "X1 and X2 contains incompatible shape."
                    )
            
            # Check if one of the dimensions is 1
            if x2.shape[0] == 1 or x2.shape[1] == 1:
                x2 = x2.ravel()
            else:
                raise ValueError(
                    "X1 and X2 contains incompatible shape."
                    )
                
            # Check the dimension
            if x1.shape[0] == x2.shape[0]:
                return x1, x2
            else:
                raise ValueError(
                    "X1 and X2 contains incompatible shape."
                    )
                
        else:
            raise ValueError(
                "X1 or X2 are multidimensional."
                )
    
    def is_symmetric(self, A, tol=1e-8):
        return np.allclose(A, A.T, atol=tol)
    
    def is_positive_definite(self, A):
        try:
            np.linalg.cholesky(A)
            return True
        except np.linalg.LinAlgError:
            return False
    
    def is_symmetric_positive_definite(self, A, tol=1e-8):
        return self.is_symmetric(A, tol) and self.is_positive_definite(A)
    
    def compute(self, X1, X2, **kwargs):
        
        if self.validate_array:
            # Validate the format of X1 and X2
            X1, X2 = self.x_format(X1, X2)
        
        # Combine instance parameters with additional arguments
        params = {**self.params, **kwargs}

        
        """Compute the chosen kernel function dynamically."""
        return self.valid_kernels[self.kernel_type](X1, X2, **params)
    
    # Kernel function definitions:
    
    @staticmethod
    def Linear(X1, X2, **kwargs):
        return np.sum(X1*X2)

    @staticmethod
    def Polynomial(X1, X2, a=1, b=1, d=2, **kwargs):
        return (a * np.sum(X1*X2) + b) ** d

    @staticmethod
    def RBF(X1, X2, gamma=1.0, **kwargs):
        return np.exp(- gamma * np.sum((X1 - X2) ** 2))
    
    @staticmethod
    def Gaussian(X1, X2, sigma=1.0, **kwargs):
        return np.exp(-np.sum((X1 - X2) ** 2) / (2 * sigma**2))

    @staticmethod
    def Sigmoid(X1, X2, sigma=1.0, r=0, **kwargs):
        return np.tanh(sigma * np.sum(X1*X2) + r)

    @staticmethod
    def Powered(X1, X2, beta=1, **kwargs):
        return np.exp( - ((np.sum((X1 - X2) ** 2))**(beta/2)) )

    @staticmethod
    def Log(X1, X2, beta=1, **kwargs):
        return - np.log( 1 + ((np.sum((X1 - X2) ** 2))**(beta/2)) )

    def GeneralizedGaussian(self, X1, X2, A=None, **kwargs):
        
        if self.validate_array:
        
            if A is None:
                raise ValueError(
                    "A SPD matrix was not informed."
                    )
                
            if not self.is_symmetric_positive_definite(A):
                
                raise ValueError(
                    "The matrix A is not SPD."
                    )
                
            if X1.shape[0] != A.shape[0] or X2.shape[0] != A.shape[0]:
                raise ValueError(
                    "The vectors are not compatible with the shape of the matrix A."
                    )
            
        return np.exp( - ((X1-X2).reshape(1,-1) @ A @ (X1-X2).reshape(-1,1)).item() )

    @staticmethod
    def Hybrid(X1, X2, sigma=1.0, tau=1.0, d=2, **kwargs):
        return ( np.exp( - np.sum((X1 - X2) ** 2) / (sigma**2))) * (tau + np.sum(X1*X2)) ** d
    
    @staticmethod
    def additive_chi2(X1, X2):
        return np.sum((2 * X1 * X2) / (X1 + X2 + 1e-9))  # Adding epsilon to avoid division by zero
    
    @staticmethod
    def Cosine(X1, X2):
        return np.sum((X1 * X2) / (((np.sum((X1) ** 2))**(1/2)) * ((np.sum((X2) ** 2))**(1/2)) + 1e-9))  # Adding epsilon to avoid division by zero
