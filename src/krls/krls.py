# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 15:08:53 2025

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

# Importing libraries
import torch
import numpy as np
from .kernel import Kernel

class base():
    
    def __init__(self, kernel_type, validate_array, **kwargs):
        
        """
        Base class for kernel-based learning models.
        
        Parameters
        ----------
        kernel_type : str
            The type of kernel function to use. Must be one of: 'Linear', 'Polynomial', 'RBF', 'Gaussian',
            'Sigmoid', 'Powered', 'Log', 'GeneralizedGaussian', 'Hybrid', additive_chi2, and Cosine.
        
        validate_array : bool
            If True, input arrays are validated before computation.
        
        **kwargs : dict
            Additional hyperparameters depending on the chosen kernel:
            - 'a', 'b', 'd' : Polynomial kernel parameters
            - 'gamma': RBF kernel parameter
            - 'sigma' : Gaussian, and Hybrid kernel parameter
            - 'r' : Sigmoid kernel parameter
            - 'beta' : Powered and Log kernel parameter
            - 'tau' : Hybrid kernel parameter
            - 'lr' : GeneralizedGaussian kernel parameters
            - 'epochs' : GeneralizedGaussian kernel parameters
        """
        
        # List of predefined valid parameters
        self.valid_params = ['kernel_type', 'validate_array', 'a', 'b', 'd', 'gamma', 'sigma', 'r', 'beta', 'tau', 'lr', 'epochs']  # Adjust this list as needed

        # Initialize the dictionary
        self.hyperparameters_dict = {"kernel_type": kernel_type, "validate_array": validate_array}
        
        # Default values for parameters
        self.default_values = {
            'a': 1,
            'b': 1,
            'd': 2, 
            'gamma': 1,
            'sigma': 10,
            'r': 0,
            'beta': 1.0,
            'tau': 1.0, 
            'lr': 0.01, 
            'epochs': 100, 
        }

        # Check if any parameters are in kwargs and are valid
        for key, value in kwargs.items():
            if key in self.valid_params:
                # Check if the value is valid (example: must be a positive float)
                if not self.is_valid_param(key, value):
                    raise ValueError(f"Invalid value for parameter '{key}': {value}")
                self.hyperparameters_dict[key] = value
            else:
                print(f"Warning: '{key}' is not a valid parameter.")

        # Set default values for parameters that were not passed in kwargs
        for param, default_value in self.default_values.items():
            if param not in self.hyperparameters_dict:
                self.hyperparameters_dict[param] = default_value
        
        # Filter correct hyperparameters
        if kernel_type in ["Gaussian"]:
            keys = ['kernel_type', 'validate_array', 'sigma']
            self.kwargs = {key: self.hyperparameters_dict.get(key, None) for key in keys}
        if kernel_type in ["RBF"]:
            keys = ['kernel_type', 'validate_array', 'gamma']
            self.kwargs = {key: self.hyperparameters_dict.get(key, None) for key in keys}
        elif kernel_type in ["Linear", "additive_chi2", "Cosine"]:
            keys = ['kernel_type', 'validate_array']
            self.kwargs = {key: self.hyperparameters_dict.get(key, None) for key in keys}
        elif kernel_type == "Polynomial":
            keys = ['kernel_type', 'validate_array', 'a', 'b', 'd']
            self.kwargs = {key: self.hyperparameters_dict.get(key, None) for key in keys}
        elif kernel_type in ["Powered","Log"]:
            keys = ['kernel_type', 'validate_array', 'beta']
            self.kwargs = {key: self.hyperparameters_dict.get(key, None) for key in keys}
        elif kernel_type == "Hybrid":
            keys = ['kernel_type', 'validate_array', 'sigma', 'tau', 'd']
            self.kwargs = {key: self.hyperparameters_dict.get(key, None) for key in keys}
        elif kernel_type == "Sigmoid":
            keys = ['kernel_type', 'validate_array', 'sigma', 'r']
            self.kwargs = {key: self.hyperparameters_dict.get(key, None) for key in keys}
        elif kernel_type == "GeneralizedGaussian":
            keys = ['kernel_type', 'validate_array']
            self.kwargs = {key: self.hyperparameters_dict.get(key, None) for key in keys}
            self.lr = self.hyperparameters_dict.get('lr')
            self.epochs = self.hyperparameters_dict.get('epochs')
        
        # Initialize the kernel
        self.kernel = Kernel(**self.kwargs)
        
        # Initialize the dictionary
        self.parameters_dict = {}
        # Computing the output in the training phase
        self.y_pred_training = None
        # Computing the residual square in the ttraining phase
        self.ResidualTrainingPhase = None
        # Computing the output in the testing phase
        self.y_pred_test = None
        
    def is_numeric_and_finite(self, array):
        return np.isfinite(array).all() and np.issubdtype(np.array(array).dtype, np.number)
    
    def output_training(self):
        return self.y_pred_training
    
    def is_valid_param(self, param, value):
        """Define validation rules for parameters here."""
        # Example validation rule: Ensure positive float for 'param1' and 'param2'
        if param in ['a', 'b', 'd', 'r'] and not isinstance(value, (int, float)):
            return False
        if param in ['gamma', 'sigma'] and (not isinstance(value, (int, float)) and value <= 0):
            return False
        if param == 'beta' and ((not isinstance(value, (int, float)) or not (0 < value <= 1))):
            return False
        if param == 'lr' and ((not isinstance(value, (int, float)) or (value <= 0))):
            return False
        if param == 'epochs' and ((not isinstance(value, (int)) or (value < 1))):
            return False
        return True
    
    def fit(self, X, y):
        
        # Shape of X and y
        X_shape = X.shape
        y_shape = y.shape
        
        # Correct format X to 2d
        if len(X_shape) == 1:
            X = X.reshape(-1,1)
        
        # Check wheather y is 1d
        if len(y_shape) > 1 and y_shape[1] > 1:
            raise TypeError(
                "This algorithm does not support multiple outputs. "
                "Please, give only single outputs instead."
            )
        
        if len(y_shape) > 1:
            y = y.ravel()
        
        # Check wheather y is 1d
        if X_shape[0] != y_shape[0]:
            raise TypeError(
                "The number of samples of X are not compatible with the number of samples in y. "
            )
            
        # Check if the inputs contain valid numbers
        if not self.is_numeric_and_finite(X):
            raise ValueError(
                "X contains incompatible values."
                " Check X for non-numeric or infinity values"
            )
            
        # Check if the inputs contain valid numbers
        if not self.is_numeric_and_finite(y):
            raise ValueError(
                "y contains incompatible values."
                " Check y for non-numeric or infinity values"
            )
            
        # Preallocate space for the outputs for better performance
        self.y_pred_training = np.zeros((y_shape))
        self.ResidualTrainingPhase = np.zeros((y_shape))
                
        # Initialize the first input-output pair
        x0 = X[0,].reshape(-1,1)
        y0 = y[0]
        
        # Initialize KRLS
        self.Initialize(x0, y0)
        
        # Compute the SPD matrix
        if self.kernel_type == "GeneralizedGaussian":
            self.A = self.learn_A(X, y)

        for k in range(1, X.shape[0]):

            # Prepare the k-th input vector
            x = X[k,].reshape((1,-1)).T
            
            # Update KRLS
            k_til = self.KRLS(x, y[k])
            
            # Compute output
            Output = np.dot(self.parameters_dict["Theta"], k_til)
            
            # Store results
            self.y_pred_training = np.append(self.y_pred_training, Output )
            self.ResidualTrainingPhase = np.append(self.ResidualTrainingPhase,(y[k]) - Output )
            
    def evolve(self, X, y):
        
        # Be sure that X is with a correct shape
        X = X.reshape(-1,self.parameters_dict["Dict"].shape[0])
        
        # Check the format of y
        if not isinstance(y, (np.ndarray)):
            y = np.array(y, ndmin=1)
            
        # Correct format X to 2d
        if len(X.shape) == 1:
            X = X.reshape(-1,1)
        
        # Check wheather y is 1d
        if len(y.shape) > 1 and y.shape[1] > 1:
            raise TypeError(
                "This algorithm does not support multiple outputs. "
                "Please, give only single outputs instead."
            )
        
        if len(y.shape) > 1:
            y = y.ravel()
        
        # Check wheather y is 1d
        if X.shape[0] != y.shape[0]:
            raise TypeError(
                "The number of samples of X are not compatible with the number of samples in y. "
            )
            
        # Check if the inputs contain valid numbers
        if not self.is_numeric_and_finite(X):
            raise ValueError(
                "X contains incompatible values."
                " Check X for non-numeric or infinity values"
            )
            
        # Check if the inputs contain valid numbers
        if not self.is_numeric_and_finite(y):
            raise ValueError(
                "y contains incompatible values."
                " Check y for non-numeric or infinity values"
            )
            
        # # Update the SPD matrix
        # if self.kernel_type == "GeneralizedGaussian":
        #     self.A = self.learn_A(X, y, self.A)
        
        for k in range(X.shape[0]):

            # Prepare the k-th input vector
            x = X[k,].reshape((1,-1)).T
            
            # # If the kernel type is the GeneralizedGaussian, update the SPD matrix
            # if self.kernel_type == "GeneralizedGaussian":
            #     self.A -= ((self.A @ x @ x.T @ self.A) / (1 + x.T @ self.A @ x))
                      
            # Update KRLS
            k_til = self.KRLS(x, y[k])
            
            # Compute output
            Output = np.dot(self.parameters_dict["Theta"], k_til)
            
            # Store the prediction
            self.y_pred_training = np.append(self.y_pred_training, Output)
            # Compute the error
            residual = abs(Output - y[k])
            self.ResidualTrainingPhase = np.append(self.ResidualTrainingPhase, residual**2)
            
    def predict(self, X):
        
        # Correct format X to 2d
        if len(X.shape) == 1:
            X = X.reshape(-1,1)
            
        # Check if the inputs contain valid numbers
        if not self.is_numeric_and_finite(X):
            raise ValueError(
                "X contains incompatible values."
                " Check X for non-numeric or infinity values"
            )
            
        # Be sure that X is with a correct shape
        X = X.reshape(-1,self.parameters_dict["Dict"].shape[0])
        
        # Preallocate space for the outputs for better performance
        self.y_pred_test = np.zeros((X.shape[0]))

        for k in range(X.shape[0]):
            
            # Prepare the first input vector
            x = X[k,].reshape((1,-1)).T

            # Compute k_til
            n_cols = self.parameters_dict["Dict"].shape[1]
            # If the kernel type is the GeneralizedGaussian, inform matrix SPD matrix
            if self.kernel_type == "GeneralizedGaussian":
                k_til = np.array([self.kernel.compute(self.parameters_dict["Dict"][:, ni].reshape(-1, 1), x, A = self.A) for ni in range(n_cols)])
            else:
                k_til = np.array([self.kernel.compute(self.parameters_dict["Dict"][:, ni].reshape(-1, 1), x) for ni in range(n_cols)])
                
            # Compute the output
            Output = np.dot(self.parameters_dict["Theta"], k_til)
            
            # Store the results
            self.y_pred_test[k,] = Output
        
        # Return the prediction
        return self.y_pred_test
    
    def compute_loss(self, X, y, A):
        """Compute the mean squared error loss."""
        predictions = X @ A  # Matrix multiplication to get predictions
        loss = torch.mean((predictions - y) ** 2)  # Mean squared error
        return loss
    
    def learn_A(self, X, y, A = None):
        
        """Learn A using gradient-based optimization."""
        
        # Ensure X and y are torch tensors
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        
        # Ensure y is a column vector
        if y.ndimension() == 1:  # if y is a 1D tensor (n_samples,)
            y = y.unsqueeze(1)  # Reshape y to (n_samples, 1)
        
        # Initialize A if not provided
        if A is None:
            
            # Create A
            d = X.shape[1]
            A = torch.eye(d, requires_grad=True)  # Initialize A as an identity matrix
            
            # Define the optimizer
            optimizer = torch.optim.Adam([A], lr=self.lr)
            
            # Train to get A
            for _ in range(self.epochs):
                loss = self.compute_loss(X, y, A)  # Call the loss function here
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
        else:
            
            # Prepare A
            A = torch.tensor(A, dtype=torch.float32, requires_grad=True)
            
            # Define the optimizer
            optimizer = torch.optim.Adam([A], lr=self.lr)
            
            # Train to get A
            loss = self.compute_loss(X, y, A)  # Call the loss function here
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Return A as a numpy array
        return A.detach().numpy()

class KRLS(base):
    
    def __init__(self, nu = 0.01, N = 100, validate_array = False, kernel_type = 'Linear', **kwargs):
        
        """
        Kernel Recursive Least Squares (KRLS) model.
        
        Parameters
        ----------
        nu : float, default=0.01
            Accuracy parameter determining the level of sparsity. Must be a positive float.
            
        N : int, default=100
            Accuracy parameter determining the level of sparsity. Must be a integer greater 
            than 1.
        
        kernel_type : str, default='Linear'
            The type of kernel function to use. Must be one of the supported kernels in 
            `base`.
        
        validate_array : bool, default=False
            If True, input arrays are validated before computation. It is more time consuming. 
        
        **kwargs : dict
            Additional kernel-specific hyperparameters passed to the `base` class.
        """
        
        # Call __init__ of the base class
        super().__init__(kernel_type, validate_array, **kwargs)
        
        if not (nu > 0):
            raise ValueError("nu must be a positive float.")
            
        if (not (N > 1)) and (not(isinstance(N, (int)))):
            raise ValueError("N must be a positive float.")
        
        # Hyperparameters
        # nu is an accuracy parameter determining the level of sparsity
        self.nu = nu
        # Maximum number of vectors in the dictionary
        self.N = N
        # Validate array
        self.validate_array = validate_array
        # Kernel type
        self.kernel_type = kernel_type
        
    def get_params(self, deep=True):
        return {
            'nu': self.nu,
            'N': self.N,
            'validate_array': self.validate_array,
            'kernel_type': self.kernel_type,
            **self.kwargs  # Merge self.kwargs into the dictionary
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self
    
    def Initialize(self, x, y):
        
        # If the kernel type is the GeneralizedGaussian, initialize the SPD matrix
        if self.kernel_type == "GeneralizedGaussian":
            self.A = np.eye(x.shape[0])
        
        # Compute the variables for the dictionary
        # Check if the kernel type is the Generalized Gaussian
        if self.kernel_type == "GeneralizedGaussian":
            k11 = self.kernel.compute(x, x, A = self.A)
        else:
            k11 = self.kernel.compute(x, x)
        
        # Update Kinv and Theta
        Kinv = np.ones((1,1)) / ( k11 ) if k11 != 0 else np.ones((1,1))
        Theta = np.ones((1,)) * y / k11 if k11 != 0 else np.ones((1,))
        
        # Fill the dictionary
        self.parameters_dict.update({"Kinv": Kinv, "Theta": Theta, "P": np.ones((1,1)), "m": 1., "Dict": x})
        
        # Initialize first output and residual
        self.y_pred_training = np.append(self.y_pred_training, y)
        self.ResidualTrainingPhase = np.append(self.ResidualTrainingPhase, 0.)
        
    def KRLS(self, x, y):
                        
        # Compute k_til
        n_cols = self.parameters_dict["Dict"].shape[1]
        # If the kernel type is the GeneralizedGaussian, inform matrix SPD matrix
        if self.kernel_type == "GeneralizedGaussian":
            k_til = np.array([self.kernel.compute(self.parameters_dict["Dict"][:, ni].reshape(-1, 1), x, A = self.A) for ni in range(n_cols)])
        else:
            k_til = np.array([self.kernel.compute(self.parameters_dict["Dict"][:, ni].reshape(-1, 1), x) for ni in range(n_cols)])
        
        # Compute a
        a = np.matmul(self.parameters_dict["Kinv"], k_til)
        
        # Compute delta
        # Check if the kernel type is the Generalized Gaussian
        if self.kernel_type == "GeneralizedGaussian":
            delta = self.kernel.compute(x, x, A = self.A) - np.dot(k_til, a)
        else:
            delta = self.kernel.compute(x, x) - np.dot(k_til, a)
        
        # Avoid zero division
        if delta == 0:
            delta = 1.
            
        # Compute the residual
        EstimatedError = y - np.dot(k_til, self.parameters_dict["Theta"]) 
        
        # Novelty criterion
        if delta > self.nu and n_cols < self.N:
            
            # Update Dict in-place
            self.parameters_dict["Dict"] = np.hstack([self.parameters_dict["Dict"], x])
            self.parameters_dict["m"] += 1
            
            # Update Kinv                      
            self.parameters_dict["Kinv"] = (1/delta)*(self.parameters_dict["Kinv"] * delta + np.outer(a, a))
            self.parameters_dict["Kinv"] = np.pad(self.parameters_dict["Kinv"], ((0, 1), (0, 1)), mode='constant')
            self.parameters_dict["Kinv"][-1, -1] = 1/delta
            self.parameters_dict["Kinv"][:-1, -1] = self.parameters_dict["Kinv"][-1, :-1] = (1/delta) * (-a)
            
            # Update P similarly
            self.parameters_dict["P"] = np.pad(self.parameters_dict["P"], ((0, 1), (0, 1)), mode='constant')
            self.parameters_dict["P"][-1, -1] = 1.
                        
            # Updating Theta
            self.parameters_dict["Theta"] -= (a / delta) * EstimatedError
            self.parameters_dict["Theta"] = np.append(self.parameters_dict["Theta"], ( 1 / delta ) * EstimatedError )
            
            # Update k_til
            if self.kernel_type == "GeneralizedGaussian":
                k_til = np.append(k_til, self.kernel.compute(x, x, A = self.A))
            else:
                k_til = np.append(k_til, self.kernel.compute(x, x))
        
        else:
            
            # Precompute terms at once
            A_P = np.dot(self.parameters_dict["P"], a)
            A_P_A = np.dot( A_P, a )
            
            # Compute q more efficiently
            q = A_P / (1 + A_P_A)
            
            # Update P
            self.parameters_dict["P"] -= (np.matmul(np.outer(A_P, a), self.parameters_dict["P"])) / (1 + A_P_A)
            
            # Update Theta
            self.parameters_dict["Theta"] += np.dot(self.parameters_dict["Kinv"], q) * EstimatedError
        
        # Return the kernel vector
        return k_til

    def compute_loss(self, X, y, A):
        """Compute the mean squared error loss."""
        predictions = X @ A  # Matrix multiplication to get predictions
        loss = torch.mean((predictions - y) ** 2)  # Mean squared error
        return loss
    
    def learn_A(self, X, y, A = None):
        
        """Learn A using gradient-based optimization."""
        
        # Ensure X and y are torch tensors
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        
        # Ensure y is a column vector
        if y.ndimension() == 1:  # if y is a 1D tensor (n_samples,)
            y = y.unsqueeze(1)  # Reshape y to (n_samples, 1)
        
        # Initialize A if not provided
        if A is None:
            
            # Create A
            d = X.shape[1]
            A = torch.eye(d, requires_grad=True)  # Initialize A as an identity matrix
            
            # Define the optimizer
            optimizer = torch.optim.Adam([A], lr=self.lr)
            
            # Train to get A
            for _ in range(self.epochs):
                loss = self.compute_loss(X, y, A)  # Call the loss function here
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
        else:
            
            # Prepare A
            A = torch.tensor(A, dtype=torch.float32, requires_grad=True)
            
            # Define the optimizer
            optimizer = torch.optim.Adam([A], lr=self.lr)
            
            # Train to get A
            loss = self.compute_loss(X, y, A)  # Call the loss function here
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        return A.detach().numpy()  # Return A as a numpy array


class SW_KRLS(base):
    
    def __init__(self, N = 100, c = 1e-6, validate_array = False, kernel_type = 'Linear', **kwargs):
        
        """
        Sliding Window Kernel Recursive Least Squares (SW-KRLS) model.
        
        Parameters
        ----------
        N : int, default=100
            Accuracy parameter determining the level of sparsity. 
            Must be an integer greater than 1.
        
        c : float, default=1e-6
            Regularization parameter. Must be a very small float.
        
        kernel_type : str, default='Gaussian'
            The type of kernel function to use. Must be one of the supported kernels
            in `base`.
        
        validate_array : bool, default=False
            If True, input arrays are validated before computation.
        
        **kwargs : dict
            Additional kernel-specific hyperparameters passed to the `base` class.
        """
        
        # Call __init__ of the base class
        super().__init__(kernel_type, validate_array, **kwargs)
        
        if (not (N > 0)) or (not(isinstance(N,int))):
            raise ValueError("N must be a positive integer.")
        
        if (not (c > 0 and c < 1)) or (not isinstance(c, float)):
            raise ValueError("c must be a float between 0 and 1.")
         
        # Hyperparameters
        # N: maximum number of elements in the dictionary
        self.N = N
        # Regularizer
        self.c = c
        # Validate array
        self.validate_array = validate_array
        # Kernel type
        self.kernel_type = kernel_type
    
    def get_params(self, deep=True):
        return {
            'N': self.N,
            'c': self.c,
            'validate_array': self.validate_array,
            'kernel_type': self.kernel_type,
            **self.kwargs  # Merge self.kwargs into the dictionary
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self
    
    def Initialize(self, x, y):
        
        # If the kernel type is the GeneralizedGaussian, initialize the SPD matrix
        if self.kernel_type == "GeneralizedGaussian":
            self.A = np.eye(x.shape[0])
        
        # Compute the variables for the dictionary
        # Check if the kernel type is the Generalized Gaussian
        if self.kernel_type == "GeneralizedGaussian":
            k11 = self.kernel.compute(x, x, A = self.A)
        else:
            k11 = self.kernel.compute(x, x)
        
        # Update Kinv and Theta
        Kinv = np.ones((1,1)) / ( k11 + self.c)
        Theta = np.ones((1,)) * y / ( k11 + self.c)
        
        # Fill the dictionary
        self.parameters_dict.update({"Kinv": Kinv, "Theta": Theta, "m": 1., "Dict": x, "yn": np.ones((1,)) * y})
        
        # Initialize first output and residual
        self.y_pred_training = np.append(self.y_pred_training, y)
        self.ResidualTrainingPhase = np.append(self.ResidualTrainingPhase, 0.)
        
    def KRLS(self, x, y):

        # Update Dict in-place
        self.parameters_dict["Dict"] = np.hstack([self.parameters_dict["Dict"], x])
        # Update yn
        self.parameters_dict["yn"] = np.append(self.parameters_dict["yn"], y)
        # Compute k_til
        n_cols = self.parameters_dict["Dict"].shape[1]
        # If the kernel type is the GeneralizedGaussian, inform matrix SPD matrix
        if self.kernel_type == "GeneralizedGaussian":
            k_til = np.array([self.kernel.compute(self.parameters_dict["Dict"][:, ni].reshape(-1, 1), x, A = self.A) for ni in range(n_cols)])
        else:
            k_til = np.array([self.kernel.compute(self.parameters_dict["Dict"][:, ni].reshape(-1, 1), x) for ni in range(n_cols)])
        kn1 = k_til[:-1]
        knn = k_til[-1]
        # Update Kinv
        Dinv = self.parameters_dict["Kinv"]
        Dinv_kn1 = np.dot(kn1, Dinv)
        g = 1 / ( (knn + self.c) - np.dot(Dinv_kn1, kn1) )
        f = ( - Dinv_kn1 * g )
        E = Dinv - np.outer(Dinv_kn1, f)
        self.parameters_dict["Kinv"] = E 
        self.parameters_dict["Kinv"] = np.pad(self.parameters_dict["Kinv"], ((0,1),(0,1)), 'constant', constant_values=(0))
        self.parameters_dict["Kinv"][-1,-1] = g
        self.parameters_dict["Kinv"][:-1,-1] = f
        self.parameters_dict["Kinv"][-1,:-1] = f
        
        # Novelty criterion
        if n_cols > self.N:
            
            # Remove the oldest element in the dictionary
            self.parameters_dict["Dict"] = np.delete(self.parameters_dict["Dict"], 0, 1)
            # Update yn
            self.parameters_dict["yn"] = np.delete(self.parameters_dict["yn"], 0, 0)
            # Update k_til
            k_til = np.delete(k_til, 0, 0)
            # Compute Dinv
            G = self.parameters_dict["Kinv"][1:,1:]
            f = self.parameters_dict["Kinv"][1:,0]
            e = self.parameters_dict["Kinv"][0,0]
            new_K_inv = G - np.outer(f, f) / e
            # Update Kinv
            self.parameters_dict["Kinv"] = new_K_inv
        
        # Update the parameters
        self.parameters_dict["Theta"] = np.dot(self.parameters_dict["Kinv"], self.parameters_dict["yn"])
        
        # Return the kernel vector
        return k_til

class EX_KRLS(base):
    
    def __init__(self, alpha = 0.999, beta = 0.995, c = 1e-6, q = 1e-6, N = 100, validate_array = False, kernel_type = 'Linear', **kwargs):
        
        """
        Extended Kernel Recursive Least Squares (EX-KRLS) model.
        
        Parameters
        ----------

        N : int, default=100
            Accuracy parameter determining the level of sparsity. Must be an integer greater than 1.
                
        alpha : float, default=0.999
            State forgetting factor. Must be a float between 0 and 1.
        
        beta : float, default=0.995
            Data forgetting factor. Must be a float between 0 and 1.
        
        c : float, default=1e-6
            Regularization parameter. Must be a very small float.

        q : float, default=1e-6
            Trade-off between modeling variation and measurement disturbance. Must be a very small float.

        kernel_type : str, default='Linear'
            The type of kernel function to use. Must be one of the supported kernels in `base`.
        
        validate_array : bool, default=False
            If True, input arrays are validated before computation.
        
        **kwargs : dict
            Additional kernel-specific hyperparameters passed to the `base` class.
        """
        
        # Call __init__ of the base class
        super().__init__(kernel_type, validate_array, **kwargs)
        
        if (not (N > 0)) or (not(isinstance(N, int))):
            raise ValueError("N must be a positive integer.")
        
        if (not (alpha > 0 and alpha < 1)) or (not isinstance(alpha, float)):
            raise ValueError("alpha must be a float between 0 and 1.")

        if (not (beta > 0 and beta < 1)) or (not isinstance(beta, float)):
            raise ValueError("beta must be a float between 0 and 1.")
        
        if (not (c > 0 and c < 1)) or (not isinstance(c, float)):
            raise ValueError("c must be a float between 0 and 1.")

        if (not (q > 0 and q < 1)) or (not isinstance(q, float)):
            raise ValueError("q must be a float between 0 and 1.")
         
        # Hyperparameters
        # N: maximum number of elements in the dictionary
        self.N = N
        # alpha: State forgetting factor
        self.alpha = alpha
        # Data forgetting factor
        self.beta = beta
        # Regularization parameter
        self.c = c
        # Trade-off between modeling variation and measurement disturbance
        self.q = q
        # Validate array
        self.validate_array = validate_array
        # Kernel type
        self.kernel_type = kernel_type

    def get_params(self, deep=True):
        return {
            'N': self.N,
            'alpha': self.alpha,
            'beta': self.beta,
            'c': self.c,
            'q': self.q,
            'validate_array': self.validate_array,
            'kernel_type': self.kernel_type,
            **self.kwargs  # Merge self.kwargs into the dictionary
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self
    
    def Initialize(self, x, y):
        
        # Current iteration
        self.i = 1

        # If the kernel type is the GeneralizedGaussian, initialize the SPD matrix
        if self.kernel_type == "GeneralizedGaussian":
            self.A = np.eye(x.shape[0])
        
        # Compute the variables for the dictionary
        # Check if the kernel type is the Generalized Gaussian
        if self.kernel_type == "GeneralizedGaussian":
            k11 = self.kernel.compute(x, x, A = self.A)
        else:
            k11 = self.kernel.compute(x, x)
        
        # Define rho
        rho = self.c * self.beta / ( (self.alpha**2) * self.beta + self.c * self.q )
        
        # Update Kinv and Theta
        Kinv = ( np.ones((1,1)) * (self.alpha**2) ) / ( ( self.beta * self.c + k11 ) * ( (self.alpha**2) + self.beta * self.c * self.q ) )
        Theta = np.ones((1,)) * self.alpha * y / ( k11 + self.c * self.beta)
        
        # Fill the dictionary
        self.parameters_dict.update({"Kinv": Kinv, "Theta": Theta, "rho": rho, "m": 1., "Dict": x})
        
        # Initialize first output and residual
        self.y_pred_training = np.append(self.y_pred_training, y)
        self.ResidualTrainingPhase = np.append(self.ResidualTrainingPhase, 0.)
        
    def KRLS(self, x, y):
        
        # Number of cols in the dictionary
        n_cols = self.parameters_dict["Dict"].shape[1]
        if n_cols < self.N:
            # Iteration
            self.i += 1
            # Update dictionary
            self.parameters_dict["Dict"] = np.hstack([self.parameters_dict["Dict"], x])
            n_cols += 1
            # If the kernel type is the GeneralizedGaussian, inform matrix SPD matrix
            if self.kernel_type == "GeneralizedGaussian":
                k_til = np.array([self.kernel.compute(self.parameters_dict["Dict"][:, ni].reshape(-1, 1), x, A = self.A) for ni in range(n_cols)])
            else:
                k_til = np.array([self.kernel.compute(self.parameters_dict["Dict"][:, ni].reshape(-1, 1), x) for ni in range(n_cols)])
            # Compute kt and ktt
            kt = k_til[:-1]
            ktt = k_til[-1]
            # Compute z
            z = np.dot(self.parameters_dict["Kinv"], kt)
            # Compute r
            r = (self.beta**self.i) * self.parameters_dict["rho"]  + ktt - np.dot(kt, z)
            # Estimate the error
            err = y - np.dot(kt, self.parameters_dict["Theta"])
            # Update Theta
            self.parameters_dict["Theta"] = self.alpha * ( self.parameters_dict["Theta"] - z * err / r )
            self.parameters_dict["Theta"] = np.append(self.parameters_dict["Theta"], self.alpha * err / r)
            # Parcel to update of Kinv
            dummy = (self.alpha**2) + (self.beta**self.i) * self.q * self.parameters_dict["rho"]
            self.parameters_dict["rho"] = self.parameters_dict["rho"] / dummy
            # Update Kinv
            self.parameters_dict["Kinv"] = self.parameters_dict["Kinv"] * r + np.outer(z, z)
            self.parameters_dict["Kinv"] = np.pad(self.parameters_dict["Kinv"], ((0,1),(0,1)), 'constant', constant_values=(0))
            self.parameters_dict["Kinv"][-1,-1] = 1.
            self.parameters_dict["Kinv"][:-1,-1] = -z
            self.parameters_dict["Kinv"][-1,:-1] = -z
            self.parameters_dict["Kinv"] = self.alpha**2 / ( r * dummy ) * self.parameters_dict["Kinv"]
        else:
            # If the kernel type is the GeneralizedGaussian, inform matrix SPD matrix
            if self.kernel_type == "GeneralizedGaussian":
                k_til = np.array([self.kernel.compute(self.parameters_dict["Dict"][:, ni].reshape(-1, 1), x, A = self.A) for ni in range(n_cols)])
            else:
                k_til = np.array([self.kernel.compute(self.parameters_dict["Dict"][:, ni].reshape(-1, 1), x) for ni in range(n_cols)])
        
        # Return kernel vector
        return k_til

class FB_KRLS(base):
    
    def __init__(self, N = 100, c = 1e-6, validate_array = False, kernel_type = 'Linear', **kwargs):
        
        """
        Fixed Base Kernel Recursive Least Squares (FB-KRLS) model.
        
        Parameters
        ----------
        N : int, default=100
            Accuracy parameter determining the level of sparsity. 
            Must be an integer greater than 1.
                
        c : float, default=1e-6
            Regularization parameter. Must be a very small float.

        kernel_type : str, default='Linear'
            The type of kernel function to use. Must be one of the supported 
            kernels in `base`.
        
        validate_array : bool, default=False
            If True, input arrays are validated before computation.
        
        **kwargs : dict
            Additional kernel-specific hyperparameters passed to the `base` class.
        """
        
        # Call __init__ of the base class
        super().__init__(kernel_type, validate_array, **kwargs)
        
        if (not (N > 0)) or (not(isinstance(N, int))):
            raise ValueError("N must be a positive integer.")
        
        if (not (c > 0 and c < 1)) or (not isinstance(c, float)):
            raise ValueError("c must be a float between 0 and 1.")
         
        # Hyperparameters
        # N: maximum number of elements in the dictionary
        self.N = N
        # Regularization parameter
        self.c = c
        # Validate array
        self.validate_array = validate_array
        # Kernel type
        self.kernel_type = kernel_type

    def get_params(self, deep=True):
        return {
            'N': self.N,
            'c': self.c,
            'validate_array': self.validate_array,
            'kernel_type': self.kernel_type,
            **self.kwargs  # Merge self.kwargs into the dictionary
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self
    
    def Initialize(self, x, y):
        
        # If the kernel type is the GeneralizedGaussian, initialize the SPD matrix
        if self.kernel_type == "GeneralizedGaussian":
            self.A = np.eye(x.shape[0])
        
        # Compute the variables for the dictionary
        # Check if the kernel type is the Generalized Gaussian
        if self.kernel_type == "GeneralizedGaussian":
            k11 = self.kernel.compute(x, x, A = self.A)
        else:
            k11 = self.kernel.compute(x, x)
        
        # Update Kinv and Theta
        Kinv = np.ones((1,1)) / ( k11 + self.c)
        Theta = np.ones((1,)) * y / ( k11 + self.c)
        
        # Fill the dictionary
        self.parameters_dict.update({"Kinv": Kinv, "Theta": Theta, "m": 1., "Dict": x, "yn": np.ones((1,)) * y})
        
        # Initialize first output and residual
        self.y_pred_training = np.append(self.y_pred_training, y)
        self.ResidualTrainingPhase = np.append(self.ResidualTrainingPhase, 0.)
        
    def KRLS(self, x, y):
        
        # Update dictionary
        self.parameters_dict["Dict"] = np.hstack([self.parameters_dict["Dict"], x])
        # Update yn
        self.parameters_dict["yn"] = np.append(self.parameters_dict["yn"], y)
        # Compute k_til
        n_cols = self.parameters_dict["Dict"].shape[1]
        # If the kernel type is the GeneralizedGaussian, inform matrix SPD matrix
        if self.kernel_type == "GeneralizedGaussian":
            k_til = np.array([self.kernel.compute(self.parameters_dict["Dict"][:, ni].reshape(-1, 1), x, A = self.A) for ni in range(n_cols)])
        else:
            k_til = np.array([self.kernel.compute(self.parameters_dict["Dict"][:, ni].reshape(-1, 1), x) for ni in range(n_cols)])
        
        kn1 = k_til[:-1]
        knn = k_til[-1]
        # Update Kinv
        Dinv = self.parameters_dict["Kinv"]
        Dinv_kn1 = np.dot(kn1, Dinv)
        g = 1 / ( (knn + self.c) - np.dot(Dinv_kn1, kn1) )
        f = ( - Dinv_kn1 * g )
        E = Dinv - np.outer(Dinv_kn1, f)
        
        self.parameters_dict["Kinv"] = E 
        self.parameters_dict["Kinv"] = np.pad(self.parameters_dict["Kinv"], ((0,1),(0,1)), 'constant', constant_values=(0))
        self.parameters_dict["Kinv"][-1,-1] = g
        self.parameters_dict["Kinv"][:-1,-1] = f
        self.parameters_dict["Kinv"][-1,:-1] = f
        
        # Verify if the size of the dictionary is greater than N
        if self.parameters_dict["Dict"].shape[1] > self.N:
            # Compute theta
            self.parameters_dict["Theta"] = np.dot(self.parameters_dict["Kinv"], self.parameters_dict["yn"])
            theta = self.parameters_dict["Theta"]
            # Find the diagonal of Kinv
            diag = np.diagonal(self.parameters_dict["Kinv"])
            d = np.zeros(diag.shape)
            for row in range(d.shape[0]):
                if diag[row] != 0:
                    d[row] = abs(theta[row])/ diag[row]
                else:
                    d[row] = abs(theta[row])
            ind = d.argmin()
            # Remove the least relevant element in the dictionary
            self.parameters_dict["Dict"] = np.delete(self.parameters_dict["Dict"], ind, 1)
            # Update yn
            self.parameters_dict["yn"] = np.delete(self.parameters_dict["yn"], ind, 0)
            # Update k_til
            k_til = np.delete(k_til, ind)
            # Number of elements in Kinv
            idx = np.arange(self.parameters_dict["Kinv"].shape[1])
            noind = np.delete(idx, ind)
            # Compute Dinv
            G = self.parameters_dict["Kinv"][noind, :][:, noind]
            f = self.parameters_dict["Kinv"][noind, ind]
            e = self.parameters_dict["Kinv"][ind,ind]
            D_inv = G - np.outer( f, f ) / e
            # Update Kinv
            self.parameters_dict["Kinv"] = D_inv
        
        # Update the parameters
        self.parameters_dict["Theta"] = np.dot(self.parameters_dict["Kinv"], self.parameters_dict["yn"])
        
        # Return the vectors of kernel
        return k_til
    
class KRLS_T(base):
    
    def __init__(self, N = 100, c = 1e-6, gamma = 0.999 , sn2 = 1e-2, validate_array = False, kernel_type = 'Linear', **kwargs):
        
        """
        Kernel Recursive Least Squares Tracker (KRLS-T) model.
        
        Parameters
        ----------
        N : int, default=100
            Accuracy parameter determining the level of sparsity. 
            Must be an integer greater than 1.
                
        c : float, default=1e-6
            Regularization parameter. Must be a very small float.

        gamma : float, default=0.999
            Forgetting factor. Must be a float between 0 and 1.

        sn2 : float, default=1e-2
            (Sigma_n)^2. Must be a float between 0 and 1.

        kernel_type : str, default='Linear'
            The type of kernel function to use. Must be one of the supported kernels in `base`.
        
        validate_array : bool, default=False
            If True, input arrays are validated before computation.
        
        **kwargs : dict
            Additional kernel-specific hyperparameters passed to the `base` class.
        """
        
        # Call __init__ of the base class
        super().__init__(kernel_type, validate_array, **kwargs)
        
        if (not (N > 0)) or (not(isinstance(N, int))):
            raise ValueError("N must be a positive integer.")
        
        if (not (c > 0 and c < 1)) or (not isinstance(c, float)):
            raise ValueError("c must be a float between 0 and 1.")

        if (not (gamma > 0 and gamma < 1)) or (not isinstance(gamma, float)):
            raise ValueError("gamma must be a float between 0 and 1.")
        
        if (not (sn2 > 0 and sn2 < 1)) or (not isinstance(sn2, float)):
            raise ValueError("sn2 must be a float between 0 and 1.")
         
        # Hyperparameters
        # N: maximum number of elements in the dictionary
        self.N = N
        # Regularization parameter
        self.c = c
        # Forgetting factor
        self.gamma = gamma
        # Forgetting factor
        self.sn2 = sn2
        # Validate array
        self.validate_array = validate_array
        # Kernel type
        self.kernel_type = kernel_type

    def get_params(self, deep=True):
        return {
            'N': self.N,
            'c': self.c,
            'gamma': self.gamma,
            'sn2': self.sn2,
            'validate_array': self.validate_array,
            'kernel_type': self.kernel_type,
            **self.kwargs  # Merge self.kwargs into the dictionary
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self
    
    def Initialize(self, x, y):
        
        # If the kernel type is the GeneralizedGaussian, initialize the SPD matrix
        if self.kernel_type == "GeneralizedGaussian":
            self.A = np.eye(x.shape[0])
        
        # Compute the variables for the dictionary
        # Check if the kernel type is the Generalized Gaussian
        if self.kernel_type == "GeneralizedGaussian":
            k11 = self.kernel.compute(x, x, A = self.A)
        else:
            k11 = self.kernel.compute(x, x)
        
        # Initialize Sigma and mu
        Sigma = np.ones((1,)) * ( k11 - (k11**2) / ( k11 + self.sn2 ) )
        Theta = np.ones((1,)) * y * k11 / ( k11 + self.sn2 )

        # Update K, Kinv and Theta
        K = np.ones((1,1)) * ( k11 + self.c)
        Kinv = np.ones((1,1)) / ( k11 + self.c)
        
        # Fill the dictionary
        self.parameters_dict.update({"K": K, "Kinv": Kinv, "Theta": Theta, "Sigma": Sigma, "Theta": Theta, "m": 1., "Dict": x, "yn": np.ones((1,)) * y})
        
        # Initialize first output and residual
        self.y_pred_training = np.append(self.y_pred_training, y)
        self.ResidualTrainingPhase = np.append(self.ResidualTrainingPhase, 0.)
        
    def KRLS(self, x, y):

        # Update Sigma and Theta
        self.parameters_dict["Sigma"] = self.gamma * self.parameters_dict["Sigma"] + (1 - self.gamma) * self.parameters_dict["K"]
        self.parameters_dict["Theta"] *= np.sqrt(self.gamma)
        # Save the Dict and update it
        Dict_Prov = np.hstack([self.parameters_dict["Dict"], x])
        # Compute k_til
        n_cols = Dict_Prov.shape[1]
        # If the kernel type is the GeneralizedGaussian, inform matrix SPD matrix
        if self.kernel_type == "GeneralizedGaussian":
            k_til = np.array([self.kernel.compute(Dict_Prov[:, ni].reshape(-1, 1), x, A = self.A) for ni in range(n_cols)])
        else:
            k_til = np.array([self.kernel.compute(Dict_Prov[:, ni].reshape(-1, 1), x) for ni in range(n_cols)])
        kt = k_til[:-1]
        # Compute the variables for the dictionary
        # Check if the kernel type is the Generalized Gaussian
        if self.kernel_type == "GeneralizedGaussian":
            ktt = self.kernel.compute(x, x, A = self.A)
        else:
            ktt = self.kernel.compute(x, x)
        # Compute q, gamma^2, h, sf2, y_mean
        q = np.dot(self.parameters_dict["Kinv"], kt)
        gamma2 = ktt - np.dot(kt, q)
        h = np.dot(self.parameters_dict["Sigma"], q)
        sf2 = gamma2 + np.dot(q, h)
        y_mean = np.dot(q, self.parameters_dict["Theta"])
        sy2 = self.sn2 + sf2
        # Increase Sigma and Theta
        p = np.append(h, sf2)
        self.parameters_dict["Sigma"] = np.pad(self.parameters_dict["Sigma"], ((0,1),(0,1)), 'constant', constant_values=(0))
        self.parameters_dict["Sigma"][-1,-1] = sf2
        self.parameters_dict["Sigma"][:-1,-1] = h
        self.parameters_dict["Sigma"][-1,:-1] = h
        self.parameters_dict["Sigma"] = self.parameters_dict["Sigma"] - ( 1/sy2 ) * np.outer(p, p)
        self.parameters_dict["Theta"] = np.append(self.parameters_dict["Theta"], y_mean)
        self.parameters_dict["Theta"] = self.parameters_dict["Theta"] + ( ( y - y_mean ) / sy2 ) * p
        if gamma2 <= self.gamma/10:
            self.parameters_dict["Sigma"] = self.parameters_dict["Sigma"][:-1,:-1]
            self.parameters_dict["Theta"] = self.parameters_dict["Theta"][:-1]
            return kt
        else:
            # Save K and update it
            K_old = self.parameters_dict["K"]
            self.parameters_dict["K"] = np.pad(self.parameters_dict["K"], ((0,1),(0,1)), 'constant', constant_values=(0))
            self.parameters_dict["K"][-1,-1] = ktt
            self.parameters_dict["K"][:-1,-1] = kt
            self.parameters_dict["K"][-1,:-1] = kt
            # Save Kinv and update it
            Kinv_old = self.parameters_dict["Kinv"]
            self.parameters_dict["Kinv"] = np.pad(self.parameters_dict["Kinv"], ((0,1),(0,1)), 'constant', constant_values=(0))
            p1 = np.append(q, -1)
            self.parameters_dict["Kinv"] += ( 1/gamma2 ) * np.outer(p1, p1)
            if Dict_Prov.shape[1] > self.N:
                diag = np.diagonal(self.parameters_dict["Kinv"])
                errors = np.square( np.divide( np.dot(self.parameters_dict["Kinv"], self.parameters_dict["Theta"]), diag ) )
                ind = errors.argmin()
                if ind == Dict_Prov.shape[1] - 1:
                    self.parameters_dict["K"] = K_old
                    self.parameters_dict["Kinv"] = Kinv_old
                    # Indexes
                    idx = np.arange(Dict_Prov.shape[1])
                    noind = np.delete(idx, ind)
                else:
                    # Indexes to keep
                    idx = np.arange(Dict_Prov.shape[1])
                    noind = np.delete(idx, ind)
                    # Reduce K
                    self.parameters_dict["K"] = self.parameters_dict["K"][noind, :][:, noind]
                    # Reduce Kinv
                    Kinv = self.parameters_dict["Kinv"][noind, :][:, noind]
                    Kinv_s = self.parameters_dict["Kinv"][noind, ind]
                    # Kinv_sT = self.parameters_dict["Kinv"][ind,noind]
                    qs = self.parameters_dict["Kinv"][ind,ind]
                    self.parameters_dict["Kinv"] = Kinv - np.outer( Kinv_s, Kinv_s ) / qs
                    # Update the Dictionary
                    self.parameters_dict["Dict"] = Dict_Prov
                    self.parameters_dict["Dict"] = np.delete(self.parameters_dict["Dict"], ind, 1)
                # Reduce Sigma and Theta
                self.parameters_dict["Sigma"] = self.parameters_dict["Sigma"][noind, :][:, noind]
                self.parameters_dict["Theta"] = np.delete(self.parameters_dict["Theta"], ind, 0)
                # Update k_til
                k_til = k_til[noind,]
                # k_til = np.delete(k_til, ind)
            else:
                # Update the Dictionary
                self.parameters_dict["Dict"] = Dict_Prov
        
        # Return the vectors o kernel
        return k_til

class QKRLS(base):
    
    def __init__(self, N = 100, c = 1e-6, epsilon = 0.01, validate_array = False, kernel_type = 'Linear', **kwargs):
        
        """
        Quantized Kernel Recursive Least Squares (QKRLS) model.
        
        Parameters
        ----------
        N : int, default=100
            Accuracy parameter determining the level of sparsity. Must be an integer greater than 1.
                
        c : float, default=1e-6
            Regularization parameter. Must be a very small float.

        epsilon : float, default=0.01
            Quantization size. Must be a float between 0 and 1.

        kernel_type : str, default='Linear'
            The type of kernel function to use. Must be one of the supported kernels in `base`.
        
        validate_array : bool, default=False
            If True, input arrays are validated before computation.
        
        **kwargs : dict
            Additional kernel-specific hyperparameters passed to the `base` class.
        """
        
        # Call __init__ of the base class
        super().__init__(kernel_type, validate_array, **kwargs)
        
        if (not (N > 0)) or (not(isinstance(N, int))):
            raise ValueError("N must be a positive integer.")
        
        if (not (c > 0 and c < 1)) or (not isinstance(c, float)):
            raise ValueError("c must be a float between 0 and 1.")

        if (not (epsilon > 0 and epsilon < 1)) or (not isinstance(epsilon, float)):
            raise ValueError("epsilon must be a float between 0 and 1.")
         
        # Hyperparameters
        # N: maximum number of elements in the dictionary
        self.N = N
        # Regularization parameter
        self.c = c
        # Forgetting factor
        self.epsilon = epsilon
        # Validate array
        self.validate_array = validate_array
        # Kernel type
        self.kernel_type = kernel_type

    def get_params(self, deep=True):
        return {
            'N': self.N,
            'c': self.c,
            'epsilon': self.epsilon,
            'validate_array': self.validate_array,
            'kernel_type': self.kernel_type,
            **self.kwargs  # Merge self.kwargs into the dictionary
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self
    
    def Initialize(self, x, y):
        
        # If the kernel type is the GeneralizedGaussian, initialize the SPD matrix
        if self.kernel_type == "GeneralizedGaussian":
            self.A = np.eye(x.shape[0])
        
        # Compute the variables for the dictionary
        # Check if the kernel type is the Generalized Gaussian
        if self.kernel_type == "GeneralizedGaussian":
            k11 = self.kernel.compute(x, x, A = self.A)
        else:
            k11 = self.kernel.compute(x, x)
        
        # Update K, Kinv and Theta
        K = np.ones((1,1)) * ( k11 + self.c)
        Kinv = np.ones((1,1)) / ( k11 + self.c)
        Theta = np.ones((1,)) * y / ( k11 + self.c)
        
        # Fill the dictionary
        self.parameters_dict.update({"K": K, "Kinv": Kinv, "Theta": Theta, "Lambda": np.ones((1,1)), "m": 1., "Dict": x})
        
        # Initialize first output and residual
        self.y_pred_training = np.append(self.y_pred_training, y)
        self.ResidualTrainingPhase = np.append(self.ResidualTrainingPhase, 0.)
        
    def KRLS(self, x, y):
        
        # Searching for the lowest distance between the input and the dictionary inputs
        distance = []
        for ni in range(self.parameters_dict["Dict"].shape[1]):
            distance.append(np.linalg.norm(self.parameters_dict["Dict"][:,ni].reshape(-1,1) - x))
        # Find the index of minimum distance
        j = np.argmin(distance)
        # Novelty criterion
        if distance[j] <= self.epsilon:
            # Update Uppercase Lambda
            xi = np.zeros(self.parameters_dict["Theta"].shape)
            xi[j] = 1.
            self.parameters_dict["Lambda"] = self.parameters_dict["Lambda"] + np.outer(xi, xi)
            # Compute Kinvj and Kj
            Kinvj = self.parameters_dict["Kinv"][:,j]
            Kj = self.parameters_dict["K"][:,j]
            # Update Kinv
            self.parameters_dict["Kinv"] -= ( np.outer(Kinvj, np.dot(Kj, self.parameters_dict["Kinv"])) ) / ( 1 + np.dot(Kj, Kinvj) )
            # Updating Theta
            self.parameters_dict["Theta"] += Kinvj * ( y - np.dot(Kj, self.parameters_dict["Theta"]) ) / ( 1 + np.dot(Kj, Kinvj) )
            
            # Compute k_til
            n_cols = self.parameters_dict["Dict"].shape[1]
            # If the kernel type is the GeneralizedGaussian, inform matrix SPD matrix
            if self.kernel_type == "GeneralizedGaussian":
                k_til = np.array([self.kernel.compute(self.parameters_dict["Dict"][:, ni].reshape(-1, 1), x, A = self.A) for ni in range(n_cols)])
            else:
                k_til = np.array([self.kernel.compute(self.parameters_dict["Dict"][:, ni].reshape(-1, 1), x) for ni in range(n_cols)])
            
            # Return the vectors os kernels
            return k_til
            
        else:
            # Update the dictionary
            self.parameters_dict["Dict"] = np.hstack([self.parameters_dict["Dict"], x])
            # Compute k_til
            n_cols = self.parameters_dict["Dict"].shape[1]
            # If the kernel type is the GeneralizedGaussian, inform matrix SPD matrix
            if self.kernel_type == "GeneralizedGaussian":
                k_til = np.array([self.kernel.compute(self.parameters_dict["Dict"][:, ni].reshape(-1, 1), x, A = self.A) for ni in range(n_cols)])
            else:
                k_til = np.array([self.kernel.compute(self.parameters_dict["Dict"][:, ni].reshape(-1, 1), x) for ni in range(n_cols)])
            
            kt = k_til[:-1]
            ktt = k_til[-1]
            # Compute z, z_Lambda, and r
            z = np.dot(self.parameters_dict["Kinv"], kt)
            z_Lambda = np.dot(np.dot(self.parameters_dict["Kinv"], self.parameters_dict["Lambda"]), kt)
            r = self.c + ktt - np.dot(kt, z_Lambda)
            # Compute the estimated error
            e = y - np.dot(kt, self.parameters_dict["Theta"])
            # self.parameters_dict["m"] = self.parameters_dict["m"] + 1
            # Update K                      
            self.parameters_dict["K"] = np.pad(self.parameters_dict["K"], ((0,1),(0,1)), 'constant', constant_values=(0))
            self.parameters_dict["K"][-1,-1] = ktt
            self.parameters_dict["K"][:-1,-1] = kt
            self.parameters_dict["K"][-1,:-1] = kt
            # Update Lambda
            self.parameters_dict["Lambda"] = np.pad(self.parameters_dict["Lambda"], ((0,1),(0,1)), 'constant', constant_values=(0))
            # sizeLambda = self.parameters_dict["Lambda"].shape[0] - 1
            self.parameters_dict["Lambda"][-1,-1] = 1.
            # Update Kinv
            self.parameters_dict["Kinv"] = self.parameters_dict["Kinv"] * r + np.outer(z_Lambda, z)
            self.parameters_dict["Kinv"] = np.pad(self.parameters_dict["Kinv"], ((0,1),(0,1)), 'constant', constant_values=(0))
            # sizeP = self.parameters_dict["Kinv"].shape[0] - 1
            self.parameters_dict["Kinv"][-1,-1] = ktt
            self.parameters_dict["Kinv"][:-1,-1] = - z
            self.parameters_dict["Kinv"][-1,:-1] = - z
            self.parameters_dict["Kinv"] = ( 1 / r ) * self.parameters_dict["Kinv"]
            # Update Theta
            self.parameters_dict["Theta"] -= ( z_Lambda * ( 1 / r ) * e )
            self.parameters_dict["Theta"] = np.append(self.parameters_dict["Theta"], ( 1 / r ) * e )
        
            # Return the vectors os kernels
            return k_til

# class ANS_QKRLS(base):
    
#     def __init__(self, N = 100, nu = 0.01, c = 1e-6, epsilon = 0.01, mu = 1, validate_array = False, kernel_type = 'Linear', **kwargs):
        
                
#         """
#         Adaptive Normalized Sparse Quantized Kernel Recursive Least Squares (ANS-QKRLS) model.
        
#         Notes: The formulas in the source paper are incomplete and ambiguous.
        
#         Parameters
#         ----------
#         N : int, default=100
#             Accuracy parameter determining the level of sparsity. 
#             Must be an integer greater than 1.

#         nu : int, default=0.01
#             Accuracy parameter determining the level of sparsity. 
#             Must be a float between 0 and 1.
                
#         c : float, default=1e-6
#             Regularization parameter. Must be a very small float.

#         epsilon : float, default=0.01
#             Quantization size. Must be a float between 0 and 1.
        
#         mu : float, default=1
#             Coherence coefficient threshold. Must be a float between 0 and 1.

#         kernel_type : str, default='Linear'
#             The type of kernel function to use. Must be one of the supported 
#             kernels in `base`.
        
#         validate_array : bool, default=False
#             If True, input arrays are validated before computation.
        
#         **kwargs : dict
#             Additional kernel-specific hyperparameters passed to the `base` class.
#         """
        
#         # Call __init__ of the base class
#         super().__init__(kernel_type, validate_array, **kwargs)
        
#         if (not (N > 0)) or (not(isinstance(N, int))):
#             raise ValueError("N must be a positive integer.")
        
#         if (not (nu > 0 and nu < 1)) or (not isinstance(nu, float)):
#             raise ValueError("nu must be a float between 0 and 1.")
        
#         if (not (c > 0 and c < 1)) or (not isinstance(c, float)):
#             raise ValueError("c must be a float between 0 and 1.")

#         if (not (epsilon > 0 and epsilon < 1)) or (not isinstance(epsilon, float)):
#             raise ValueError("epsilon must be a float between 0 and 1.")
        
#         if (not (mu >= 0 and mu <= 1)) or (not isinstance(mu, (float,int))):
#             raise ValueError("mu must be a float between 0 and 1.")
         
#         # Hyperparameters
#         # N: maximum number of elements in the dictionary
#         self.N = N
#         # Threshold of ALD sparse rule
#         self.nu = nu
#         # Regularization parameter
#         self.c = c
#         # Forgetting factor
#         self.epsilon = epsilon
#         # Threshold for the coherence coefficient
#         self.mu = mu
#         # Validate array
#         self.validate_array = validate_array
#         # Kernel type
#         self.kernel_type = kernel_type

#     def get_params(self, deep=True):
#         return {
#             'N': self.N,
#             'nu': self.nu,
#             'c': self.c,
#             'epsilon': self.epsilon,
#             'mu': self.mu,
#             'validate_array': self.validate_array,
#             'kernel_type': self.kernel_type,
#             **self.kwargs  # Merge self.kwargs into the dictionary
#         }

#     def set_params(self, **params):
#         for key, value in params.items():
#             setattr(self, key, value)
#         return self

#     def Initialize(self, x, y):
        
#         # If the kernel type is the GeneralizedGaussian, initialize the SPD matrix
#         if self.kernel_type == "GeneralizedGaussian":
#             self.A = np.eye(x.shape[0])
        
#         # Compute the variables for the dictionary
#         # Check if the kernel type is the Generalized Gaussian
#         if self.kernel_type == "GeneralizedGaussian":
#             k11 = self.kernel.compute(x, x, A = self.A)
#         else:
#             k11 = self.kernel.compute(x, x)
        
#         # Update K, Kinv and Theta
#         K = np.ones((1,1)) * ( k11 + self.c)
#         Kinv = np.ones((1,1)) / ( k11 + self.c)
#         Theta = np.ones((1,)) * y / ( k11 + self.c)
        
#         # Fill the dictionary
#         self.parameters_dict.update({"K": K, "Kinv": Kinv, "Theta": Theta, "P": np.ones((1,1)), "m": 1., "Dict": x})
        
#         # Initialize first output and residual
#         self.y_pred_training = np.append(self.y_pred_training, y)
#         self.ResidualTrainingPhase = np.append(self.ResidualTrainingPhase, 0.)
        
#     def KRLS(self, x, y):
#         # Compute k_til
#         n_cols = self.parameters_dict["Dict"].shape[1]
#         # If the kernel type is the GeneralizedGaussian, inform matrix SPD matrix
#         if self.kernel_type == "GeneralizedGaussian":
#             k_til = np.array([self.kernel.compute(self.parameters_dict["Dict"][:, ni].reshape(-1, 1), x, A = self.A) for ni in range(n_cols)])
#         else:
#             k_til = np.array([self.kernel.compute(self.parameters_dict["Dict"][:, ni].reshape(-1, 1), x) for ni in range(n_cols)])
#         # Check if the kernel type is the Generalized Gaussian
#         if self.kernel_type == "GeneralizedGaussian":
#             ktt = self.kernel.compute(x, x, A = self.A)
#         else:
#             ktt = self.kernel.compute(x, x)
#         # Compute a
#         a = np.dot(self.parameters_dict["Kinv"], k_til)
#         delta = ktt - np.dot(k_til, a)
#         if delta == 0:
#             delta = 1.
#         # Compute the maximum value of the kernel
#         mu = max(np.abs(k_til))
#         # Compute the distances between x and the Dictionary
#         distance = []
#         for ni in range(self.parameters_dict["Dict"].shape[1]):
#             distance.append(np.linalg.norm(self.parameters_dict["Dict"][:,ni].reshape(-1,1) - x))
#         # Find the index of minimum distance
#         j = np.argmin(distance)
#         # Estimating the error
#         EstimatedError = ( y - np.dot(k_til, self.parameters_dict["Theta"]) )
        
#         # Novelty criterion
#         if delta > self.nu and mu.item() <= self.mu and distance[j] > self.epsilon and self.parameters_dict["Dict"].shape[1] < self.N:
        
#             self.parameters_dict["Dict"] = np.hstack([self.parameters_dict["Dict"], x])
#             self.parameters_dict["m"] += 1
#             # Update K  
#             self.parameters_dict["K"] = np.pad(self.parameters_dict["K"], ((0,1),(0,1)), 'constant', constant_values=(0))
#             sizeK = self.parameters_dict["K"].shape[0] - 1
#             self.parameters_dict["K"][-1,-1] = ktt
#             self.parameters_dict["K"][0:sizeK,sizeK] = k_til
#             self.parameters_dict["K"][-1,:-1] = k_til
#             # Updating Kinv                      
#             self.parameters_dict["Kinv"] = (1/delta)*(self.parameters_dict["Kinv"] * delta + np.outer(a, a))
#             self.parameters_dict["Kinv"] = np.pad(self.parameters_dict["Kinv"], ((0,1),(0,1)), 'constant', constant_values=(0))
#             # sizeKinv = self.parameters_dict["Kinv"].shape[0] - 1
#             self.parameters_dict["Kinv"][-1,-1] = (1/delta)
#             self.parameters_dict["Kinv"][:-1,-1] = (1/delta)*(-a)
#             self.parameters_dict["Kinv"][-1,:-1] = (1/delta)*(-a)
#             # Updating P
#             self.parameters_dict["P"] = np.pad(self.parameters_dict["P"], ((0,1),(0,1)), 'constant', constant_values=(0))
#             # sizeP = self.parameters_dict["P"].shape[0] - 1
#             self.parameters_dict["P"][-1,-1] = 1.
#             # Updating alpha
#             self.parameters_dict["Theta"] -= ( ( a / delta ) * EstimatedError )
#             self.parameters_dict["Theta"] = np.append(self.parameters_dict["Theta"], ( 1 / delta ) * EstimatedError)
#             k_til = np.append(k_til, ktt)
            
#         elif ( delta <= self.nu or mu.item() > self.mu ) and distance[j] <= self.epsilon:
        
#             # Compute Kinvj and Kj
#             Kinvj = self.parameters_dict["P"][:,j]
#             Kj = self.parameters_dict["K"][:,j]
#             # Updating Kinv - Kinv is equivalent to P in QKRLS
#             self.parameters_dict["Kinv"] -= ( np.outer(Kinvj, np.dot(Kj, self.parameters_dict["Kinv"])) ) / ( 1 + np.dot(Kj, Kinvj) )
#             # Updating alpha
#             self.parameters_dict["Theta"] += np.dot(Kinvj, ( y - np.dot(Kj, self.parameters_dict["Theta"]) ) ) / ( ( 1 + np.dot(Kj, Kinvj) ) )
            
#         else:
#             # Compute parcels
#             A_P = np.dot(a, self.parameters_dict["P"])
#             A_P_A = np.dot(A_P, a)
#             # Calculating q
#             q = A_P / ( 1 + A_P_A )
#             # Updating P
#             self.parameters_dict["P"] -= np.outer(q, A_P)
#             # Updating alpha
#             self.parameters_dict["Theta"] += ( np.dot(self.parameters_dict["Kinv"], q) * EstimatedError ) / ( self.c + np.linalg.norm(k_til)**2 )
        
#         # Return the vector of kernels
#         return k_til

class ADA_KRLS(base):
    
    def __init__(self, N = 100, nu = 0.01, c = 1e-6, validate_array = False, kernel_type = 'Linear', **kwargs):
        
        """
        Adaptive Dynamic Adjustment Kernel Recursive Least Squares (ADA-KRLS) model.
        
        Parameters
        ----------
        N : int, default=100
            Accuracy parameter determining the level of sparsity. 
            Must be an integer greater than 1.

        nu : int, default=0.01
            Accuracy parameter determining the level of sparsity. Must be a float between 0 and 1.
                
        c : float, default=1e-6
            Regularization parameter. Must be a very small float.

        kernel_type : str, default='Linear'
            The type of kernel function to use. Must be one of the supported kernels in `base`.
        
        validate_array : bool, default=False
            If True, input arrays are validated before computation.
        
        **kwargs : dict
            Additional kernel-specific hyperparameters passed to the `base` class.
        """
        
        # Call __init__ of the base class
        super().__init__(kernel_type, validate_array, **kwargs)
        
        if (not (N > 0)) or (not(isinstance(N, int))):
            raise ValueError("N must be a positive integer.")
        
        if (not (nu > 0 and nu < 1)) or (not isinstance(nu, float)):
            raise ValueError("nu must be a float between 0 and 1.")
        
        if (not (c > 0 and c < 1)) or (not isinstance(c, float)):
            raise ValueError("c must be a float between 0 and 1.")
         
        # Hyperparameters
        # N: maximum number of elements in the dictionary
        self.N = N
        # Threshold of ALD sparse rule
        self.nu = nu
        # Regularization parameter
        self.c = c
        # Validate array
        self.validate_array = validate_array
        # Kernel type
        self.kernel_type = kernel_type

    def get_params(self, deep=True):
        return {
            'N': self.N,
            'nu': self.nu,
            'c': self.c,
            'validate_array': self.validate_array,
            'kernel_type': self.kernel_type,
            **self.kwargs  # Merge self.kwargs into the dictionary
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self
    
    def Initialize(self, x, y):
        
        # If the kernel type is the GeneralizedGaussian, initialize the SPD matrix
        if self.kernel_type == "GeneralizedGaussian":
            self.A = np.eye(x.shape[0])
        
        # Compute the variables for the dictionary
        # Check if the kernel type is the Generalized Gaussian
        if self.kernel_type == "GeneralizedGaussian":
            k11 = self.kernel.compute(x, x, A = self.A)
        else:
            k11 = self.kernel.compute(x, x)
        
        # Update Kinv and Theta
        Kinv = np.ones((1,1)) / ( k11 + self.c)
        Theta = np.ones((1,)) * y / ( k11 + self.c)
        
        # Fill the dictionary
        self.parameters_dict.update({"Kinv": Kinv, "Theta": Theta, "P": np.ones((1,1)), "m": 1., "Dict": x, "yn": np.ones((1,)) * y})
        
        # Initialize first output and residual
        self.y_pred_training = np.append(self.y_pred_training, y)
        self.ResidualTrainingPhase = np.append(self.ResidualTrainingPhase, 0.)
        
    def KRLS(self, x, y):
        # Compute k_til
        n_cols = self.parameters_dict["Dict"].shape[1]
        # If the kernel type is the GeneralizedGaussian, inform matrix SPD matrix
        if self.kernel_type == "GeneralizedGaussian":
            k_til = np.array([self.kernel.compute(self.parameters_dict["Dict"][:, ni].reshape(-1, 1), x, A = self.A) for ni in range(n_cols)])
        else:
            k_til = np.array([self.kernel.compute(self.parameters_dict["Dict"][:, ni].reshape(-1, 1), x) for ni in range(n_cols)])
        # Check if the kernel type is the Generalized Gaussian
        if self.kernel_type == "GeneralizedGaussian":
            ktt = self.kernel.compute(x, x, A = self.A)
        else:
            ktt = self.kernel.compute(x, x)
        # Compute a
        a = np.dot(self.parameters_dict["Kinv"], k_til)
        delta = ktt - np.dot( k_til, a )
        if delta == 0:
            delta = 1.
        # Estimating the error
        EstimatedError = ( y - np.dot(k_til, self.parameters_dict["Theta"]) )
        # Novelty criterion
        if delta > self.nu and self.parameters_dict["Dict"].shape[1] < self.N:
        
            # Update the dictionary
            self.parameters_dict["Dict"] = np.hstack([self.parameters_dict["Dict"], x])
            # Update yn
            self.parameters_dict["yn"] = np.append(self.parameters_dict["yn"], y)
            # Update m
            self.parameters_dict["m"] += 1
            # Update Kinv                      
            self.parameters_dict["Kinv"] = (1/delta)*(self.parameters_dict["Kinv"] * delta + np.outer(a, a))
            self.parameters_dict["Kinv"] = np.pad(self.parameters_dict["Kinv"], ((0,1),(0,1)), 'constant', constant_values=(0))
            # sizeKinv = self.parameters_dict["Kinv"].shape[0] - 1
            self.parameters_dict["Kinv"][-1,-1] = (1/delta)
            self.parameters_dict["Kinv"][:-1,-1] = (1/delta)*(-a)
            self.parameters_dict["Kinv"][-1,:-1] = (1/delta)*(-a)
            # Update P
            self.parameters_dict["P"] = np.pad(self.parameters_dict["P"], ((0,1),(0,1)), 'constant', constant_values=(0))
            # sizeP = self.parameters_dict["P"].shape[0] - 1
            self.parameters_dict["P"][-1,-1] = 1.
            # Update Theta
            self.parameters_dict["Theta"] = self.parameters_dict["Theta"] - ( ( a / delta ) * EstimatedError )
            self.parameters_dict["Theta"] = np.append(self.parameters_dict["Theta"], ( 1 / delta ) * EstimatedError )
            k_til = np.append(k_til, ktt)
            
        # Verify if the size of the dictionary is greater than N
        elif delta > self.nu and self.parameters_dict["Dict"].shape[1] > self.N:
        
            # Update dictionary
            self.parameters_dict["Dict"] = np.hstack([self.parameters_dict["Dict"], x])
            # Update yn
            self.parameters_dict["yn"] = np.append(self.parameters_dict["yn"], y)
            # Compute k_til
            kt = k_til
            k_til = np.append(k_til, ktt)
            # Compute Dinv
            D_inv = self.parameters_dict["Kinv"]
            # Update Kinv
            g = 1 / ( ( ktt + self.c ) - np.dot(np.dot(kt, D_inv), kt) )
            f = ( - np.dot(D_inv, kt) * g )
            E = D_inv - np.outer(np.dot(D_inv, kt), f)
            self.parameters_dict["Kinv"] = E 
            self.parameters_dict["Kinv"] = np.pad(self.parameters_dict["Kinv"], ((0,1),(0,1)), 'constant', constant_values=(0))
            self.parameters_dict["Kinv"][-1,-1] = g
            self.parameters_dict["Kinv"][:-1,-1] = f
            self.parameters_dict["Kinv"][-1,:-1] = f
            
            # Compute Theta
            self.parameters_dict["Theta"] = np.dot(self.parameters_dict["Kinv"], self.parameters_dict["yn"])
            alpha = self.parameters_dict["Theta"]
            # Find the diagonal of Kinv
            diag = np.diagonal(self.parameters_dict["Kinv"])
            d = np.zeros(diag.shape)
            for row in range(d.shape[0]):
                if diag[row] != 0:
                    d[row] = abs(alpha[row])/ diag[row]
                else:
                    d[row] = abs(alpha[row])
            j = d.argmin()
            # Remove the least relevant element in the dictionary
            self.parameters_dict["Dict"] = np.delete(self.parameters_dict["Dict"], j, 1)
            # Update yn
            self.parameters_dict["yn"] = np.delete(self.parameters_dict["yn"], j, 0)
            # Update k
            k_til = np.delete(k_til, j)
            # Number of elements in Kinv
            idx = np.arange(self.parameters_dict["Kinv"].shape[1])
            noj = np.delete(idx, j)
            # Compute Dinv
            G = self.parameters_dict["Kinv"][noj, :][:, noj]
            f = self.parameters_dict["Kinv"][noj, j].reshape(-1,1)
            e = self.parameters_dict["Kinv"][j,j]
            D_inv = G - np.outer(f, f) / e
            # Update Kinv
            self.parameters_dict["Kinv"]= D_inv
            # Compute Theta
            self.parameters_dict["Theta"] = np.dot(self.parameters_dict["Kinv"], self.parameters_dict["yn"])
                
        else:
            # Compute parcels            
            A_P = np.dot(a, self.parameters_dict["P"])
            A_P_A = np.dot(A_P, a)
            # Calculating q
            q = A_P / ( 1 + A_P_A )
            # Updating P
            self.parameters_dict["P"] = self.parameters_dict["P"] - np.outer(q, A_P)
            # Updating alpha
            self.parameters_dict["Theta"] = self.parameters_dict["Theta"] + ( np.dot(self.parameters_dict["Kinv"], q) * EstimatedError ) / ( self.c + np.linalg.norm(k_til)**2 )
        
        return k_til

class QALD_KRLS(base):
    
    def __init__(self, N = 100, c = 1e-6, nu = 0.01, epsilon1 = 0.1, epsilon2 = 0.1,validate_array = False, kernel_type = 'Linear', **kwargs):
        
        """
        Quantized Adaptive Dynamic Adjustment Kernel Recursive Least Squares (QALD-KRLS) model.
        
        Parameters
        ----------
        N : int, default=100
            Accuracy parameter determining the level of sparsity. 
            Must be an integer greater than 1.
 
        c : float, default=1e-6
            Regularization parameter. Must be a very small float.
        
        nu : int, default=0.01
            Accuracy parameter determining the level of sparsity. 
            Must be a float between 0 and 1.

        epsilon1 : int, default=0.1
            Accuracy parameter determining the level of sparsity. 
            Must be a float between 0 and 1.

        epsilon2 : int, default=0.1
            Accuracy parameter determining the level of sparsity. 
            Must be a float between 0 and 1.
                
        kernel_type : str, default='Linear'
            The type of kernel function to use. Must be one of the 
            supported kernels in `base`.
        
        validate_array : bool, default=False
            If True, input arrays are validated before computation.
        
        **kwargs : dict
            Additional kernel-specific hyperparameters passed to the `base` class.
        """
        
        # Call __init__ of the base class
        super().__init__(kernel_type, validate_array, **kwargs)
        
        if (not (N > 0)) or (not(isinstance(N, int))):
            raise ValueError("N must be a positive integer.")
        
        if (not (c > 0 and c < 1)) or (not isinstance(c, float)):
            raise ValueError("c must be a float between 0 and 1.")

        if (not (nu > 0 and nu < 1)) or (not isinstance(nu, float)):
            raise ValueError("nu must be a float between 0 and 1.")
        
        if (not (epsilon1 > 0 and epsilon1 < 1)) or (not isinstance(epsilon1, float)):
            raise ValueError("epsilon1 must be a float between 0 and 1.")
        
        if (not (epsilon2 > 0 and epsilon2 < 1)) or (not isinstance(epsilon2, float)):
            raise ValueError("epsilon2 must be a float between 0 and 1.")
        
         
        # Hyperparameters
        # N: maximum number of elements in the dictionary
        self.N = N
        # Regularization parameter
        self.c = c
        # Threshold of ALD sparse rule
        self.nu = nu
        # Threshold of ALD sparse rule
        self.epsilon1 = epsilon1
        # Threshold of ALD sparse rule
        self.epsilon2 = epsilon2
        # Validate array
        self.validate_array = validate_array
        # Kernel type
        self.kernel_type = kernel_type

    def get_params(self, deep=True):
        return {
            'N': self.N,
            'c': self.c,
            'nu': self.nu,
            'epsilon1': self.epsilon1,            
            'epsilon2': self.epsilon2,
            'validate_array': self.validate_array,
            'kernel_type': self.kernel_type,
            **self.kwargs  # Merge self.kwargs into the dictionary
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self
    
    def Initialize(self, x, y):
        
        # If the kernel type is the GeneralizedGaussian, initialize the SPD matrix
        if self.kernel_type == "GeneralizedGaussian":
            self.A = np.eye(x.shape[0])
        
        # Compute the variables for the dictionary
        # Check if the kernel type is the Generalized Gaussian
        if self.kernel_type == "GeneralizedGaussian":
            k11 = self.kernel.compute(x, x, A = self.A)
        else:
            k11 = self.kernel.compute(x, x)
        
        # Update Kinv and Theta
        Kinv = np.ones((1,1)) / ( k11 + self.c)
        Theta = np.ones((1,)) * y / ( k11 + self.c)
        
        # Fill the dictionary
        self.parameters_dict.update({"Kinv": Kinv, "Theta": Theta, "P": np.ones((1,1)), "m": 1., "Dict": x})
        
        # Initialize first output and residual
        self.y_pred_training = np.append(self.y_pred_training, y)
        self.ResidualTrainingPhase = np.append(self.ResidualTrainingPhase, 0.)
        
    def KRLS(self, x, y):
        # Compute k_til
        n_cols = self.parameters_dict["Dict"].shape[1]
        # If the kernel type is the GeneralizedGaussian, inform matrix SPD matrix
        if self.kernel_type == "GeneralizedGaussian":
            k_til = np.array([self.kernel.compute(self.parameters_dict["Dict"][:, ni].reshape(-1, 1), x, A = self.A) for ni in range(n_cols)])
        else:
            k_til = np.array([self.kernel.compute(self.parameters_dict["Dict"][:, ni].reshape(-1, 1), x) for ni in range(n_cols)])
        # Check if the kernel type is the Generalized Gaussian
        if self.kernel_type == "GeneralizedGaussian":
            ktt = self.kernel.compute(x, x, A = self.A)
        else:
            ktt = self.kernel.compute(x, x)
        # Compute a
        a = np.dot(self.parameters_dict["Kinv"], k_til)
        delta = ktt - np.dot( k_til, a )
        if delta == 0:
            delta = 1.
        # Searching for the lowest distance between the input and the dictionary inputs
        distance = []
        for ni in range(self.parameters_dict["Dict"].shape[1]):
            distance.append(np.linalg.norm(self.parameters_dict["Dict"][:,ni].reshape(-1,1) - x))
        # Find the index of minimum distance
        j = np.argmin(distance)
        # Estimating the error
        EstimatedError = ( y - np.dot(k_til, self.parameters_dict["Theta"]) )
        # Novelty criterion
        if delta > self.nu and distance[j] > self.epsilon1 and n_cols < self.N:
            self.parameters_dict["Dict"] = np.hstack([self.parameters_dict["Dict"], x])
            self.parameters_dict["m"] += 1
            # Updating Kinv                      
            self.parameters_dict["Kinv"] = (1/delta)*(self.parameters_dict["Kinv"] * delta + np.outer(a, a))
            self.parameters_dict["Kinv"] = np.pad(self.parameters_dict["Kinv"], ((0,1),(0,1)), 'constant', constant_values=(0))
            # sizeKinv = self.parameters_dict["Kinv"].shape[0] - 1
            self.parameters_dict["Kinv"][-1,-1] = (1/delta)
            self.parameters_dict["Kinv"][:-1,-1] = (1/delta)*(-a)
            self.parameters_dict["Kinv"][-1,:-1] = (1/delta)*(-a)
            # Updating P
            self.parameters_dict["P"] = np.pad(self.parameters_dict["P"], ((0,1),(0,1)), 'constant', constant_values=(0))
            # sizeP = self.parameters_dict["P"].shape[0] - 1
            self.parameters_dict["P"][-1,-1] = 1.
            # Updating Theta
            self.parameters_dict["Theta"] = self.parameters_dict["Theta"] - ( ( a / delta ) * EstimatedError )
            self.parameters_dict["Theta"] = np.append(self.parameters_dict["Theta"], ( 1 / delta ) * EstimatedError)
            k_til = np.append(k_til, ktt)
        
        else:
        
            if distance[j] <= self.epsilon2:
                xi = np.zeros(self.parameters_dict["Theta"].shape)
                xi[j] = 1.
                a = xi
            
            # Compute parcels            
            A_P = np.dot(a, self.parameters_dict["P"])
            A_P_A = np.dot(A_P, a)
            # Calculating q
            q = A_P / ( 1 + A_P_A )
            # Updating P
            self.parameters_dict["P"] = self.parameters_dict["P"] - np.outer(q, A_P)
            # Updating alpha
            self.parameters_dict["Theta"] = self.parameters_dict["Theta"] + ( np.dot(self.parameters_dict["Kinv"], q) * EstimatedError ) / ( self.c + np.linalg.norm(k_til)**2 )
            
        return k_til

class Light_KRLS(base):
    
    def __init__(self, N = 100, c = 1e-6, validate_array = False, kernel_type = 'Gaussian', **kwargs):
        
        """
        Light Kernel Recursive Least Squares (Light-KRLS) model.
        
        Parameters
        ----------
        N : int, default=100
            Accuracy parameter determining the level of sparsity. 
            Must be an integer greater than 1.
 
        c : float, default=1e-6
            Regularization parameter. Must be a very small float.
        
        kernel_type : str, default='Gaussian'
            The type of kernel function to use. Must be one of the 
            supported kernels in `base`.
        
        validate_array : bool, default=False
            If True, input arrays are validated before computation.
        
        **kwargs : dict
            Additional kernel-specific hyperparameters passed to the `base` class.
        """
        
        # Call __init__ of the base class
        super().__init__(kernel_type, validate_array, **kwargs)
        
        if (not (N > 0)) or (not(isinstance(N, int))):
            raise ValueError("N must be a positive integer.")
        
        if (not (c > 0 and c < 1)) or (not isinstance(c, float)):
            raise ValueError("c must be a float between 0 and 1.")
         
        # Hyperparameters
        # N: maximum number of elements in the dictionary
        self.N = N
        # Regularization parameter
        self.c = c
        # Validate array
        self.validate_array = validate_array
        # Kernel type
        self.kernel_type = kernel_type

    def get_params(self, deep=True):
        return {
            'N': self.N,
            'c': self.c,
            'validate_array': self.validate_array,
            'kernel_type': self.kernel_type,
            **self.kwargs  # Merge self.kwargs into the dictionary
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self
    
    def Initialize(self, x, y):
        
        # If the kernel type is the GeneralizedGaussian, initialize the SPD matrix
        if self.kernel_type == "GeneralizedGaussian":
            self.A = np.eye(x.shape[0])
        
        # Compute the variables for the dictionary
        # Check if the kernel type is the Generalized Gaussian
        if self.kernel_type == "GeneralizedGaussian":
            k11 = self.kernel.compute(x, x, A = self.A)
        else:
            k11 = self.kernel.compute(x, x)
        
        # Update Kinv and Theta
        Kinv = np.ones((1,1)) / ( k11 + self.c)
        Theta = np.ones((1,)) * y / ( k11 + self.c)
        
        # Fill the dictionary
        self.parameters_dict.update({"Kinv": Kinv, "Theta": Theta, "P": np.ones((1,1)), "m": 1., "Dict": x})
        
        # Initialize first output and residual
        self.y_pred_training = np.append(self.y_pred_training, y)
        self.ResidualTrainingPhase = np.append(self.ResidualTrainingPhase, 0.)
        
    def KRLS(self, x, y):
        
        # Compute k_til
        n_cols = self.parameters_dict["Dict"].shape[1]
        # If the kernel type is the GeneralizedGaussian, inform matrix SPD matrix
        if self.kernel_type == "GeneralizedGaussian":
            k_til = np.array([self.kernel.compute(self.parameters_dict["Dict"][:, ni].reshape(-1, 1), x, A = self.A) for ni in range(n_cols)])
        else:
            k_til = np.array([self.kernel.compute(self.parameters_dict["Dict"][:, ni].reshape(-1, 1), x) for ni in range(n_cols)])
        
        knn = k_til[-1]
        # Compute the coefficients a
        a = np.dot(self.parameters_dict["Kinv"], k_til)
        # Estimate the error
        error = ( y - np.dot(k_til, self.parameters_dict["Theta"]) )
        # Add a new input to the dictionary if the size of the dictionary is smaller than dict_size
        if self.parameters_dict["m"] <= self.N:
            self.parameters_dict["Dict"] = np.hstack([self.parameters_dict["Dict"], x])
            self.parameters_dict["m"] += 1
            # Compute the approximate linear dependence (ALD)
            delta = knn - np.dot(k_til, a)
            if delta == 0:
                delta = 1.
            # Update Kinv
            self.parameters_dict["Kinv"] = (1/delta)*(self.parameters_dict["Kinv"] * delta + np.outer(a, a))
            self.parameters_dict["Kinv"] = np.pad(self.parameters_dict["Kinv"], ((0,1),(0,1)), 'constant', constant_values=(0))
            # sizeKinv = self.parameters_dict["Kinv"].shape[0] - 1
            self.parameters_dict["Kinv"][-1,-1] = (1/delta)
            self.parameters_dict["Kinv"][:-1,-1] = (1/delta)*(-a)
            self.parameters_dict["Kinv"][-1,:-1] = (1/delta)*(-a)
            # Update P
            self.parameters_dict["P"] = np.pad(self.parameters_dict["P"], ((0,1),(0,1)), 'constant', constant_values=(0))
            # sizeP = self.parameters_dict["P"].shape[0] - 1
            self.parameters_dict["P"][-1,-1] = 1.
            # Update alpha
            self.parameters_dict["Theta"] = self.parameters_dict["Theta"] - ( ( a / delta ) * error )
            self.parameters_dict["Theta"] = np.append(self.parameters_dict["Theta"], ( 1 / delta ) * error)
            k_til = np.append(k_til, knn)
        
        else:
            
            # Compute parcels            
            A_P = np.dot(a, self.parameters_dict["P"])
            A_P_A = np.dot(A_P, a)
            # Calculating q
            q = A_P / ( 1 + A_P_A )
            # Updating P
            self.parameters_dict["P"] = self.parameters_dict["P"] - np.outer(q, A_P)
            # Updating alpha
            self.parameters_dict["Theta"] = self.parameters_dict["Theta"] + ( np.dot(self.parameters_dict["Kinv"], q) * error ) #/ ( self.c + np.linalg.norm(k_til)**2 )
        
        # Return the vector of kernels
        return k_til