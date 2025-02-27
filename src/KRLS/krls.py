# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 15:08:53 2025

@author: Kaike Sa Teles Rocha Alves
@email: kaikerochaalves@outlook.com
"""

# Importing libraries
import numpy as np
from kernel import Kernel

class base():
    
    def __init__(self, kernel_type, validate_array, **kwargs):
        
        """
        Base class for kernel-based learning models.
        
        Parameters
        ----------
        kernel_type : str
            The type of kernel function to use. Must be one of: 'Linear', 'Polynomial', 'RBF', 'Gaussian',
            'Sigmoid', 'Powered', 'Log', 'GeneralizedGaussian', 'Hybrid'.
        
        validate_array : bool
            If True, input arrays are validated before computation.
        
        **kwargs : dict
            Additional hyperparameters depending on the chosen kernel:
            - 'a', 'b', 'd' : Polynomial kernel parameters
            - 'sigma' : Gaussian, RBF, and Hybrid kernel parameter
            - 'r' : Sigmoid kernel parameter
            - 'beta' : Powered and Log kernel parameter
            - 'tau' : Hybrid kernel parameter
        """
        
        # List of predefined valid parameters
        self.valid_params = ['kernel_type', 'validate_array', 'a', 'b', 'd', 'sigma', 'r', 'beta', 'tau']  # Adjust this list as needed

        # Initialize the dictionary
        self.hyperparameters_dict = {"kernel_type": kernel_type, "validate_array": validate_array}
        
        # Default values for parameters
        self.default_values = {
            'a': 1,
            'b': 1,
            'd': 2, 
            'sigma': 10,
            'r': 0,
            'beta': 1.0,
            'tau': 1.0, 
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
        if kernel_type in ["Gaussian","RBF"]:
            keys = ['kernel_type', 'validate_array', 'sigma']
            self.kwargs = {key: self.hyperparameters_dict.get(key, None) for key in keys}
        elif kernel_type in ['Linear', 'GeneralizedGaussian']:
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
        if param == 'sigma' and (not isinstance(value, (int, float)) and value <= 0):
            return False
        # Example validation rule for 'param3' (must be a float between 0 and 10)
        if param == 'beta' and ((not isinstance(value, (int, float)) or not (0 < value <= 1))):
            return False
        return True

class KRLS(base):
    
    def __init__(self, nu = 0.1, kernel_type = 'Gaussian', validate_array = False, **kwargs):
        
        """
        Kernel Recursive Least Squares (KRLS) model.
        
        Parameters
        ----------
        nu : float, default=0.1
            Accuracy parameter determining the level of sparsity. Must be a positive float.
        
        kernel_type : str, default='Gaussian'
            The type of kernel function to use. Must be one of the supported kernels in `base`.
        
        validate_array : bool, default=False
            If True, input arrays are validated before computation.
        
        **kwargs : dict
            Additional kernel-specific hyperparameters passed to the `base` class.
        """
        
        # Call __init__ of the base class
        super().__init__(kernel_type, validate_array, **kwargs)
        
        if not (nu > 0):
            raise ValueError("nu must be a positive float.")
        
        # Hyperparameters
        # Kernel type
        self.kernel_type = kernel_type
        # nu is an accuracy parameter determining the level of sparsity
        self.nu = nu
        # Validate array
        self.validate_array = validate_array
        
    def get_params(self, deep=True):
        return {
            'nu': self.nu,
            'kernel_type': self.kernel_type,
            'validate_array': self.validate_array,
            **self.kwargs  # Merge self.kwargs into the dictionary
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self
         
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

        for k in range(1, X.shape[0]):

            # Prepare the k-th input vector
            x = X[k,].reshape((1,-1)).T
            
            # If the kernel type is the GeneralizedGaussian, update the SPD matrix
            if self.kernel_type == "GeneralizedGaussian":
                self.A -= ((self.A @ x @ x.T @ self.A) / (1 + x.T @ self.A @ x))
                      
            # Update KRLS
            k_til = self.KRLS(x, y[k])
            
            # Compute output
            Output = np.dot(self.parameters_dict["Theta"], k_til)
            
            # Store results
            self.y_pred_training = np.append(self.y_pred_training, Output )
            self.ResidualTrainingPhase = np.append(self.ResidualTrainingPhase,(y[k]) - Output )
            
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
            
        return self.y_pred_test
    
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
            delta = self.kernel.compute(x, x, A = self.A) - np.dot(k_til, a).item()
        else:
            delta = self.kernel.compute(x, x) - np.dot(k_til, a).item()
        
        # Avoid zero division
        if delta == 0:
            delta = 1.
            
        # Compute the residual
        EstimatedError = y - np.dot(k_til, self.parameters_dict["Theta"]) 
        
        # Novelty criterion
        if delta > self.nu:
            
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
        
        return k_til