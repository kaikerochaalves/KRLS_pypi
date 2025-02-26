# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 15:08:53 2025

@author: Kaike Sa Teles Rocha Alves
@email: kaikerochaalves@outlook.com
"""

# Importing libraries
import numpy as np

class base():
    
    def __init__(self, **kwargs):
        
        # List of predefined valid parameters
        self.valid_params = ['kernel_type', 'a', 'b', 'd', 'sigma', 'r', 'beta', 'tau']  # Adjust this list as needed

        # Initialize the dictionary
        self.hyperparameters_dict = {}

        # Default values for parameters
        self.default_values = {
            'kernel_type': 'Hybrid',
            'a': 1,
            'b': 1,
            'd': 2, 
            'sigma': 1.0,
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
        if self.hyperparameters_dict['kernel_type'] in ["Gaussian","RBF"]:
            keys = ['kernel_type', 'sigma']
            self.kwargs = {key: self.hyperparameters_dict.get(key, None) for key in keys}
        elif self.hyperparameters_dict['kernel_type'] in ["Linear", "GeneralizedGaussian"]:
            keys = ['kernel_type']
            self.kwargs = {key: self.hyperparameters_dict.get(key, None) for key in keys}
        elif self.hyperparameters_dict['kernel_type'] == "Polynomial":
            keys = ['kernel_type', 'a', 'b', 'd']
            self.kwargs = {key: self.hyperparameters_dict.get(key, None) for key in keys}
        elif self.hyperparameters_dict['kernel_type'] in ["Powered","Log"]:
            keys = ['kernel_type', 'beta']
            self.kwargs = {key: self.hyperparameters_dict.get(key, None) for key in keys}
        elif self.hyperparameters_dict['kernel_type'] == "Hybrid":
            keys = ['kernel_type', 'sigma', 'tau', 'd']
            self.kwargs = {key: self.hyperparameters_dict.get(key, None) for key in keys}
        
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
    
    def __init__(self, nu = 0.1, sigma = 0.1, **kwargs):
        
        # Call __init__ of the base class
        super().__init__(**kwargs)
        
        if not (nu > 0):
            raise ValueError("nu must be a positive float.")
        if not (sigma > 0):
            raise ValueError("sigma must be a positive float.")
        
        # Hyperparameters
        # Kernel width
        self.sigma = sigma
        # nu is an accuracy parameter determining the level of sparsity
        self.nu = nu
        
         
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
                      
            # Update KRLS
            k_til = self.KRLS(x, y[k])
            
            # Compute output
            Output = self.parameters_dict["Theta"].T @ k_til
            
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
            
        # # Be sure that X is with a correct shape
        # X = X.reshape(-1,self.parameters.loc[self.parameters.index[0],'Center'].shape[0])
        
        # Preallocate space for the outputs for better performance
        self.y_pred_test = np.zeros((X.shape[0]))

        for k in range(X.shape[0]):
            
            # Prepare the first input vector
            x = X[k,].reshape((1,-1)).T

            # Compute k
            k_til = np.array(())
            for ni in range(self.parameters_dict["Dict"].shape[1]):
                k_til = np.append(k_til, [self.Kernel(self.parameters_dict["Dict"][:,ni].reshape(-1,1), x)])
            k_til = k_til.reshape(k_til.shape[0],1)
            
            # Compute the output
            Output = self.parameters_dict["Theta"].T @ k_til
            
            # Store the results
            self.y_pred_test[k,] = Output.item()
            
        return self.y_pred_test

    def Kernel(self, x1, x2):
        k = np.exp( - ( 1/2 ) * ( (np.linalg.norm( x1 - x2 ))**2 ) / ( self.sigma**2 ) )
        return k
    
    def Initialize(self, x, y):
        
        # Compute the variables for the dictionary
        k11 = self.Kernel(x, x)
        Kinv = np.ones((1,1)) / ( k11 )
        alpha = np.ones((1,1)) * y / k11
        
        # Fill the dictionary
        self.parameters_dict.update({"Kinv": Kinv, "Theta": alpha, "P": np.ones((1,1)), "m": 1., "Dict": x})
        
        # Initialize first output and residual
        self.y_pred_training = np.append(self.y_pred_training, y)
        self.ResidualTrainingPhase = np.append(self.ResidualTrainingPhase, 0.)
        
    def KRLS(self, x, y):
    
        # Compute k
        k = np.array(())
        for ni in range(self.parameters_dict["Dict"].shape[1]):
            k = np.append(k, [self.Kernel(self.parameters_dict["Dict"][:,ni].reshape(-1,1), x)])
        k_til = k.reshape(-1,1)
        # Compute a
        a = np.matmul(self.parameters_dict["Kinv"], k)
        A = a.reshape(-1,1)
        delta = self.Kernel(x, x) - ( k_til.T @ A ).item()
        if delta == 0:
            delta = 1.
        # Estimating the error
        EstimatedError = ( y - np.matmul(k_til.T, self.parameters_dict["Theta"]) ).item()
        # Novelty criterion
        if delta > self.nu:
            self.parameters_dict["Dict"] = np.hstack([self.parameters_dict["Dict"], x])
            self.parameters_dict["m"] = self.parameters_dict["m"] + 1
            # Updating Kinv                      
            self.parameters_dict["Kinv"] = (1/delta)*(self.parameters_dict["Kinv"] * delta + np.matmul(A, A.T))
            self.parameters_dict["Kinv"] = np.lib.pad(self.parameters_dict["Kinv"], ((0,1),(0,1)), 'constant', constant_values=(0))
            sizeKinv = self.parameters_dict["Kinv"].shape[0] - 1
            self.parameters_dict["Kinv"][sizeKinv,sizeKinv] = (1/delta)
            self.parameters_dict["Kinv"][0:sizeKinv,sizeKinv] = (1/delta)*(-a)
            self.parameters_dict["Kinv"][sizeKinv,0:sizeKinv] = (1/delta)*(-a)
            # Updating P
            self.parameters_dict["P"] = np.lib.pad(self.parameters_dict["P"], ((0,1),(0,1)), 'constant', constant_values=(0))
            sizeP = self.parameters_dict["P"].shape[0] - 1
            self.parameters_dict["P"][sizeP,sizeP] = 1.
            # Updating alpha
            self.parameters_dict["Theta"] = self.parameters_dict["Theta"] - ( ( A / delta ) * EstimatedError )
            self.parameters_dict["Theta"] = np.vstack([self.parameters_dict["Theta"], ( 1 / delta ) * EstimatedError ])
            k_til = np.append(k_til, self.Kernel(x, x).reshape(1,1), axis=0)
        
        else:

            # Calculating q
            q = np.matmul( self.parameters_dict["P"], A) / ( 1 + np.matmul(np.matmul(A.T, self.parameters_dict["P"]), A ) )
            # Updating P
            self.parameters_dict["P"] = self.parameters_dict["P"] - (np.matmul(np.matmul(np.matmul(self.parameters_dict["P"], A), A.T), self.parameters_dict["P"])) / ( 1 + np.matmul(np.matmul(A.T, self.parameters_dict["P"]), A))
            # Updating alpha
            self.parameters_dict["Theta"] = self.parameters_dict["Theta"] + np.matmul(self.parameters_dict["Kinv"], q) * EstimatedError
        
        return k_til