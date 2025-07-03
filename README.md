# KRLS: Kernel Recursive Least Squares

## Project description

Author: Kaike Sa Teles Rocha Alves

KRLS: Kernel Recursive Least Squares is a package that contains machine learning models developed by coded by Kaike Alves during his PhD research. 

    Author: Kaike Sa Teles Rocha Alves (PhD)
    Email: kaikerochaalves@outlook.com or kaike.alves@estudante.ufjf.br

Doi to cite the code: http://dx.doi.org/10.5281/zenodo.15800969

Github repository: https://github.com/kaikerochaalves/KRLS_pypi.git


It provides:

1. Kernel Recursive Least Squares: KRLS
2. Sliding Window Kernel Recursive Least Squares: SW-KRLS
3. Extended Kernel Recursive Least Squares: EX-KRLS
4. Fixed Base Kernel Recursive Least Squares: FB-KRLS
5. Kernel Recursive Least Squares Tracker: KRLS-T
6. Quantized Kernel Recursive Least Squares: QKRLS
7. Adaptive Dynamic Adjustment Kernel Recursive Least Squares: ADA-KRLS
8. Quantized Adaptive Dynamic Adjustment Kernel Recursive Least Squares: QALD-KRLS
9. Light Kernel Recursive Least Squares: Light-KRLS


Cite: SA TELES ROCHA ALVES, K. (2025). KRLS: Kernel Recursive Least Squares. Zenodo. https://doi.org/10.5281/zenodo.15800969

## Description:

KRLS: A Novel Python Library for Kernel Recursive Least Squares Model

KRLS (Kernel Recursive Least Squares) is a groundbreaking Python library available on PyPI (https://pypi.org/project/krls/) that introduces a suite of advanced fuzzy inference systems. This library is specifically designed to tackle the complexities of time series forecasting and classification problems by offering machine learning models that prioritize high accuracy.

At its core, KRLS features data-driven machine learning models. 

## Instructions

To install the library use the command: 

    pip install krls

The library provides 6 models in fuzzy systems, as follows:

### KRLS

To import the KRLS, simply type the command:

    from krls.krls import KRLS

Hyperparameters:

    nu : float, default=0.01
    Accuracy parameter determining the level of sparsity. Must be a positive float.

    N : int, default=100
    Accuracy parameter determining the level of sparsity. Must be a integer greater 
    than 1.

    kernel_type : str
    The type of kernel function to use. Must be one of: 'Linear', 'Polynomial', 'RBF', 'Gaussian',
    'Sigmoid', 'Powered', 'Log', 'GeneralizedGaussian', 'Hybrid', additive_chi2, and Cosine.

    validate_array : bool
    If True, input arrays are validated before computation.

    kernel_type : str, default='Linear'
    The type of kernel function to use. Must be one of the supported kernels in `base`.

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

Example of KRLS:

    from krls.krls import KRLS
    model = KRLS(nu=0.001, N=500, kernel_type="Gaussian", sigma=0.5)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

### SW-KRLS

To import the SW-KRLS, simply type the command:

    from krls.krls import SW_KRLS

Hyperparameters:

    N : int, default=100
    Accuracy parameter determining the level of sparsity. 
    Must be an integer greater than 1.

    c : float, default=1e-6
    Regularization parameter. Must be a very small float.

    kernel_type : str
    The type of kernel function to use. Must be one of: 'Linear', 'Polynomial', 'RBF', 'Gaussian',
    'Sigmoid', 'Powered', 'Log', 'GeneralizedGaussian', 'Hybrid', additive_chi2, and Cosine.

    validate_array : bool
    If True, input arrays are validated before computation.

    kernel_type : str, default='Linear'
    The type of kernel function to use. Must be one of the supported kernels in `base`.

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

Example of SW_KRLS:

    from krls.krls import SW_KRLS
    model = SW_KRLS(N=500, kernel_type="Gaussian", sigma=0.5)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

### EX-KRLS

To import the EX-KRLS, simply type the command:

    from krls.krls import EX_KRLS

Hyperparameters:

    alpha : float, default=0.999
    State forgetting factor. Must be a float between 0 and 1.

    beta : float, default=0.995
    Data forgetting factor. Must be a float between 0 and 1.

    c : float, default=1e-6
    Regularization parameter. Must be a very small float.

    q : float, default=1e-6
    Trade-off between modeling variation and measurement disturbance. Must be a very small float.

    N : int, default=100
    Accuracy parameter determining the level of sparsity. Must be an integer greater than 1.

    kernel_type : str
    The type of kernel function to use. Must be one of: 'Linear', 'Polynomial', 'RBF', 'Gaussian',
    'Sigmoid', 'Powered', 'Log', 'GeneralizedGaussian', 'Hybrid', additive_chi2, and Cosine.

    validate_array : bool
    If True, input arrays are validated before computation.

    kernel_type : str, default='Linear'
    The type of kernel function to use. Must be one of the supported kernels in `base`.

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

Example of EX_KRLS:

    from krls.krls import EX_KRLS
    model = EX_KRLS(kernel_type="Gaussian", sigma=0.5)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

### FB-KRLS

To import the EX-KRLS, simply type the command:

    from krls.krls import FB_KRLS

Hyperparameters:

    N : int, default=100
    Accuracy parameter determining the level of sparsity. 
    Must be an integer greater than 1.

    c : float, default=1e-6
    Regularization parameter. Must be a very small float.

    validate_array : bool
    If True, input arrays are validated before computation.

    kernel_type : str, default='Linear'
    The type of kernel function to use. Must be one of the supported kernels in `base`.

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

Example of FB_KRLS:

    from krls.krls import FB_KRLS
    model = FB_KRLS(N=500, kernel_type="Gaussian", sigma=0.5)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

### KRLS-T

To import the KRLS-T, simply type the command:

    from krls.krls import KRLS_T

Hyperparameters:

    N : int, default=100
    Accuracy parameter determining the level of sparsity. 
    Must be an integer greater than 1.

    c : float, default=1e-6
    Regularization parameter. Must be a very small float.

    validate_array : bool
    If True, input arrays are validated before computation.

    kernel_type : str, default='Linear'
    The type of kernel function to use. Must be one of the supported kernels in `base`.

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

Example of KRLS_T:

    from krls.krls import KRLS_T
    model = KRLS_T(N=500, kernel_type="Gaussian", sigma=0.5)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

### QKRLS

To import the QKRLS, simply type the command:

    from krls.krls import QKRLS

Hyperparameters:

    N : int, default=100
    Accuracy parameter determining the level of sparsity. Must be an integer greater than 1.
        
    c : float, default=1e-6
    Regularization parameter. Must be a very small float.

    epsilon : float, default=0.01
    Quantization size. Must be a float between 0 and 1.

    validate_array : bool
    If True, input arrays are validated before computation.

    kernel_type : str, default='Linear'
    The type of kernel function to use. Must be one of the supported kernels in `base`.

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

Example of QKRLS:

    from krls.krls import QKRLS
    model = QKRLS(N=500, kernel_type="Gaussian", sigma=0.5)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

### ADA-KRLS

To import the ADA-KRLS, simply type the command:

    from krls.krls import ADA_KRLS

Hyperparameters:

    N : int, default=100
    Accuracy parameter determining the level of sparsity. 
    Must be an integer greater than 1.

    nu : int, default=0.01
    Accuracy parameter determining the level of sparsity. Must be a float between 0 and 1.

    c : float, default=1e-6
    Regularization parameter. Must be a very small float.

    validate_array : bool
    If True, input arrays are validated before computation.

    kernel_type : str, default='Linear'
    The type of kernel function to use. Must be one of the supported kernels in `base`.

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

Example of ADA-KRLS:

    from krls.krls import ADA_KRLS
    model = ADA_KRLS(N=500, kernel_type="Gaussian", sigma=0.5)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

### QALD-KRLS

To import the QALD-KRLS, simply type the command:

    from krls.krls import QALD_KRLS

Hyperparameters:

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

    kernel_type : str, default='Gaussian'
    The type of kernel function to use. Must be one of the 
    supported kernels in `base`.

    validate_array : bool
    If True, input arrays are validated before computation.

    kernel_type : str, default='Linear'
    The type of kernel function to use. Must be one of the supported kernels in `base`.

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

Example of QALD-KRLS:

    from krls.krls import QALD_KRLS
    model = QALD_KRLS(N=500, epsilon1 = 1e-4, kernel_type="Gaussian", sigma=0.5)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

### Light-KRLS

To import the Light-KRLS, simply type the command:

    from krls.krls import Light_KRLS

Hyperparameters:

    N : int, default=100
    Accuracy parameter determining the level of sparsity. 
    Must be an integer greater than 1.

    c : float, default=1e-6
    Regularization parameter. Must be a very small float.

    kernel_type : str, default='Gaussian'
    The type of kernel function to use. Must be one of the 
    supported kernels in `base`.

    validate_array : bool
    If True, input arrays are validated before computation.

    kernel_type : str, default='Linear'
    The type of kernel function to use. Must be one of the supported kernels in `base`.

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

Example of Light-KRLS:

    from krls.krls import Light_KRLS
    model = Light_KRLS(N=500, kernel_type="Gaussian", sigma=0.5)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

## Extra information

Code of Conduct:

KRLS is a library developed by Kaike Alves. Please read the Code of Conduct for guidance.

Call for Contributions:

The project welcomes your expertise and enthusiasm!

Small improvements or fixes are always appreciated. If you are considering larger contributions to the source code, please contact by email first.