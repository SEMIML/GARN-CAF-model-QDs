import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error

def fit_and_evaluate(T_measured, T_theoretical):
    def quadratic_func(x, a, b, c):
        return a * x**2 + b * x + c
    
    T_measured = np.uint16(T_measured)
    #print(T_measured)
    T_theoretical = np.uint16(T_theoretical)
    #print(T_theoretical)
    # Fit the quadratic function
    popt, _ = curve_fit(quadratic_func, T_measured, T_theoretical)
    T_theoretical_fit = quadratic_func(T_measured, *popt)
    mse = mean_squared_error(T_theoretical, T_theoretical_fit)
    
    # Output the quadratic fit parameters and MSE
    a, b, c = popt
    #print(f'Quadratic fit parameters: a={a}, b={b}, c={c}, MSE={mse}')
    
    # Calculate x for which quadratic function gives output 490
    coefficients_490 = [a, b, c - 490]
    roots_490 = np.roots(coefficients_490)
    x_490 = roots_490[np.isreal(roots_490)].real  # Select the real roots
    positive_x_490 = [round(x, 1) for x in x_490 if x > 0]
    #print(f"Measured temperatures for theoretical 490: {positive_x_490}")
    
    # Calculate x for which quadratic function gives output 600
    coefficients_600 = [a, b, c - 600]
    roots_600 = np.roots(coefficients_600)
    x_600 = roots_600[np.isreal(roots_600)].real  # Select the real roots
    positive_x_600 = [round(x, 1) for x in x_600 if x > 0]
    #print(f"Measured temperatures for theoretical 600: {positive_x_600}")
    reture_result=np.array([a, b, c, mse, *positive_x_490, *positive_x_600])
    
    reture_result=reture_result.tolist()
      
    return reture_result

# Example usage
T_measured = np.array([118, 343, 415, 450])
T_theoretical = np.array([350, 510, 580, 620])
results = fit_and_evaluate(T_measured, T_theoretical)
print("Results:", results)
