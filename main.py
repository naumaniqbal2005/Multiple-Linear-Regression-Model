import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import f

dataset = pd.read_csv('data.csv')

X = dataset.iloc[:, 1:].values 
y = dataset.iloc[:, 0].values

X = np.append(arr=np.ones((X.shape[0], 1)).astype(int), values=X, axis=1)


def backward_elimination_ssr(X, y, alpha=0.05):
    full_model = sm.OLS(y, X).fit()
    remaining_cols = list(range(X.shape[1]))
    
    # Calculate F-critical value
    df_numerator = 1  # df for one variable being removed
    df_denominator = len(y) - X.shape[1]  # residual df
    f_critical = f.ppf(1 - alpha, df_numerator, df_denominator)

    print("Full model equation:")
    print_equation(full_model.params, remaining_cols)

    num_vars = X.shape[1]
    for i in range(num_vars):
        current_model = sm.OLS(y, X).fit()
        
        # Find variable with lowest F-statistic using SSR method
        min_f_stat = float('inf')
        min_f_index = -1
        
        for j in range(1, len(current_model.params)):  # Skip intercept
            # Test removing each variable
            X_temp = np.delete(X, j, axis=1)
            reduced_model = sm.OLS(y, X_temp).fit()
            
            # Calculate F-statistic using ESS (Explained Sum of Squares / SSR)
            # F = (SSR_full - SSR_reduced) / MSE_full
            ess_diff = current_model.ess - reduced_model.ess
            f_stat = ess_diff / current_model.mse_resid
            
            if f_stat < min_f_stat:
                min_f_stat = f_stat
                min_f_index = j
        
        # Remove variable if F-statistic is below critical value
        if min_f_stat < f_critical:
            remaining_cols.pop(min_f_index)
            X = np.delete(X, min_f_index, axis=1)
        else:
            break

    final_model = sm.OLS(y, X).fit()
    print("Reduced model equation:")
    print_equation(final_model.params, remaining_cols)
    
    return X


def print_equation(params, col_indices):
    equation = f"y = {params[0]:.4f}"
    for i in range(1, len(params)):
        original_col = col_indices[i]
        if params[i] < 0:
            params[i] = -params[i]
            equation += f" - {params[i]:.4f}x{original_col}"
        else:
            equation += f" + {params[i]:.4f}x{original_col}"
    print(equation)

X_optimized = backward_elimination_ssr(X, y)
