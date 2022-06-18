import numpy as np
from scipy.stats import chi2, t, norm

def pvalue_integration(p, method="fisher"):
    """
    P value integration.

    Arguments:
        p: np.ndarray
            array of p values
        method: str, default="fisher"
            integration method. "fisher", "MG" or "stouffer"
    
    Retruns:
        p_norm: np.ndarray
           normalized p values

    Raises
        ValueError: if method not "fisher", "MG" or "stouffer".
    """
    for i in range(p.shape[1]):
        zero_idx = np.where(p[:, i] == 0)[0]
        if len(zero_idx) > 0:
            p[zero_idx, i] = np.linspace(1e-10, 1e-323, len(zero_idx))

        almost_zero_idx = np.where(p[:, i] < 1e-323)[0]
        if len(almost_zero_idx) > 0:
            p[almost_zero_idx, i] = 1e-323

        almost_one_idx = np.where(p[:, i] > 0.99999999999999994)
        if len(almost_one_idx) > 0:
            p[almost_one_idx, i] = 0.99999999999999994

    if method == "fisher":
        z = -2 * np.log(p)
        df = np.sum(~np.isnan(z), axis=1)
        z = np.nansum(z, axis=1)
        p_norm = 1 - chi2.cdf(z, 2*df)
    elif method == "MG":
        z = np.log(p/(1-p))
        df = np.sum(~np.isnan(z), axis=1)
        z = np.nansum(z, axis=1) * (-1)*np.sqrt((15*df+12)/((5*df+2)*df*np.pi**2))
        p_norm = 1 - t.cdf(z, 5*df+4)
    elif method == "stouffer":
        z = (-1) * norm.ppf(p)
        df = np.sum(~np.isnan(z), axis=1)
        z = np.nansum(z, axis=1) / np.sqrt(df)
        p_norm = 1 - norm.cdf(z)
    else:
        raise ValueError(f"method must be either 'fisher', 'MG', 'stouffer'")
    
    return p_norm

