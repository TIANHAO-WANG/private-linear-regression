import numpy as np

def adassp(X, y, epsilon, delta):
    BX = 1
    BY = 1

    n, d = X.shape

    varrho=0.05

    # set the eigenvalue limit
    eta = np.sqrt(d*np.log(6/delta)*np.log(2*d**2/varrho))*BX**2/(epsilon/3)

    XTy = np.dot(X.T, y)
    XTX = np.dot(X.T, X) + np.identity(d)

    S = np.linalg.svd(XTX)
    S = np.diag(S[1])
    logsod = np.log(6/delta)

    lamb_min = S[-1, -1] + np.random.standard_normal()*BX**2*np.sqrt(logsod)/(epsilon/3) - logsod/(epsilon/3)

    lamb = max(0, eta-lamb_min)

    XTyhat = XTy + (np.sqrt(np.log(6/delta))/(epsilon/3))*BX*BY*np.random.standard_normal(d)

    Z = np.random.standard_normal((d, d))
    Z = 0.5*(Z + Z.T)

    XTXhat = XTX + (np.sqrt(np.log(6/delta))/(epsilon/3))*BX*BX*Z

    thetahat = np.linalg.solve( XTXhat+lamb*np.identity(d), XTyhat )

    return thetahat

