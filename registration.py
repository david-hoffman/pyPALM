import scipy.spatial.distance as distance


class BaseCPD(object):
    pass


class TranslationCPD(BaseCPD):
    """A class that performs translation only coherent point drift
    
    Refs:
    """
    def __init__(self, X, Y):
        """
        
        Parameters
        X : ndarray (N, D)
            Fixed point cloud, an N by D array of N original observations in an n-dimensional space
        Y : ndarray (M, D)
            Moving point cloud, an M by D array of N original observations in an n-dimensional space
        """
        self.X = X
        self.Y = Y
        self.N, self.D = self.X.shape
        self.M, D = self.Y.shape
        assert D == self.D, "Point clouds have different dimensions"
    
    @property
    def TY(self):
        return self.Y + self.translation
    
    @property
    def pGMM(self):
        """The probability density of the gaussian mixture model along the fixed points"""
        norm_factor = self.M * (2 * np.pi * self.var) ** (self.D / 2)
        p_mat = np.exp(- self.dist_matrix / 2 / self.var) / norm_factor
        # sum along the fixed points
        return p_mat.sum(0)
    
    @property
    def p(self):
        """The total probability including the uniform distribution"""
        return self.w / self.N + (1 - self.w) * self.pGMM
    
    @property
    def dist_matrix(self):
        """This gives us a matrix of ||x - T(y)||**2, eq (1)"""
        dist_matrix = distance.cdist(self.TY, self.X, 'sqeuclidean')
        assert dist_matrix.shape == (self.M, self.N), "Error with dist_matrix"
        return dist_matrix
    
    @property
    def p_old(self):
        """The posterior probabilities of the GMM components"""
        # This gives us a matrix of ||x - y||**2, eq (1)
        # But we want the rows to be m and columns n
        p_mat = np.exp(-self.dist_matrix / 2 / self.var)
        c = (2 * np.pi * self.var) ** (self.D / 2)
        c *= self.w / (1 - self.w)
        c *= self.M / self.N
        # sum along the moving points
        p_old = p_mat / (p_mat.sum(0, keepdims=True) + c)
        # update Q so we can track convergence
        # equation (5)
        Np = p_old.sum()
        self.Q = (p_old * self.dist_matrix).sum() / 2 / self.var + Np * self.D * np.log(self.var) / 2
        return p_old
    
    def update(self):
        # E-step, compute p_old
        p_old = self.p_old
        # restrict Np to a reasonable range
        Np = min(100.0, max(p_old.sum(), np.finfo(float).eps))
        # M-step update transformation and variance
        # these are the transposes of the equations on 
        # p. 2265
        mu_x = (p_old @ self.X).sum(0, keepdims=True) / Np
        mu_y = (p_old.T @ self.Y).sum(0, keepdims=True) / Np
        self.translation = (mu_x - mu_y)
        assert self.translation.shape == (1, self.D), "Translation wrong shape"
        # update the variance
        Xhat = self.X - mu_x
        Yhat = self.Y - mu_y
        A = Xhat.T @ p_old.T @ Yhat
        self.var = np.trace(Xhat.T @ np.diag(p_old.sum(0)) @ Xhat) - np.trace(A.T)
        self.var /= Np * self.D
        # make sure self.var is positive
        if self.var < np.finfo(float).eps:
            self.var = self.dist_matrix.sum() / (self.D * self.N * self.M)
        
        # Q_delta = np.abs((self.Q_old - self.Q) / self.Q_old)
        Q_delta = np.abs((self.Q_old - self.Q))
        self.Q_old = self.Q
        return Q_delta
        
    def __call__(self, tol=1e-6, maxiters=1000, weight=0, init_var=None, transform=None):
        """perform the actual registration"""
        if transform is None:
            self.translation = np.ones((1, self.D))
        self.w = weight
        if init_var is None:
            self.var = self.dist_matrix.sum() / (self.D * self.N * self.M)
        self.Q_old = 0.0
        for i in range(maxiters):
            # do iterations
            delta_Q = self.update()
            if delta_Q <= tol:
                break
        self.iteration = i
        # return results
        return self.TY, self.translation, self.p_old