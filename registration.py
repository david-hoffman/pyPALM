#!/usr/bin/env python
# -*- coding: utf-8 -*-
# registration.py
"""
Classes for registering point sets. Based on:
Myronenko and Xubo Song - 2010 - Point Set Registration Coherent Point Drift
DOI: 10.1109/TPAMI.2010.46

Copyright (c) 2018, David Hoffman
"""

import numpy as np
import scipy.spatial.distance as distance
import scipy.linalg as la
import logging


class BaseCPD(object):
    """Base class for the coherent point drift algorithm based on:
    Myronenko and Xubo Song - 2010 - Point Set Registration Coherent Point Drift
    DOI: 10.1109/TPAMI.2010.46
    """
    def __init__(self, X, Y):
        """Set up the registration class that will actually perform the CPD algorithm
        
        Parameters
        ----------
        X : ndarray (N, D)
            Fixed point cloud, an N by D array of N original observations in an n-dimensional space
        Y : ndarray (M, D)
            Moving point cloud, an M by D array of N original observations in an n-dimensional space
        """
        self.X = X
        self.Y = Y
        # extract the dimensions
        self.N, self.D = self.X.shape
        self.M, D = self.Y.shape
        assert D == self.D, "Point clouds have different dimensions"
    
    def updateTY(self):
        """Update the transformed point cloud and distance matrix"""
        self.TY = self.Y @ self.B.T + self.translation
        # we need to update the distance matrix too
        # This gives us a matrix of ||x - T(y)||**2, eq (1)
        # But we want the rows to be m and columns n
        self.dist_matrix = distance.cdist(self.TY, self.X, 'sqeuclidean')
        # make sure we have the right shape
        assert self.dist_matrix.shape == (self.M, self.N), "Error with dist_matrix"
    
    # these are defined in the paper but not used, included here for completeness
    # @property
    # def pGMM(self):
    #     """The probability density of the gaussian mixture model along the fixed points"""
    #     norm_factor = self.M * (2 * np.pi * self.var) ** (self.D / 2)
    #     p_mat = np.exp(- self.dist_matrix / 2 / self.var) / norm_factor
    #     # sum along the fixed points
    #     return p_mat.sum(0)
    
    # @property
    # def p(self):
    #     """The total probability including the uniform distribution"""
    #     return self.w / self.N + (1 - self.w) * self.pGMM
    
    def estep(self):
        """The expectation step were we calculate the posterior probability of the GMM centroids"""
        # calculate "P_old" via equation 6
        p_mat = np.exp(-self.dist_matrix / 2 / self.var)
        c = (2 * np.pi * self.var) ** (self.D / 2)
        c *= self.w / (1 - self.w)
        c *= self.M / self.N
        # sum along the moving points, i.e. along M
        denominator = p_mat.sum(0, keepdims=True)
        assert denominator.size == self.N, "Calculation of denominator failed {}".format(denominator.shape)
        # check if denominator is all zeros, which means p_mat is all zeros
        if (denominator == 0).all():
            # then the final p should just be a uniform distribution
            # should log or warn user this is happening
            logging.warning("P_mat is null, resetting to uniform probabilities")
            p_old = np.ones_like(p_mat) / self.M
        else:
            p_old = p_mat / (denominator + c)
        # compute Np, make sure it's neither zero nor more than N
        self.Np = min(self.N, max(p_old.sum(), np.finfo(float).eps))
        # update Q so we can track convergence using equation (5)
        self.Q = (p_old * self.dist_matrix).sum() / 2 / self.var + self.Np * self.D * np.log(self.var) / 2
        # store p_old
        self.p_old = p_old
        
    def updateB(self):
        """Update B matrix, this is the only method that needs to be overloaded for
        the various linear transformation subclasses, more will need to be done for
        non-rigid transformation models"""
        raise NotImplementedError
        
    def mstep(self):
        """Maximization step: update transformation and variance these are the transposes of the
        equations on p. 2265 and 2266
        """

        # calculate intermediate values
        mu_x = (self.p_old @ self.X).sum(0, keepdims=True) / self.Np
        mu_y = (self.p_old.T @ self.Y).sum(0, keepdims=True) / self.Np
        assert mu_x.size == mu_y.size == self.D, "Dimensions on mu's are wrong"
        Xhat = self.Xhat = self.X - mu_x
        Yhat = self.Yhat = self.Y - mu_y
        
        # calculate A
        self.A = A = Xhat.T @ self.p_old.T @ Yhat
        
        # calculate B
        B = self.updateB()
        
        # calculate translation
        self.translation = (mu_x - mu_y @ B.T)

        # calculate estimate of variance
        self.var = np.trace(Xhat.T @ np.diag(self.p_old.sum(0)) @ Xhat) - np.trace(A @ B.T)
        self.var /= self.Np * self.D
        # make sure self.var is positive
        if self.var < np.finfo(float).eps:
            self.var = np.finfo(float).eps
            logging.warning("Variance has dropped below machine precision, setting to eps")
            # self.var = self.init_var = self.init_var * 2
            # self.translation = -self.Y.mean(axis=0) + self.X.mean(axis=0)
            # print("Var small resetting to", self.var)
    
    def update(self):
        # expectation, maximization followed by transformation
        self.estep()
        self.mstep()
        self.updateTY()
        # now update Q to follow convergence
        # Q_delta = np.abs((self.Q_old - self.Q) / self.Q_old)
        Q_delta = np.abs((self.Q_old - self.Q))
        self.Q_old = self.Q
        return Q_delta
    
    def __call__(self, tol=1e-6, maxiters=1000, init_var=None, weight=0, B=None, translation=None):
        """perform the actual registration"""
        # initialize starting transform if requested
        if translation is None:
            translation = np.ones((1, self.D))
        self.translation = translation
        if B is None:
            B = np.eye(self.D)
        self.B = B

        # update to the initial position
        self.updateTY()
        
        # set up initial variance
        if init_var is None:
            init_var = self.dist_matrix.sum() / (self.D * self.N * self.M)
        self.var = self.init_var = init_var
        
        # initialize the weight of the uniform distribution
        assert 0 <= weight <= 1, "Weight must be between 0 and 1"
        self.w = weight
        
        # initialize Q
        self.Q_old = np.finfo(float).eps
        
        for i in range(maxiters):
            # do iterations
            delta_Q = self.update()
            if delta_Q <= tol:
                break
            self.iteration = i
        else:
            logging.warning(("Maximum iterations ({}) reached with" +
                             " out convergence, final delta_Q = {:.3e}").format(i, delta_Q))

        return self.TY


class TranslationCPD(BaseCPD):
    """Coherent point drift with a translation only transformation model"""
    def updateB(self):
        """Translation only means that B should be identity"""
        self.B = np.eye(self.D)
        return self.B


class SimilarityCPD(BaseCPD):
    """Coherent point drift with a similarity (translation, rotation and isotropic scaling)
    transformation model"""
    # this class is specifically designed so that it can be easily subclassed to represent
    # a rigid transformation model.
    def calculateR():
        """Calculate the estimated rotation matrix, eq. (9)"""
        U, S, VT = la.svd(self.A)
        c = np.ones_like(s)
        c[-1] = la.det(U @ VT)
        C = np.diag(c)
        self.R = U @ C @ VT
        return self.R
    
    def calculateS():
        """Calculate the scale factor, Fig 2 p. 2266"""
        a = self.Yhat.T @ np.diag(self.p_old.sum(1)) @ self.Yhat
        self.s = np.trace(self.A.T @ self.R) / np.trace(a)
        return self.s
    
    def updateB(self):
        """B in this case is just the rotation matrix multiplied by the scale factor"""
        R = self.calculateR()
        s = self.calculateS()
        self.B = s * R
        return self.B


class RigidCPD(SimilarityCPD):
    """Coherent point drift with a rigid or Euclidean (translation and rotation) transformation model"""
    def calculateS(self):
        """No scaling for this guy"""
        return 1

EuclideanCPD = RigidCPD


class AffineCPD(BaseCPD):
    """Coherent point drift with a similarity (translation, rotation, shear and anisotropic scaling)
    transformation model"""
    def updateB(self):
        """Solve for B using equations in Fig. 3 p. 2266"""
        a = self.Yhat.T @ np.diag(self.p_old.sum(1)) @ self.Yhat
        # solve B = self.A @ np.inv(a) == B @ a = self.A == a.T @ B.T = self.A.T
        # self.B = la.solve(a.T, self.A.T).T
        # a is a symmetric matrix
        self.B = la.solve(a, self.A.T).T
        return self.B
