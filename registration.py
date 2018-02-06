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


class BaseCPD(object):
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
    
    def updateTY(self):
        self.TY = self.Y @ self.B.T + self.translation
    
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
    
    @property
    def dist_matrix(self):
        """This gives us a matrix of ||x - T(y)||**2, eq (1)"""
        # This gives us a matrix of ||x - y||**2, eq (1)
        # But we want the rows to be m and columns n
        dist_matrix = distance.cdist(self.TY, self.X, 'sqeuclidean')
        assert dist_matrix.shape == (self.M, self.N), "Error with dist_matrix"
        return dist_matrix
    
    def estep(self):
        """The posterior probabilities of the GMM components"""
        # save distance matrix to minimize computation
        dist_matrix = self.dist_matrix
        p_mat = np.exp(-dist_matrix / 2 / self.var)
        c = (2 * np.pi * self.var) ** (self.D / 2)
        c *= self.w / (1 - self.w)
        c *= self.M / self.N
        # sum along the moving points
        denominator = p_mat.sum(0, keepdims=True)
        if (denominator == 0).all():
            # if denominator is all zeros, which means p_mat is all zeros
            # then the final p should just be a uniform distribution
            # should log or warn user this is happening
            print("warn")
            p_old = np.ones_like(p_mat) / self.M 
        else:
            p_old = p_mat / (denominator + c)
        # compute Np, make sure it's neither zero or more than N
        self.Np = min(self.N, max(p_old.sum(), np.finfo(float).eps))
        # update Q so we can track convergence
        # equation (5)
        self.Q = (p_old * dist_matrix).sum() / 2 / self.var + self.Np * self.D * np.log(self.var) / 2
        self.p_old = p_old
        
    def updateB(self):
        """Update B matrix, this is the only method that needs to be overloaded"""
        raise NotImplementedError
        
    def mstep(self):
        # M-step update transformation and variance
        # these are the transposes of the equations on 
        # p. 2265
        p_old = self.p_old
        mu_x = (p_old @ self.X).sum(0, keepdims=True) / self.Np
        mu_y = (p_old.T @ self.Y).sum(0, keepdims=True) / self.Np
        # update the variance
        Xhat = self.Xhat = self.X - mu_x
        Yhat = self.Yhat = self.Y - mu_y
        
        self.A = A = Xhat.T @ p_old.T @ Yhat
        
        B = self.updateB()
        
        # calculate translation
        self.translation = (mu_x - mu_y @ B.T)
        
        self.var = np.trace(Xhat.T @ np.diag(p_old.sum(0)) @ Xhat) - np.trace(A @ B.T)
        self.var /= self.Np * self.D
        # make sure self.var is positive
        if self.var < np.finfo(float).eps:
            self.var = np.finfo(float).eps
            # self.var = self.init_var = self.init_var * 2
            # self.translation = -self.Y.mean(axis=0) + self.X.mean(axis=0)
            # print("Var small resetting to", self.var)
    
    def update(self):
        # E-step, compute p_old, Np, Q
        self.estep()
        self.mstep()
        self.updateTY()
        # Q_delta = np.abs((self.Q_old - self.Q) / self.Q_old)
        Q_delta = np.abs((self.Q_old - self.Q))
        self.Q_old = self.Q
        return Q_delta
    
    def register(self, tol=1e-6, maxiters=1000, init_var=None, weight=0):
        """perform the actual registration"""
        # update to the initial position
        self.updateTY()
        
        if init_var is None:
            init_var = self.dist_matrix.sum() / (self.D * self.N * self.M)
        
        self.var = self.init_var = init_var
            
        self.w = weight
        
        self.Q_old = np.finfo(float).eps
        
        for i in range(maxiters):
            # do iterations
            delta_Q = self.update()
            if delta_Q <= tol:
                break
            self.iteration = i


class TranslationCPD(BaseCPD):
    
    def updateB(self):
        """Update the A and B matrices, this is the only method that needs to be overloaded"""
        self.B = np.eye(self.D)
        return self.B
        
    def __call__(self, tol=1e-6, maxiters=1000, init_var=None, weight=0, translation=None):
        """perform the actual registration"""
        if translation is None:
            translation = np.ones((1, self.D))
        self.translation = translation
        self.B = np.eye(self.D)
        
        # run the registration
        self.register(tol, maxiters, init_var, weight)
        
        return self.TY, self.translation, self.p_old
    
    
class RigidCPD(BaseCPD):
    
    def updateB(self):
        """Update the A and B matrices, this is the only method that needs to be overloaded"""
        self.B = self.A @ self.Yhat
        return self.B
        
    def __call__(self, tol=1e-6, maxiters=1000, init_var=None, weight=0, translation=None):
        """perform the actual registration"""
        if translation is None:
            translation = np.ones((1, self.D))
        self.translation = translation
        self.B = np.eye(self.D)
        
        # run the registration
        self.register(tol, maxiters, init_var, weight)
        
        return self.TY, self.translation, self.p_old
    
    
class SimilarityCPD(BaseCPD):
    
    def updateB(self):
        """Update the A and B matrices, this is the only method that needs to be overloaded"""
        self.B = np.eye(self.D)
        return self.B
        
    def __call__(self, tol=1e-6, maxiters=1000, init_var=None, weight=0, translation=None):
        """perform the actual registration"""
        if translation is None:
            translation = np.ones((1, self.D))
        self.translation = translation
        self.B = np.eye(self.D)
        
        # run the registration
        self.register(tol, maxiters, init_var, weight)
        
        return self.TY, self.translation, self.p_old
    
    
class AffineCPD(BaseCPD):
    
    def updateB(self):
        """Update the A and B matrices, this is the only method that needs to be overloaded"""
        a = self.Yhat.T @ np.diag(self.p_old.sum(1)) @ self.Yhat
        # solve B = self.A @ np.inv(a) == B @ a = self.A == a.T @ B.T = self.A.T
        # self.B = la.solve(a.T, self.A.T).T
        # a is a symmetric matrix
        self.B = la.solve(a, self.A.T).T
        return self.B
        
    def __call__(self, tol=1e-6, maxiters=1000, init_var=None, weight=0, translation=None, B=None):
        """perform the actual registration"""
        if translation is None:
            translation = np.ones((1, self.D))
        self.translation = translation
        
        if B is None:
            B = np.eye(self.D)
        self.B = B
        
        # run the registration
        self.register(tol, maxiters, init_var, weight)
        
        return self.TY, self.translation, self.p_old
