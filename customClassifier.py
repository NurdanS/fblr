import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.extmath import randomized_svd
from sklearn.preprocessing import label_binarize
import math
import copy
import threading
import warnings


class FastLogisticRegressionLowRank(BaseEstimator, ClassifierMixin):
    rank = None

    verbose = 0
    def __init__(self, epsilon=1e-10, lambda_ssr=0, f=0, gamma=0, energyPercentile=99.9999,
                 convergenceTolerance=1e-3, minimumIteration=2, maximumIteration=10, fit_intercept=True,
                 multi_class=None, n_jobs=1):
        self.epsilon = epsilon
        self.lambda_ssr = lambda_ssr
        self.f = f
        self.gamma = gamma
        self.energyPercentile = energyPercentile
        self.convergenceTolerance = convergenceTolerance
        self.minimumIteration = minimumIteration
        self.maximumIteration = maximumIteration
        self.fit_intercept = fit_intercept;
        self.multi_class = multi_class
        self.classes_ = None

    def fit(self, X, y, sample_weight=None):

        X, y = check_X_y(X, y)

        self.classes_ = unique_labels(y)

        # Call your existing implementation here
        w = self.fastLogisticRegressionLowRank(X, y)
        self.intercept_ = w[0] if self.fit_intercept else None
        self.coef_ = w[1:]

        self.is_fitted_ = True

        return self

    def predict(self, X):
        y_proba = self.predict_proba(X)
        predictions = np.argmax(y_proba, axis=1)

        return predictions

    def predict_proba(self, X):
        # Check if fit has been called
        check_is_fitted(self, ['coef_', 'classes_'])

        if (self.fit_intercept):
            X = np.hstack((np.ones((X.shape[0], 1)), X))

        w = np.hstack((self.intercept_, self.coef_)).reshape(1, -1) if self.fit_intercept else self.coef_
        y_hat = X.dot(w.T)
        e = np.exp(np.minimum(y_hat, 100))
        p = (e / (1.0 + e))
        p = p.reshape(-1, 1)
        y_proba = np.hstack((1 - p, p))

        return y_proba

    def get_params(self, deep=True):
        return {
            'epsilon': self.epsilon,
            'lambda_ssr': self.lambda_ssr,
            'f': self.f,
            'gamma': self.gamma,
            'energyPercentile': self.energyPercentile,
            'convergenceTolerance': self.convergenceTolerance,
            'minimumIteration': self.minimumIteration,
            'maximumIteration': self.maximumIteration
        }

    def set_params(self, **params):
        for param, value in params.items():
            setattr(self, param, value)
        return self

    def computeU(self, X_train, S, Vh):
        s_inverse = (1 / S).flatten()
        # U = X_train.dot((Vh.T).dot(np.diag(s_inverse)))
        U = X_train.dot(np.multiply(s_inverse, Vh.T))
        return U

    def fastLogisticRegressionLowRank(self, X_train, y_train):
        randomSeedValue = 12345
        np.random.seed(randomSeedValue)

        n = X_train.shape[0]
        d = X_train.shape[1] + (1 if self.fit_intercept else 0)
                
        doDataReduction = None
        doFeatureReduction = None
            
        if (self.fit_intercept):
            X_train = np.hstack((np.ones((X_train.shape[0], 1)), X_train))

                           
        lowRankDataCountThreshold = max(d * 100, 100000)   
        doDataReduction = (n > 2 * lowRankDataCountThreshold)
        
        d_limit = 500
        r_hat = min(math.sqrt(d*d_limit), d)
        T = 0.90
        doFeatureReduction = ((n*d**2) /(n*d*r_hat+n*r_hat**2+r_hat**2*d)) <= 2*T
    
        X_train_subset = X_train[:lowRankDataCountThreshold, ] if doDataReduction else X_train

        if (doFeatureReduction):
            d_hat = min((int)(n/50),d)
            [_, S, Vh] = randomized_svd(X_train_subset, n_components = d_hat, random_state=randomSeedValue)
            U = self.computeU(X_train, S, Vh)
        else:
            [U, S, Vh] = np.linalg.svd(X_train_subset, full_matrices=False, compute_uv=True)          
            
        # Low-rank approximation
        r = self.determineRank(S)
        S = S[:r]
        Vh = Vh[:r, :]
        FastLogisticRegressionLowRank.rank = r

        if (FastLogisticRegressionLowRank.verbose):
            print("Rank = ", r)

        if (doDataReduction):
            S *= math.sqrt((n-1) / (lowRankDataCountThreshold-1))    
            U = self.computeU(X_train, S, Vh)
        else:
            U = U[:, :r]

            
        if (FastLogisticRegressionLowRank.verbose):
            print("Data Reduction = ", doDataReduction, ",    Feature Reduction = ", doFeatureReduction)
            
       
        F = np.multiply(1 / S, Vh.T).dot(U.T)
        
        # w_0 = Fy
        w_0 = F.dot(y_train)
     
        log2 = math.log(2.0)
        t = log2
        q = 0.5

        doRegularization = (self.lambda_ssr > 0 or self.gamma > 0)
        if (doRegularization):          
            I = np.identity(d)
            # G = S V'
            G = np.multiply(Vh, S.reshape(-1, 1))
            # v = (1/n)(G' U'(y_train - q))
            v = (1 / n) * (G.T).dot((U.T).dot(y_train - q))
            
            p = np.ones(d)
            if (self.fit_intercept):
                p[0] = 0
        else:           
            # y_q = U(U'(y_train - q))
            y_q = U.dot(((U.T).dot(y_train - q)))
            
        w = copy.deepcopy(w_0)
        for iteration in range(self.maximumIteration + 1):
            w_hat = copy.deepcopy(w)
            
            o = X_train.dot(w_hat)

            # o_clip = np.clip(o, -40, 40)
            # z = (np.log(1.0 + np.exp(o_clip)) - log2 - 0.5*o_clip) / (o_clip*o_clip + epsilon)
            z = (np.log(1.0 + np.exp(np.minimum(o, 100))) - log2 - 0.5 * o) / (o * o + self.epsilon)
            
            # weight update
            if (doRegularization):
                h = p / (np.abs(w_hat) ** (2 - self.f) + self.epsilon)
                H = np.diag(h.flatten())

                # Z = np.diag(z.flatten())
                # A = (2/n)*(X_train.T).dot(Z).dot(X_train) + (lambda_ssr/d)*I + (gamma/d)*H
                # A = (2/n)*(G.T).( (U.T).dot(Z).dot(U) ).dot(G) + (lambda_ssr/d)*I + (gamma/d)*H
                A = (2 / n) * (G.T).dot(np.multiply(z, U.T).dot(U)).dot(G) + (self.lambda_ssr / d) * I + (
                            self.gamma / d) * H
                b = (self.lambda_ssr / d) * w + v

                w = np.linalg.solve(A, b)
                
            else:
                # w = (1/2)*(F Z^-1 y_q)
                # w = (1/2)*np.multiply(1/z, F).dot(y_q)
                w = (1 / 2) * F.dot(y_q / z)
                
            change = np.max(np.abs(w - w_hat))
            if (iteration >= self.minimumIteration and change <= self.convergenceTolerance):
                break

        return w
       

         
    def determineRank(self, S):
        d = S.shape[0]

        percentile = self.energyPercentile / 100.0
        logS = np.log(S + 1.0)
        cdf = np.cumsum(logS) / np.sum(logS)
        r = np.min(np.where(np.logical_and(cdf > percentile, logS > 1e-10))) + 1
        r = max(min(r, d), 1)

        return r

    def fastLogisticRegression(X_train, y_train, fit_intercept=True, epsilon=1e-10, lambda_ssr=0, f=0, gamma=0,
                               convergenceTolerance=1e-3, minimumIteration=2, maximumIteration=10):
        if (FastLogisticRegressionLowRank.verbose):
            costList = []

        if (fit_intercept):
            X_train = np.hstack((np.ones((X_train.shape[0], 1)), X_train))

        n = X_train.shape[0]
        d = X_train.shape[1]

        I = np.identity(d)

        beta = 1e-8
        A = (X_train.T).dot(X_train) + beta * I
        b = (X_train.T).dot(y_train)
        w_0 = np.linalg.solve(A, b)

        doRegularization = (lambda_ssr > 0 or gamma > 0)

        log2 = math.log(2.0)
        t = log2
        q = 0.5

        v = (1 / n) * (X_train.T).dot(y_train - q)

        w = copy.deepcopy(w_0)
        for iteration in range(maximumIteration + 1):
            w_hat = copy.deepcopy(w)

            o = X_train.dot(w_hat)
            z = (np.log(1.0 + np.exp(o)) - log2 - 0.5 * o) / (o * o + epsilon)

            # calculate cost
            if (FastLogisticRegressionLowRank.verbose):
                cost = (1 / n) * np.sum(o * (z * o + (q - y_train))) + t
                costList.append(cost.item())

            # weight update
            if (doRegularization):
                p = np.ones(d)
                p[0] = 0

                h = p / (np.abs(w_hat) ** (2 - f) + epsilon)
                H = np.diag(h.flatten())

                # A = (2/n) * (X_train.T Z X_train) + (lambda_ssr/d)*I + (gamma/d)*H
                A = (2 / n) * np.multiply(z, X_train.T).dot(X_train) + (lambda_ssr / d) * I + (gamma / d) * H
                b = (lambda_ssr / d) * w + v
                w = np.linalg.solve(A, b)
            else:
                # A = (2/n) * (X_train.T Z X_train)
                A = (2 / n) * np.multiply(z, X_train.T).dot(X_train)
                w = np.linalg.solve(A, v)

            change = np.max(np.abs(w - w_hat))
            if (iteration >= minimumIteration and change <= convergenceTolerance):
                break

        if (FastLogisticRegressionLowRank.verbose):
            return w, w_0, costList
        else:
            return w

