"""
Computation of the poserior probabilities of H1 (independence) and
H2 (dependence) about a given confusion matrix (N).
"""

from scipy.special import gammaln
import numpy as np


def log_betabinomial(n, k, a, b):
    """Beta-binomial distribution in the log-scale based on the gamma
    function (to improve numerical stability).
    
    See http://en.wikipedia.org/wiki/Beta-binomial_distribution
    """
    tmp0 = gammaln(n + 1) - (gammaln(k + 1) + gammaln(n - k + 1))
    tmp1 = gammaln(a + k) + gammaln(n + b - k)
    tmp2 = - gammaln(a + b + n) + gammaln(a + b) - (gammaln(a) + gammaln(b))
    return (tmp0 + tmp1 + tmp2).sum(0)


def betabinomial(n, k, a, b):
    """Beta-binomial distribution.
    """
    return np.exp(log_betabinomial(n, k, a, b))


def log_binomial(n, k):
    """Log of the binomial coefficiet based on the gamma function.

    Note that Gamma(n + 1) = n! when n is integer. The use of the log
    scale improves numerical stability.
    """
    return gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)


def binomial(n, k):
    """Stable binomial coefficient computation that handles both
    integers and floats.
    """
    return np.exp(log_binomial(n, k))


def notation(N):
    """Assign values according to Section 2.1.
    """
    N = np.array(N)
    [[n_11,n_12], [n_21,n_22]] = N
    n_1s = N[0].sum()
    n_2s = N[1].sum()
    m = N.sum()
    return n_11, n_22, n_12, n_21, n_1s, n_2s, m


def compute_likelihood_H1(N, alpha, beta):
    """Compute the integrated likelihood under H1 (indpendence).
    """
    n_11, n_22, n_12, n_21, n_1s, n_2s, m = notation(N)
    p_H = binomial(n_1s, n_11) * binomial(n_2s, n_21) / binomial(m , n_11 + n_21) * betabinomial(m, n_11 + n_21, alpha, beta)
    return p_H


def compute_likelihood_H2(N, alpha_1, beta_1, alpha_2, beta_2):
    """Compute the integrated likelihood under H2 (dependence).
    """
    n_11, n_22, n_12, n_21, n_1s, n_2s, m = notation(N)
    p_H = betabinomial(n_1s, n_11, alpha_1, beta_1) * betabinomial(n_2s, n_21, alpha_2, beta_2)
    return p_H


def compute_posteriors(N, p_H1=0.5, p_H2=0.5, alpha=1, beta=1, alpha_1=1, beta_1=1, alpha_2=1, beta_2=1):
    """Compute posterior probability for each hypotheses.
    """
    p_N_given_H2 = compute_likelihood_H2(N, alpha_1=alpha_1, beta_1=beta_1, alpha_2=alpha_2, beta_2=beta_2)
    p_N_given_H1 = compute_likelihood_H1(N, alpha=alpha, beta=beta)    
    p_H2_given_N = (p_N_given_H2 * p_H2)/((p_N_given_H1 * p_H1)+(p_N_given_H2 * p_H2))
    p_H1_given_N = 1 - p_H2_given_N
    return p_H1_given_N, p_H2_given_N


if __name__ == '__main__':

    N = np.array([[ 1,  1],
                  [ 1,  1]])

    p_H1_given_N , p_H2_given_N = compute_posteriors(N)

    print "Confusion matrix:"
    print N
    print "p(H1|N) =", p_H1_given_N
    print "p(H2|N) =", p_H2_given_N
    
    








