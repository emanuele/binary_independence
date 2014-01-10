import numpy as np
from bayesian_binary_test import compute_posteriors

if __name__ == '__main__':

    print "assumption 1: uninformative priors for the beta distribution."
    [alpha, beta] = [1., 1.]
    [alpha_1, beta_1] = [1., 1.]
    [alpha_2, beta_2] = [1., 1.]

    print "assumption 2: equal prior probability for each hypothesis."
    p_H1 = 0.5
    p_H2 = 1 - p_H1

    print
    print "Confusion matrices from the example in Section 1."
    M_1 = np.array([[65, 5],
                    [30, 0]])
    
    M_2 = np.array([[40, 30],
                    [5, 25]])

    for N in [M_1, M_2]:
        print "Confusion Matrix:"
        print N

        p_H1_given_N, p_H2_given_N = compute_posteriors(N, p_H1, p_H2, alpha, beta, alpha_1, beta_1, alpha_2, beta_2)

        print "Computing the posterior probabilities:"

        print 'Posterior Probability p(H2|N):', np.around(p_H2_given_N, decimals = 4)
        print 'Posterior Probability p(H1|N):', np.around(p_H1_given_N, decimals = 4)
        print
        

