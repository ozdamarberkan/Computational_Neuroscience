
import sys

#Imports
import numpy as np
import math
import matplotlib.pyplot as plt

question = sys.argv[1]

def berkan_ozdamar_21602353_hw1(question):
    if question == '1' :

        A = np.array([[1, 0, -1, 2], [2, 1, -1, 5], [3, 3, 0, 9]])
        b = np.array([1, 4, 9])

        # Part a
        print('Part a')
        print('\n')
        # Since x_3 and x_4 are free variables, we will assign them random numbers with numpy.random
        x_3 = np.random.random()
        x_4 = np.random.random()

        x_n = np.array([x_3 - 2 * x_4, -x_3 - x_4, x_3, x_4])
        print('Solution for Ax = 0')
        print(x_n)
        print('\n')
        result = np.around(A.dot(x_n), 6)
        print('Confirming that Ax = 0')
        print(result)

        # Part b
        print('Part b')
        print('\n')
        # Since x_3 and x_4 are free variables, we will write solution set in terms of them. And since we are looking for
        # a particular solution, assigning x_3 and x_4 as 0 will make x_p = [1 2 0 0]^T. Then check whether A.x_p = b where
        # b is [1 4 9]^T
        x_3 = 0
        x_4 = 0

        print('Solution for Ax = b')
        x_p = np.array([1 + x_3 - 2 * x_4, 2 - x_3 - x_4, x_3, x_4])
        print(x_p)
        print('\n')
        print('Confirming that Ax = b')
        result = A.dot(x_p)
        print(result)

        # Part c
        print('Part c')
        print('\n')
        x_3 = np.random.random()
        x_4 = np.random.random()

        x_c = np.array([1 + x_3 - 2 * x_4, 2 - x_3 - x_4, x_3, x_4])

        print('Confirming that Ax = b')
        result = A.dot(x_c)
        print(result)

        # Part d
        print('Part d')
        u, s, v = np.linalg.svd(A)

        # Make sigma in the correct form which is 3x4 matrix
        sigma = np.zeros((3, 4))
        for i in range(len(sigma[:, 0])):
            for j in range(len(sigma[0])):
                if (i == j):
                    # Instead of zero, it assigns third sigma value as 2*10^-16, which makes problem when taking reciprocal.
                    # To resolve that issue, i have just assigned 0 instead 2*10^-16
                    if (s[i] > 10 ** -15):
                        sigma[i, j] = s[i]
                    else:
                        sigma[i, j] = 0

        # To find sigma_plus, take reciprocals of the non-zero values.
        sigma_plus = np.zeros((3, 4))
        for i in range(len(sigma[:, 0])):
            for j in range(len(sigma[0])):
                if (i == j and sigma[i, j] != 0):
                    sigma_plus[i, j] = 1 / sigma[i, j]

        print('U is:')
        print(u)
        print('V is:')
        print(v)
        print('sigma is:')
        print(sigma)

        A_pseudo = (v.T).dot(sigma_plus.T).dot(u.T)
        print('Pseudo inverse of A by SVD decomposition :')
        print(A_pseudo)
        print('\n')
        print('To see how accurate our pseudo inverse of A, is we check A.A_pseudo.A = A')
        print(A.dot(A_pseudo).dot(A))

        # Finally, find pseudo inverse of A with numpy.linalg.pinv(A)
        A_pseudo_2 = np.linalg.pinv(A)
        print('\n')
        print('Pseudo inverse of A with numpy.linalg.pinv(A)')
        print(A_pseudo_2)

        # Part e
        print('Part e')
        print('\n')
        # 1. Set free variables to zero
        x_3 = 0
        x_4 = 0
        x_sparsest = np.array([1 + x_3 - 2 * x_4, 2 - x_3 - x_4, x_3, x_4])

        print('Sparsest solution example for x is')
        print(x_sparsest)

        # 2. Set one of the free variables to zero. And set other free variable such that one of the pivots become 0.
        x_3 = 0
        x_4 = 1 / 2
        x_sparsest = np.array([1 + x_3 - 2 * x_4, 2 - x_3 - x_4, x_3, x_4])

        print('Sparsest solution example for x is')
        print(x_sparsest)

        # 3.Set free variables such that both pivots are zero.
        x_3 = 1
        x_4 = 1
        x_sparsest = np.array([1 + x_3 - 2 * x_4, 2 - x_3 - x_4, x_3, x_4])

        print('Sparsest solution example for x is')
        print(x_sparsest)

        # Part f
        print('Part f')
        print('\n')
        L2 = A_pseudo.dot(b)
        print('Least norm solution is:')
        print(L2)

    elif question == '2' :

        # Part a
        print('Part A')
        print('\n')

        def nChooseK(n, k):
            return math.factorial(n) / (math.factorial(k) * math.factorial(n - k))

        def bernoulliDist(n, k, p):
            return nChooseK(n, k) * (p ** k) * ((1 - p) ** (n - k))

        x = np.arange(0, 1.001, 0.001)
        likelihood_lang = bernoulliDist(869, 103, x)
        likelihood_notlang = bernoulliDist(2353, 199, x)

        index = np.arange(0, 1001)
        plt.xlim(0, 200)
        plt.bar(index, likelihood_lang)
        plt.xlabel('Probability * 1000')
        plt.ylabel('Likelihood')
        plt.title('Likelihood distribution of language involved tasks')
        plt.show(block=False)

        plt.xlim(0, 200)
        plt.bar(index, likelihood_notlang)
        plt.xlabel('Probability * 1000')
        plt.ylabel('Likelihood')
        plt.title('Likelihood distribution of tasks that do not involve language')
        plt.show(block=False)

        # Part b
        print('Part b')
        print('\n')

        max_l = np.amax(likelihood_lang)
        max_nl = np.amax(likelihood_notlang)

        max_l_prob = 0
        max_nl_prob = 0
        for i in range(len(likelihood_lang)):
            if (likelihood_lang[i] == max_l):
                # Dividing the i by 1000 since the probability is scaled by 1000.
                max_l_prob = i / 1000

            if (likelihood_notlang[i] == max_nl):
                max_nl_prob = i / 1000

        print(
            'The probability that maximizes the likelihood of language involving tasks: (probability, likelihood of that probability)')
        print(max_l_prob, max_l)
        print(
            'The probability that maximizes the likelihood of not language involving tasks: (probability, likelihood of that probability)')
        print(max_nl_prob, max_nl)

        # Part c
        print('Part c')
        print('\n')
        # x has 1001 values, and since it is given that uniformly distributed, prior P(X) = 1/1001
        prior = 1 / 1001

        normalizer_l = 0
        posterior_l = np.zeros(len(likelihood_lang))
        normalizer_nl = 0
        posterior_nl = np.zeros(len(likelihood_notlang))

        for i in range(len(likelihood_lang)):
            normalizer_l += likelihood_lang[i] * prior
            normalizer_nl += likelihood_notlang[i] * prior

        posterior_l[:] = likelihood_lang[:] * prior / normalizer_l
        posterior_nl[:] = likelihood_notlang[:] * prior / normalizer_nl

        plt.xlim(0, 200)
        plt.bar(index, posterior_l)
        plt.xlabel('Probability * 1000')
        plt.ylabel('Posterior')
        plt.title('Posterior distribution of language involved tasks')
        plt.show(block=False)

        plt.xlim(0, 200)
        plt.bar(index, posterior_nl)
        plt.xlabel('Probability * 1000')
        plt.ylabel('Posterior')
        plt.title('Posterior distribution of tasks that do not involve language')
        plt.show(block=False)

        def pdf_to_cdf(posterior):
            lowerbound = 0
            temp_min = 0
            upperbound = np.inf
            temp_max = np.inf
            cdf = np.zeros(len(posterior) + 1)
            for i in range(1, len(posterior) + 1):
                for j in range(i):
                    cdf[i] += posterior[j]
                    # Since i have iterated i by 1, when finding lowerbound and upperbound, i decrease 1 from i.
                if (cdf[i] >= 0.025 and lowerbound <= temp_min):
                    temp_min = cdf[i]
                    lowerbound = (i - 1) / 1000
                if (cdf[i] >= 0.975 and upperbound >= temp_max):
                    temp_max = cdf[i]
                    upperbound = (i - 1) / 1000
            return cdf, lowerbound, upperbound

        cdf_l, lower_l, upper_l = pdf_to_cdf(posterior_l)
        cdf_nl, lower_nl, upper_nl = pdf_to_cdf(posterior_nl)

        index2 = np.arange(0, 1002)
        plt.bar(index2, cdf_l)
        plt.xlabel('Probability * 1000')
        plt.ylabel('P(X < x|data)')
        plt.title('CDF of posterior distribution of tasks that involve language')
        plt.show(block=False)

        plt.bar(index2, cdf_nl)
        plt.xlabel('Probability * 1000')
        plt.ylabel('P(X < x|data)')
        plt.title('CDF of posterior distribution of tasks that do not involve language')
        plt.show(block=False)

        print('Confidence interval of x_l is (', lower_l, ',', upper_l, ')')
        print('Confidence interval of x_nl is (', lower_nl, ',', upper_nl, ')')

        # Part d
        print('Part d')
        print('\n')

        matrixPosterior_l = np.matrix(posterior_l)
        matrixPosterior_nl = np.matrix(posterior_nl)
        joint_dist = (matrixPosterior_l.T).dot(matrixPosterior_nl)

        plt.imshow(joint_dist)
        plt.xlabel('x_nl * 1000')
        plt.ylabel('x_l * 1000')
        plt.colorbar()
        plt.show(block=False)

        x_l_greater = 0
        x_l_notgreater = 0
        for i in range(len(joint_dist)):
            for j in range(len(joint_dist)):
                if (i > j):
                    x_l_greater += joint_dist[i, j]
                else:
                    x_l_notgreater += joint_dist[i, j]

        print('Sum of posteriors such that x_l > x_nl')
        print(x_l_greater)
        print('Sum of posteriors such that x_l <= x_nl')
        print(x_l_notgreater)

        # Part e
        print('Part e')
        print('\n')
        PROB_LANG = 0.5

        prob_lang_active = (max_l_prob * PROB_LANG) / ((max_l_prob * PROB_LANG) + max_nl_prob * (1 - PROB_LANG))
        print('P(language | activation) is :')
        print(prob_lang_active)

berkan_ozdamar_21602353_hw1(question)



