
import sys
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import hdf5storage
import h5py
from scipy.stats import norm

question = sys.argv[1]

def berkan_ozdamar_21602353_hw3(question):
    if question == '1' :

        with h5py.File('hw3_data2.mat', 'r') as file:
            Xn = list(file['Xn'])
        with h5py.File('hw3_data2.mat', 'r') as file:
            Yn = list(file['Yn'])

        Xn = np.array(Xn)
        Yn = np.array(Yn)
        Xn = Xn.T
        Yn = Yn.flatten()

        print(np.shape(Xn))
        print(np.shape(Yn))

        # %%

        # Part A

        def ridge_regression(X, Y, lambdaa):
            weight = np.linalg.inv(X.T.dot(X) + lambdaa * (np.identity(np.shape(X)[1]))).dot(X.T).dot(Y)
            return weight

        # %%

        def R2(Xtrain, Ytrain, Xtest, Ytest, lambdaa):
            weight = ridge_regression(Xtrain, Ytrain, lambdaa)
            prediction = Xtest.dot(weight)
            R2 = (np.corrcoef(Ytest, prediction)[0, 1]) ** 2
            return R2

        # %%

        def k_fold_cross_validation(X, Y, lambdaaK, foldNum):
            size = np.shape(X)[0]
            index = int(size / foldNum)
            validX = np.zeros((100, np.shape(X)[1]))
            testX = np.zeros((100, np.shape(X)[1]))
            trainX = np.zeros((800, np.shape(X)[1]))
            validY = np.zeros(100)
            testY = np.zeros(100)
            trainY = np.zeros(800)

            r2_valid = []
            r2_test = []
            for i in range(foldNum):
                validStartindex = (i * index) % size
                validEndindex = (i + 1) * index % size
                testStartindex = (i + 1) * index % size
                testEndindex = (i + 2) * index % size
                trainStartindex = (i + 2) * index % size
                trainEndindex = (i + 10) * index % size

                if (validEndindex == 0):
                    validEndindex = 1000
                elif (testEndindex == 0):
                    testEndindex = 1000

                validX = X[validStartindex: validEndindex][:]
                validY = Y[validStartindex: validEndindex]
                testX = X[testStartindex: testEndindex][:]
                testY = Y[testStartindex: testEndindex]

                if (trainStartindex >= trainEndindex):
                    trainX[0: size - trainStartindex - 1][:] = X[trainStartindex: -1][:]
                    trainX[size - trainStartindex - 1: -1][:] = X[0: trainEndindex][:]
                    trainY[0: size - trainStartindex - 1] = Y[trainStartindex: -1]
                    trainY[size - trainStartindex - 1: -1] = Y[0: trainEndindex]
                else:
                    trainX = X[trainStartindex % size: trainEndindex % size][:]
                    trainY = Y[trainStartindex % size: trainEndindex % size]

                for lambdaa in lambdaaK:
                    r2_valid.append(R2(trainX, trainY, validX, validY, lambdaa))
                    r2_test.append(R2(trainX, trainY, testX, testY, lambdaa))

            return r2_valid, r2_test

        # %%

        K_FOLD = 10
        lambda_arr = np.logspace(0, 12, num=500, base=10)

        r2_valid, r2_test = k_fold_cross_validation(Xn, Yn, lambda_arr, K_FOLD)

        # %%

        # Get the mean of every 500th item in r2_valid and r2_test and store them in r2_valid_final and r2_test_final,
        # so we get the mean r2 valid and test values.

        r2_valid_final = np.zeros(500)
        r2_test_final = np.zeros(500)

        temp = 0
        for i in range(500):
            for j in range(K_FOLD):
                temp += r2_valid[i + j * 500]
            r2_valid_final[i] = np.mean(temp)
            temp = 0

        temp = 0
        for i in range(500):
            for j in range(K_FOLD):
                temp += r2_test[i + j * 500]
            r2_test_final[i] = np.mean(temp)
            temp = 0

        # %%

        # The optimal lambda value which makes R2 max for validation data
        max_r2_valid = max(r2_valid_final)
        for i in range(len(r2_valid_final)):
            if (r2_valid_final[i] == max_r2_valid):
                optimal_lambda_valid_index = i

        optimal_lambda_valid = lambda_arr[optimal_lambda_valid_index]
        print("The optimal lambda value for validation data is " + str(optimal_lambda_valid))

        # The optimal lambda value which makes R2 max for test data
        max_r2_test = max(r2_test_final)
        for i in range(len(r2_test_final)):
            if (r2_test_final[i] == max_r2_test):
                optimal_lambda_test_index = i

        optimal_lambda_test = lambda_arr[optimal_lambda_test_index]
        print("The optimal lambda value for test data is " + str(optimal_lambda_test))

        # %%

        fig_num = 0
        plt.figure(fig_num, figsize=(8, 8))
        t = np.logspace(0, 12, num=5000, base=10)
        plt.ylabel('R^2 values')
        plt.xlabel('lambda values')
        plt.title('Graph of R^2 values vs lambda values')
        plt.plot(r2_valid_final, color='r')
        plt.plot(r2_test_final, color='b')
        plt.grid()
        plt.legend(['validation data', 'test data', ])
        plt.xscale('log')
        plt.show(block=False)

        # %%

        # Part B

        def bootstrap(X, Y, lambdaa, number_iterations):
            size = np.shape(X)[0]
            x_bootstrap = np.zeros((np.shape(X)[0], np.shape(X)[1]))
            y_bootstrap = np.zeros(np.shape(Y))
            weights_bootstrap = np.zeros((100, 500))

            for i in range(number_iterations):
                for j in range(size):
                    x_bootstrap = np.array(x_bootstrap)
                    y_bootstrap = np.array(y_bootstrap)
                    index = np.random.randint(0, 1000, size=1)
                    x_bootstrap[j, :] = X[index, :]
                    y_bootstrap[j] = Y[index]
                weights_bootstrap[:, i] = ridge_regression(x_bootstrap, y_bootstrap, lambdaa)
                x_bootstrap = np.zeros((np.shape(X)[0], np.shape(X)[1]))
                y_bootstrap = np.zeros(np.shape(Y))
            weights_bootstrap = np.array(weights_bootstrap)
            weights_mean = np.mean(weights_bootstrap, axis=1)
            weights_std = np.std(weights_bootstrap, axis=1)
            return weights_mean, weights_std

        # %%

        weights_mean_OLS, weights_std_OLS = bootstrap(Xn, Yn, 0, 500)

        # %%

        fig_num += 1
        plt.figure(fig_num, figsize=(8, 8))
        plt.title('Graph of bootstrap weights (OLS)')
        plt.xlabel('i Value')
        plt.ylabel('Weight Value')
        plt.errorbar(np.arange(1, 101), weights_mean_OLS, 2 * weights_std_OLS, ecolor='r', elinewidth=0.5, capsize=2)
        plt.show(block=False)

        # %%

        # Part C

        weights_mean_ridge, weights_std_ridge = bootstrap(Xn, Yn, optimal_lambda_test, 500)

        # %%

        fig_num += 1
        plt.figure(fig_num, figsize=(8, 8))
        plt.title('Graph of bootstrap weights (Ridge)')
        plt.xlabel('i Value')
        plt.ylabel('Weight Value')
        plt.errorbar(np.arange(1, 101), weights_mean_ridge, 2 * weights_std_OLS, ecolor='r', elinewidth=0.5, capsize=2)
        plt.show(block=False)

        # %%



    elif question == '2' :

        with h5py.File('hw3_data3.mat', 'r') as file:
            pop1 = list(file['pop1'])
        with h5py.File('hw3_data3.mat', 'r') as file:
            pop2 = list(file['pop2'])

        pop1 = np.array(pop1)
        pop2 = np.array(pop2)
        pop1.flatten()
        pop2.flatten()

        print("Shape of pop1 is: ")
        print(np.shape(pop1))
        print("Shape of pop2 is: ")
        print(np.shape(pop2))

        # %%

        # Part A

        def bootstrap(data, number_iterations):
            np.random.seed(7)
            size = np.shape(data)[0]
            data_bootstrap = np.zeros(np.shape(data)[0])
            result_bootstrap = []

            for i in range(number_iterations):
                for j in range(size):
                    data_bootstrap = np.array(data_bootstrap)
                    index = np.random.randint(0, size, size=1)
                    data_bootstrap[j] = data[index]
                result_bootstrap.append(data_bootstrap)
                data_bootstrap = np.zeros(np.shape(data)[0])
            result_bootstrap = np.array(result_bootstrap)
            return result_bootstrap

        # %%

        def difference_of_means(pop1, pop2, number_iterations):
            pops = np.concatenate((pop1, pop2))
            pops_bootstrap = bootstrap(pops, number_iterations)
            pop1_bootstrap = []
            pop2_bootstrap = []
            for i in range(np.shape(pops_bootstrap)[1]):
                if (i < np.shape(pop1)[0]):
                    pop1_bootstrap.append(pops_bootstrap[:, i])
                else:
                    pop2_bootstrap.append(pops_bootstrap[:, i])
            pop1_bootstrap = np.array(pop1_bootstrap)
            pop2_bootstrap = np.array(pop2_bootstrap)
            mean1 = np.mean(pop1_bootstrap, axis=0)
            mean2 = np.mean(pop2_bootstrap, axis=0)
            diff_of_means = mean1 - mean2
            sigma = np.std(diff_of_means)
            mu = np.mean(diff_of_means)
            return diff_of_means, sigma, mu

        # %%

        diff_of_means, sigma_pops, mu_pops = difference_of_means(pop1, pop2, 10000)

        # %%

        fig_num = 0
        plt.figure(fig_num, figsize=(8, 8))
        plt.xlabel('Difference of means (x)')
        plt.ylabel('P(x)')
        plt.title('Difference of pop1 and pop2 means')
        plt.hist(diff_of_means, bins=60, density=True)
        plt.show(block=False)

        # %%

        def z_and_p_values(line, sigma, mu):
            z = (line - mu) / sigma
            p = 2 * (1 - norm.cdf(np.abs(z)))
            return z, p

        # %%

        pops_line = np.mean(pop1) - np.mean(pop2)
        z, p = z_and_p_values(pops_line, sigma_pops, mu_pops)
        print("z-value is : ")
        print(z)
        print("\n")
        print("Two side p-value is : ")
        print(p)

        # %%

        # Part B

        with h5py.File('hw3_data3.mat', 'r') as file:
            vox1 = list(file['vox1'])
        with h5py.File('hw3_data3.mat', 'r') as file:
            vox2 = list(file['vox2'])

        vox1 = np.array(vox1)
        vox2 = np.array(vox2)

        print("Shape of vox1 is: ")
        print(np.shape(vox1))
        print("Shape of vox2 is: ")
        print(np.shape(vox2))

        # %%

        def correlation(vox1, vox2, number_iterations, PartC=False):
            size = np.shape(vox1)[0]
            vox1_bootstrap = bootstrap(vox1, number_iterations)
            vox2_bootstrap = bootstrap(vox2, number_iterations)
            if (PartC == True):
                np.random.seed(15)
                np.random.shuffle(vox2_bootstrap)

            result_bootstrap = np.zeros(number_iterations)

            for i in range(number_iterations):
                result_bootstrap[i] = np.corrcoef(vox1_bootstrap[i], vox2_bootstrap[i])[0, 1]
            result_bootstrap = np.array(result_bootstrap)
            return result_bootstrap

        # %%

        cor_bootstrap = correlation(vox1, vox2, 10000)

        # %%

        sorted_cor_bootstrap = np.sort(cor_bootstrap)
        corr_mean = np.mean(sorted_cor_bootstrap)
        lowerPercentile = np.percentile(sorted_cor_bootstrap, 5 / 2)
        upperPercentile = np.percentile(sorted_cor_bootstrap, 95 + (5 / 2))
        print("Mean if correlation is: " + str(corr_mean))
        print("\n")
        print('95 Percent Confidence Interval : (%1.4f, %1.4f)' % (lowerPercentile, upperPercentile))

        # %%

        # Part C
        cor_bootstrap_c = correlation(vox1, vox2, 10000, True)

        # %%

        fig_num += 1
        plt.figure(fig_num, figsize=(8, 8))
        plt.yticks([])
        plt.xlabel('Correlation (y)')
        plt.ylabel('P(y)')
        plt.title('Correlation between vox1 and vox2')
        plt.hist(cor_bootstrap_c, bins=60, density=True)
        plt.show(block=False)

        # %%

        vox1 = np.array(vox1)
        vox2 = np.array(vox2)
        vox1 = vox1.flatten()
        vox2 = vox2.flatten()

        c_line = np.corrcoef(vox1, vox2)[0, 1]
        sigma_c = np.std(cor_bootstrap_c)
        mu_c = np.mean(cor_bootstrap_c)

        z = (c_line - mu_c) / sigma_c
        p = 1 - norm.cdf(z)
        print("z-value is : ")
        print(z)
        print("\n")
        print("p-value is : ")
        print(p)

        # %%

        # Part D

        with h5py.File('hw3_data3.mat', 'r') as file:
            building = list(file['building'])
        with h5py.File('hw3_data3.mat', 'r') as file:
            face = list(file['face'])

        building = np.array(building)
        face = np.array(face)

        print("Shape of building is: ")
        print(np.shape(building))
        print("Shape of face is: ")
        print(np.shape(face))

        # %%

        difference_in_means = []
        size = np.shape(face)[0]
        sample = []
        for i in range(10000):
            for j in range(size):
                choices = []
                index = np.random.randint(0, size, size=1)
                choices.append(building[index] - face[index])
                choices.append(face[index] - building[index])
                for k in range(2):
                    choices.append(0)
                chosenOne = int(np.random.randint(0, len(choices), size=1))
                sample.append(choices[chosenOne])
            difference_in_means.append(np.mean(sample))
            sample = []
        difference_in_means = np.array(difference_in_means)
        difference_in_means = difference_in_means.flatten()

        # %%

        fig_num += 1
        plt.figure(fig_num, figsize=(8, 8))
        plt.yticks([])
        plt.xlabel('Difference of means (x)')
        plt.ylabel('P(x)')
        plt.title('Difference of face and building means\n (Assume subject population is same)')
        plt.hist(difference_in_means, bins=60, density=True)
        plt.show(block=False)

        # %%

        mu_fb = np.mean(difference_in_means)
        sigma_fb = np.std(difference_in_means)
        fb_line = np.mean(building) - np.mean(face)

        z, p = z_and_p_values(fb_line, sigma_fb, mu_fb)
        print("z-value is : ")
        print(z)
        print("\n")
        print("Two side p-value is : ")
        print(p)

        # %%

        # Part E
        diff_of_means_fb, sigma_fb_2, mu_fb_2 = difference_of_means(face, building, 10000)

        # %%

        fig_num += 1
        plt.figure(fig_num, figsize=(8, 8))
        plt.yticks([])
        plt.xlabel('Difference in means (x)')
        plt.ylabel('P(x)')
        plt.title('Difference of face and building means\n (Assume subject population is distint)')

        plt.hist(diff_of_means_fb, bins=60, density=True)
        plt.show(block=False)

        # %%

        mu_fb_2 = np.mean(diff_of_means_fb)
        sigma_fb_2 = np.std(diff_of_means_fb)
        fb_line_2 = np.mean(building) - np.mean(face)

        z, p = z_and_p_values(fb_line_2, sigma_fb_2, mu_fb_2)
        print("z-value is : ")
        print(z)
        print("\n")
        print("Two side p-value is : ")
        print(p)

berkan_ozdamar_21602353_hw3(question)



