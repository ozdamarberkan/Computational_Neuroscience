
import sys
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import hdf5storage
import h5py
# For Part C and D
from sklearn.decomposition import FastICA
from sklearn.decomposition import NMF


question = sys.argv[1]

def berkan_ozdamar_21602353_hw4(question):
    if question == '1' :

        with h5py.File('hw4_data1.mat', 'r') as file:
            faces = list(file['faces'])

        faces = np.array(faces)
        # faces = faces.T
        print(np.shape(faces))

        # %%

        # Part A

        # A sample stimuli
        figure_num = 0
        plt.figure(figure_num)
        plt.title('Sample Face Image (15th image)')
        plt.xlabel('32 pixels')
        plt.ylabel('32 pixels')
        plt.imshow(faces[:, 15].reshape(32, 32).T, cmap=plt.cm.bone)

        plt.show(block=False)

        # %%

        def PCA(data, numberOfPC):
            data = data.T
            data = data - np.mean(data, axis=0)
            covarianceMatrix = np.dot(data.T, data)
            eigenvalues, eigenvectors = np.linalg.eig(covarianceMatrix)
            eigenvectors = eigenvectors.T
            indexs = np.argsort(eigenvalues)[::-1]
            eigenvectors_sorted = eigenvectors[indexs]
            eigenvalues_sorted = eigenvalues[indexs]

            # store first n eigenvectors
            eigenvectors_f = eigenvectors_sorted[0:numberOfPC]

            variance = []
            normalizer = np.sum(eigenvalues)
            for i in range(numberOfPC):
                variance.append(eigenvalues[i] / normalizer)
            result = np.dot(data, eigenvectors_f.T).dot(eigenvectors_f) + np.mean(data, axis=0)
            result = np.array(result)
            result = np.real(result)
            eigenvectors_f = np.real(eigenvectors_f)
            return result, variance, eigenvectors_f

        # %%

        pca_face100, var100, PC100 = PCA(faces, 100)
        pca_face25, var25, PC25 = PCA(faces, 25)

        # %%

        print(np.shape(pca_face25))

        # %%

        figure_num += 1
        plt.figure(figure_num)
        plt.plot(var100)
        plt.title('Proportion of Variance Explained by Each Principal Component')
        plt.xlabel('Principal Component')
        plt.ylabel('Proportion of Variance')
        plt.grid()
        plt.show(block=False)

        # %%

        figure_num += 1
        plt.figure(figure_num, figsize=(6, 6))
        for i in range(25):
            ax1 = plt.subplot(5, 5, -i + 25)
            ax1.imshow(PC25[i].reshape(32, 32).T, cmap=plt.cm.bone)
            ax1.set_yticks([])
            ax1.set_xticks([])
        plt.subplots_adjust(wspace=0.05, hspace=0.05, left=0, right=1, bottom=0, top=1)
        plt.show()

        # %%

        # Part B

        pca_face10, var10, PC10 = PCA(faces, 10)
        pca_face25, var25, PC25 = PCA(faces, 25)
        pca_face50, var50, PC50 = PCA(faces, 50)

        # %%

        figure_num += 1
        plt.figure(figure_num, figsize=(6, 6))
        for i in range(36):
            ax1 = plt.subplot(6, 6, i + 1)
            ax1.imshow(faces[:, i].reshape(32, 32).T, cmap=plt.cm.bone)
            ax1.set_yticks([])
            ax1.set_xticks([])
        plt.subplots_adjust(wspace=0.05, hspace=0.05, left=0, right=1, bottom=0, top=1)
        plt.show()

        # %%

        figure_num += 1
        plt.figure(figure_num, figsize=(6, 6))
        for i in range(36):
            ax1 = plt.subplot(6, 6, i + 1)
            ax1.imshow(pca_face10[i].reshape(32, 32).T, cmap=plt.cm.bone)
            ax1.set_yticks([])
            ax1.set_xticks([])
        plt.subplots_adjust(wspace=0.05, hspace=0.05, left=0, right=1, bottom=0, top=1)
        plt.show()

        # %%

        figure_num += 1
        plt.figure(figure_num, figsize=(6, 6))
        for i in range(36):
            ax1 = plt.subplot(6, 6, i + 1)
            ax1.imshow(pca_face25[i].reshape(32, 32).T, cmap=plt.cm.bone)
            ax1.set_yticks([])
            ax1.set_xticks([])
        plt.subplots_adjust(wspace=0.05, hspace=0.05, left=0, right=1, bottom=0, top=1)
        plt.show()

        # %%

        figure_num += 1
        plt.figure(figure_num, figsize=(6, 6))
        for i in range(36):
            ax1 = plt.subplot(6, 6, i + 1)
            ax1.imshow(pca_face50[i].reshape(32, 32).T, cmap=plt.cm.bone)
            ax1.set_yticks([])
            ax1.set_xticks([])
        plt.subplots_adjust(wspace=0.05, hspace=0.05, left=0, right=1, bottom=0, top=1)
        plt.show()

        # %%

        # the mean and standard deviation
        MSE_PCA10 = np.mean((pca_face10.T - faces) ** 2)
        std_PCA10 = np.std(np.mean((faces.T - pca_face10) ** 2, axis=1))
        MSE_PCA25 = np.mean((pca_face25.T - faces) ** 2)
        std_PCA25 = np.std(np.mean((faces.T - pca_face25) ** 2, axis=1))
        MSE_PCA50 = np.mean((pca_face50.T - faces) ** 2)
        std_PCA50 = np.std(np.mean((faces.T - pca_face50) ** 2, axis=1))

        # %%

        print('10 PCs:')
        print('mean of MSEs = %f' % MSE_PCA10)
        print('std of MSEs = % f' % std_PCA10)
        print('\n')
        print('25 PCs:')
        print('mean of MSEs = %f' % MSE_PCA25)
        print('std of MSEs = % f' % std_PCA25)
        print('\n')
        print('50 PCs:')
        print('mean of MSEs = %f' % MSE_PCA50)
        print('std of MSEs = % f' % std_PCA50)

        # %%

        # Part C

        # %%

        ica_component10 = FastICA(10)
        ica_component10.fit(faces.T)
        ica_component25 = FastICA(25)
        ica_component25.fit(faces.T)
        ica_component50 = FastICA(50)
        ica_component50.fit(faces.T)

        # %%

        figure_num += 1
        plt.figure(figure_num, figsize=(6, 3))
        for i in range(10):
            ax1 = plt.subplot(2, 5, i + 1)
            ax1.imshow(ica_component10.components_[i].reshape(32, 32).T, cmap=plt.cm.bone)
            ax1.set_yticks([])
            ax1.set_xticks([])
        plt.subplots_adjust(wspace=0.05, hspace=0.05, left=0, right=1, bottom=0, top=1)
        plt.show()

        # %%

        figure_num += 1
        plt.figure(figure_num, figsize=(6, 6))
        for i in range(25):
            ax1 = plt.subplot(5, 5, i + 1)
            ax1.imshow(ica_component25.components_[i].reshape(32, 32).T, cmap=plt.cm.bone)
            ax1.set_yticks([])
            ax1.set_xticks([])
        plt.subplots_adjust(wspace=0.05, hspace=0.05, left=0, right=1, bottom=0, top=1)
        plt.show()

        # %%

        figure_num += 1
        plt.figure(figure_num, figsize=(10, 5))
        for i in range(50):
            ax1 = plt.subplot(5, 10, i + 1)
            ax1.imshow(ica_component50.components_[i].reshape(32, 32).T, cmap=plt.cm.bone)
            ax1.set_yticks([])
            ax1.set_xticks([])
        plt.subplots_adjust(wspace=0.05, hspace=0.05, left=0, right=1, bottom=0, top=1)
        plt.show()

        # %%

        ica_face10 = ica_component10.fit_transform(faces).dot(ica_component10.mixing_.T) + ica_component10.mean_
        ica_face25 = ica_component25.fit_transform(faces).dot(ica_component25.mixing_.T) + ica_component25.mean_
        ica_face50 = ica_component50.fit_transform(faces).dot(ica_component50.mixing_.T) + ica_component50.mean_
        ica_face10 = ica_face10.T
        ica_face25 = ica_face25.T
        ica_face50 = ica_face50.T

        # %%

        figure_num += 1
        plt.figure(figure_num, figsize=(6, 6))
        for i in range(30):
            ax1 = plt.subplot(6, 5, i + 1)
            ax1.imshow(ica_face10[i].reshape(32, 32).T, cmap=plt.cm.bone)
            ax1.set_yticks([])
            ax1.set_xticks([])
        plt.subplots_adjust(wspace=0.05, hspace=0.05, left=0, right=1, bottom=0, top=1)
        plt.show()

        # %%

        figure_num += 1
        plt.figure(figure_num, figsize=(6, 6))
        for i in range(30):
            ax1 = plt.subplot(6, 5, i + 1)
            ax1.imshow(ica_face25[i].reshape(32, 32).T, cmap=plt.cm.bone)
            ax1.set_yticks([])
            ax1.set_xticks([])
        plt.subplots_adjust(wspace=0.05, hspace=0.05, left=0, right=1, bottom=0, top=1)
        plt.show()

        # %%

        figure_num += 1
        plt.figure(figure_num, figsize=(6, 6))
        for i in range(30):
            ax1 = plt.subplot(6, 5, i + 1)
            ax1.imshow(ica_face50[i].reshape(32, 32).T, cmap=plt.cm.bone)
            ax1.set_yticks([])
            ax1.set_xticks([])
        plt.subplots_adjust(wspace=0.05, hspace=0.05, left=0, right=1, bottom=0, top=1)
        plt.show()

        # %%

        # the mean and standard deviation
        MSE_ICA10 = np.mean((ica_face10.T - faces) ** 2)
        std_ICA10 = np.std(np.mean((faces.T - ica_face10) ** 2, axis=1))
        MSE_ICA25 = np.mean((ica_face25.T - faces) ** 2)
        std_ICA25 = np.std(np.mean((faces.T - ica_face25) ** 2, axis=1))
        MSE_ICA50 = np.mean((ica_face50.T - faces) ** 2)
        std_ICA50 = np.std(np.mean((faces.T - ica_face50) ** 2, axis=1))

        # %%

        print('10 ICs:')
        print('mean of MSEs = %f' % MSE_ICA10)
        print('std of MSEs = % f' % std_ICA10)
        print('\n')
        print('25 ICs:')
        print('mean of MSEs = %f' % MSE_ICA25)
        print('std of MSEs = % f' % std_ICA25)
        print('\n')
        print('50 ICs:')
        print('mean of MSEs = %f' % MSE_ICA50)
        print('std of MSEs = % f' % std_ICA50)

        # %%

        # Part D

        nmf_10 = NMF(10, solver="mu")
        nmf_component10 = nmf_10.fit_transform(faces.T + np.abs(np.min(faces.T)))

        nmf_25 = NMF(25, solver="mu")
        nmf_component25 = nmf_25.fit_transform(faces.T + np.abs(np.min(faces.T)))

        nmf_50 = NMF(50, solver="mu")
        nmf_component50 = nmf_50.fit_transform(faces.T + np.abs(np.min(faces.T)))

        # %%

        figure_num += 1
        plt.figure(figure_num, figsize=(6, 3))
        for i in range(10):
            ax1 = plt.subplot(2, 5, i + 1)
            ax1.imshow(nmf_10.components_[i].reshape(32, 32).T, cmap=plt.cm.bone)
            ax1.set_yticks([])
            ax1.set_xticks([])
        plt.subplots_adjust(wspace=0.05, hspace=0.05, left=0, right=1, bottom=0, top=1)
        plt.show()

        # %%

        figure_num += 1
        plt.figure(figure_num, figsize=(6, 6))
        for i in range(25):
            ax1 = plt.subplot(5, 5, i + 1)
            ax1.imshow(nmf_25.components_[i].reshape(32, 32).T, cmap=plt.cm.bone)
            ax1.set_yticks([])
            ax1.set_xticks([])
        plt.subplots_adjust(wspace=0.05, hspace=0.05, left=0, right=1, bottom=0, top=1)
        plt.show()

        # %%

        figure_num += 1
        plt.figure(figure_num, figsize=(10, 5))
        for i in range(50):
            ax1 = plt.subplot(5, 10, i + 1)
            ax1.imshow(nmf_50.components_[i].reshape(32, 32).T, cmap=plt.cm.bone)
            ax1.set_yticks([])
            ax1.set_xticks([])
        plt.subplots_adjust(wspace=0.05, hspace=0.05, left=0, right=1, bottom=0, top=1)
        plt.show()

        # %%

        nmf_face10 = nmf_component10.dot(nmf_10.components_) - np.abs(np.min(faces.T))
        nmf_face25 = nmf_component25.dot(nmf_25.components_) - np.abs(np.min(faces.T))
        nmf_face50 = nmf_component50.dot(nmf_50.components_) - np.abs(np.min(faces.T))
        # nmf_face10 = nmf_face10.T
        # nmf_face25 = nmf_face25.T
        # nmf_face50 = nmf_face50.T

        # %%

        figure_num += 1
        plt.figure(figure_num, figsize=(6, 6))
        for i in range(30):
            ax1 = plt.subplot(6, 5, i + 1)
            ax1.imshow(nmf_face10[i].reshape(32, 32).T, cmap=plt.cm.bone)
            ax1.set_yticks([])
            ax1.set_xticks([])
        plt.subplots_adjust(wspace=0.05, hspace=0.05, left=0, right=1, bottom=0, top=1)
        plt.show()

        # %%

        figure_num += 1
        plt.figure(figure_num, figsize=(6, 6))
        for i in range(30):
            ax1 = plt.subplot(6, 5, i + 1)
            ax1.imshow(nmf_face25[i].reshape(32, 32).T, cmap=plt.cm.bone)
            ax1.set_yticks([])
            ax1.set_xticks([])
        plt.subplots_adjust(wspace=0.05, hspace=0.05, left=0, right=1, bottom=0, top=1)
        plt.show()

        # %%

        figure_num += 1
        plt.figure(figure_num, figsize=(6, 6))
        for i in range(30):
            ax1 = plt.subplot(6, 5, i + 1)
            ax1.imshow(nmf_face50[i].reshape(32, 32).T, cmap=plt.cm.bone)
            ax1.set_yticks([])
            ax1.set_xticks([])
        plt.subplots_adjust(wspace=0.05, hspace=0.05, left=0, right=1, bottom=0, top=1)
        plt.show()

        # %%

        # the mean and standard deviation
        MSE_NMF10 = np.mean((nmf_face10.T - faces) ** 2)
        std_NMF10 = np.std(np.mean((faces.T - nmf_face10) ** 2, axis=1))
        MSE_NMF25 = np.mean((nmf_face25.T - faces) ** 2)
        std_NMF25 = np.std(np.mean((faces.T - nmf_face25) ** 2, axis=1))
        MSE_NMF50 = np.mean((nmf_face50.T - faces) ** 2)
        std_NMF50 = np.std(np.mean((faces.T - nmf_face50) ** 2, axis=1))

        # %%

        print('10 MFs:')
        print('mean of MSEs = %f' % MSE_NMF10)
        print('std of MSEs = % f' % std_NMF10)
        print('\n')
        print('25 MFs:')
        print('mean of MSEs = %f' % MSE_NMF25)
        print('std of MSEs = % f' % std_NMF25)
        print('\n')
        print('50 MFs:')
        print('mean of MSEs = %f' % MSE_NMF50)
        print('std of MSEs = % f' % std_NMF50)

    elif question == '2' :

        # %%

        # Part A

        def tuningCurves(A, x, mu, sigma):
            return A * np.exp(-((x - mu) ** 2) / (2 * (sigma ** 2)))

        # %%

        mu = np.arange(-10, 11)
        responses = []
        for i in range(len(mu)):
            responses.append(tuningCurves(1, np.linspace(-16, 17, 750), mu[i], 1))

        # %%

        fig_num = 0
        plt.figure(fig_num)
        plt.title('Tuning Curves of the Neurons')
        plt.xlabel('Stimulus')
        plt.ylabel('Response')
        for i in range(len(responses)):
            plt.plot(np.linspace(-16, 17, 750), responses[i])
        plt.show(block=False)

        # %%

        response_x = []
        for i in range(len(mu)):
            response_x.append(tuningCurves(1, -1, mu[i], 1))

        # %%

        fig_num += 1
        plt.figure(fig_num)
        plt.title('Population Response at Stimulus x = -1')
        plt.xlabel('Chosen Stimulus')
        plt.ylabel('Population Response')
        plt.plot(mu, response_x, marker='o')
        plt.show(block=False)

        # %%

        # Part B

        numberOfTrials = 200
        responses_B = []
        stimuli = []
        est_WTA = []
        error_WTA = []
        np.random.seed(7)
        for i in range(numberOfTrials):
            response_B = []
            random = 10 * np.random.random_sample() - 5
            stimuli.append(random)
            for k in range(len(mu)):
                response_B.append(tuningCurves(1, stimuli[i], mu[k], 1))
            response_B = response_B + np.random.normal(0, 0.05, 21)
            chosen_index = np.argmax(response_B)
            est_WTA.append(mu[chosen_index])
            error_WTA.append(np.abs(stimuli[i] - est_WTA[i]))
            responses_B.append(response_B)
        error_WTA_mean = np.mean(error_WTA)
        error_WTA_std = np.std(error_WTA)

        # %%

        fig_num += 1
        plt.figure(fig_num)
        plt.xlabel('Trials')
        plt.ylabel('Stimuli')
        plt.title('Scatter of Actual and Estimated Stimuli \n(Winner Take All Decoder)')
        x_index = np.arange(0, numberOfTrials)
        plt.scatter(x_index, stimuli, color='r', s=10)
        plt.scatter(x_index, est_WTA, color='skyblue', s=10)
        plt.legend(['actual', 'estimated'], loc='upper right')
        plt.show(block=False)

        # %%

        print('Mean of error:', error_WTA_mean)
        print('Standard deviation of error:', error_WTA_std)

        # %%

        # Part C

        def MLE_decoder(A, x, mu, sigma, response):
            loglikelihood = 0
            loglikelihoods = []
            for i in range(len(x)):
                for k in range(len(mu)):
                    loglikelihood += (response[k] - tuningCurves(A, x[i], mu[k], sigma)) ** 2
                loglikelihoods.append(loglikelihood)
                loglikelihood = 0
            min_index = np.argmin(loglikelihoods)
            est_stim = x[min_index]
            return est_stim

        # %%

        est_MLE = []
        error_MLE = []
        for i in range(len(responses_B)):
            est_MLE.append(float(MLE_decoder(1, np.linspace(-5, 5, 500), mu, 1, responses_B[i])))
            error_MLE.append(float(np.abs(stimuli[i] - est_MLE[i])))
        error_MLE_mean = np.mean(error_MLE)
        error_MLE_std = np.std(error_MLE)

        # %%

        fig_num += 1
        plt.figure(fig_num)
        plt.xlabel('Trials')
        plt.ylabel('Stimuli')
        plt.title('Scatter of Actual and Estimated Stimuli \n(MLE Decoder)')
        x_index = np.arange(0, numberOfTrials)
        plt.scatter(x_index, stimuli, color='r', s=10)
        plt.scatter(x_index, est_MLE, color='skyblue', s=10)
        plt.legend(['actual', 'estimated'], loc='upper right')
        plt.show(block=False)

        # %%

        print('Mean of error:', error_MLE_mean)
        print('Standard deviation of error:', error_MLE_std)

        # %%

        # Part D

        def MAP_decoder(A, x, mu, sigma, response):
            logPosterior = 0
            logPosteriors = []
            for i in range(len(x)):
                for k in range(len(mu)):
                    logPosterior += (response[k] - tuningCurves(A, x[i], mu[k], sigma)) ** 2
                logPosterior = (logPosterior / (2 * (sigma / 20) ** 2)) + (x[i] ** 2) / (2 * 2.5 ** 2)
                logPosteriors.append(logPosterior)
                logPosterior = 0
            min_index = np.argmin(logPosteriors)
            est_stim = x[min_index]
            return est_stim

        # %%

        est_MAP = []
        error_MAP = []
        for i in range(len(responses_B)):
            est_MAP.append(float(MAP_decoder(1, np.linspace(-5, 5, 500), mu, 1, responses_B[i])))
            error_MAP.append(float(np.abs(stimuli[i] - est_MAP[i])))
        error_MAP_mean = np.mean(error_MAP)
        error_MAP_std = np.std(error_MAP)

        # %%

        fig_num += 1
        plt.figure(fig_num)
        plt.xlabel('Trials')
        plt.ylabel('Stimuli')
        plt.title('Scatter of Actual and Estimated Stimuli \n(MAP Decoder)')
        x_index = np.arange(0, numberOfTrials)
        plt.scatter(x_index, stimuli, color='r', s=10)
        plt.scatter(x_index, est_MAP, color='skyblue', s=10)
        plt.legend(['actual', 'estimated'], loc='upper right')
        plt.show(block=False)

        # %%

        print('Mean of error:', error_MAP_mean)
        print('Standard deviation of error:', error_MAP_std)

        # %%

        # Part E

        sigmas = [0.1, 0.2, 0.5, 1, 2, 5]

        # %%

        numberOfTrials = 200
        responses_E = []
        stimuli_E = []
        est_MLE_E = []
        error_MLE_E = []
        errors_MLE_E = []
        np.random.seed(5)
        for i in range(numberOfTrials):
            response_E = []
            random = 10 * np.random.random_sample() - 5
            stimuli_E.append(random)
            error_MLE_E = []
            for k, sigma in enumerate(sigmas):
                response_E = (tuningCurves(1, stimuli_E[i], mu, sigma)) + np.random.normal(0, 0.05, 21)
                est_MLE_E.append(MLE_decoder(1, np.linspace(-5, 5, 500), mu, sigma, response_E))
                error_MLE_E.append(np.abs(stimuli_E[i] - float(est_MLE_E[i * 6 + k])))
                responses_E.append(response_E)
            errors_MLE_E.append(error_MLE_E)
        errors_MLE_E = np.array(errors_MLE_E)
        est_MLE_E = np.array(est_MLE_E)
        responses_E = np.array(responses_E)
        stimuli_E = np.array(stimuli_E)

        # %%

        errors_MLE_E_mean = []
        errors_MLE_E_std = []
        for i in range(len(sigmas)):
            error_MLE_E_mean = np.mean(errors_MLE_E[:, i])
            error_MLE_E_std = np.std(errors_MLE_E[:, i])
            print('sigma = %.1f' % sigmas[i])
            print('Mean of errors', error_MLE_E_mean)
            print('Standard deviation of errors ', error_MLE_E_std)
            print('\n')
            errors_MLE_E_mean.append(error_MLE_E_mean)
            errors_MLE_E_std.append(error_MLE_E_std)
        errors_MLE_E_mean = np.array(errors_MLE_E_mean)
        errors_MLE_E_std = np.array(errors_MLE_E_std)

        # %%

        fig_num += 1
        plt.figure(fig_num)
        plt.xlabel('Standard Deviation of Error')
        plt.ylabel('Mean Error')
        plt.title('Mean Error vs Standard Deviation of Error')
        plt.errorbar(sigmas, errors_MLE_E_mean, yerr=errors_MLE_E_std,
                     marker='o', markerfacecolor='r', ecolor='r')

        plt.show(block=False)


berkan_ozdamar_21602353_hw4(question)



