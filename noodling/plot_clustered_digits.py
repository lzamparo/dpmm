"""
A demo of GMM (eventually versus DPMM) clustering of hand drawn digits data
"""
print(__doc__)

from time import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn import metrics
from sklearn.mixture import GMM
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

np.random.seed(42)

digits = load_digits()
data = scale(digits.data)

n_samples, n_features = data.shape
n_digits = len(np.unique(digits.target))
labels = digits.target

sample_size = 300

def make_ellipses(gmm, ax, n_components):
    for n in xrange(n_components):
        v, w = np.linalg.eigh(gmm._get_covars()[n][:2, :2])
        u = w[0] / np.linalg.norm(w[0])
        angle = np.arctan2(u[1], u[0])
        angle = 180 * angle / np.pi  # convert to degrees
        v *= 9
        ell = mpl.patches.Ellipse(gmm.means_[n, :2], v[0], v[1],
                                  180 + angle, color=plt.cm.spectral(n / 10.))
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(0.5)
        ax.add_artist(ell)

print("n_digits: %d, \t n_samples %d, \t n_features %d"
      % (n_digits, n_samples, n_features))


print(79 * '_')
print('% 9s' % 'init'
      '    time  inertia    homo   compl  v-meas     ARI AMI  silhouette')


def bench_gmm(estimator, name, data):
    t0 = time()
    estimator.fit(data)
    logprob = estimator.score(data)
    predicted_labels = estimator.predict(data)
    print('% 9s   %.2fs    %i   %.3f   %.3f   %.3f   %.3f   %.3f    %.3f'
          % (name, (time() - t0), logprob.sum(),
             metrics.homogeneity_score(labels, predicted_labels),
             metrics.completeness_score(labels, predicted_labels),
             metrics.v_measure_score(labels, predicted_labels),
             metrics.adjusted_rand_score(labels, predicted_labels),
             metrics.adjusted_mutual_info_score(labels,  predicted_labels),
             metrics.silhouette_score(data, predicted_labels,
                                      metric='euclidean',
                                      sample_size=sample_size)))

#bench_gmm(GMM(n_components=n_digits, covariance_type='tied', init_params='wmc', n_init=20, n_iter=100),
              #name="GMM tied", data=data)

#bench_gmm(GMM(n_components=n_digits, covariance_type='full', init_params='wmc', n_init=20, n_iter=100),
              #name="GMM full", data=data)

print(79 * '_')

###############################################################################
# Visualize the results on PCA-reduced data using colours and ellipses

reduced_data = PCA(n_components=2).fit_transform(data)
gmm = GMM(n_components=n_digits, covariance_type='full', init_params = 'wc', n_init=20, n_iter=100)
# Cheat by initializing the means to the means of the labled data points
gmm.means_ = np.array([reduced_data[digits.target == i].mean(axis=0)
                                  for i in xrange(n_digits)])
t0 = time()
gmm.fit(reduced_data)
print("Model fitting done in %.3f" % (time() - t0))

plt.figure(1)
plt.clf()
h = plt.subplot(1,1,1)

make_ellipses(gmm,h,n_digits)

for n in range(n_digits):
    digit_data = reduced_data[digits.target == n]
    plt.scatter(digit_data[:, 0], digit_data[:, 1], 0.8, color=plt.cm.spectral(n / 10.),
                    label=digits.target_names[n])

# Plot the means as a white X
centroids = gmm.means_
plt.scatter(centroids[:, 0], centroids[:, 1],
            marker='x', s=169, linewidths=3,
            color='w', zorder=10)

plt.title('Gaussian Mixture Model clustering on the digits dataset (PCA-reduced data)\n'
          'Means are marked with white cross')
plt.xticks(())
plt.yticks(())
plt.show()
