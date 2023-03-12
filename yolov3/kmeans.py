import functools
from typing import Union, Callable
import torch

from torch import Tensor


def iou_distances(boxes0: Tensor, boxes1: Tensor, out=None) -> Tensor:
    """
    :param boxes0: [N, (w, h)]
    :param boxes1: [M, (w, h)]
    :return: [N, M]
    """
    w0, h0 = boxes0.unbind(dim=1)
    w1, h1 = boxes1.unbind(dim=1)
    w0, h0 = w0[:, None], h0[:, None]
    iw = torch.minimum(w0, w1)
    ih = torch.minimum(h0, h1)

    a0 = torch.mul(w0, h0)  # area of boxes0
    a1 = torch.mul(w1, h1)  # area of boxes1
    out = torch.mul(iw, ih, out=out)  # area of intersection
    union = a0 + a1 - out + 1e-8

    # in case w or h is 0, add a small value to make sure this
    # function behaves consistent with other dists functions
    out.add_(1e-8)

    # dist = 1.0 - IoU
    out.div_(union).mul_(-1.0).add_(1.0)
    return out


def euclidean_distances(tensor1: Tensor, tensor2: Tensor, out=None) -> Tensor:
    """
    :param tensor1: [N, D]
    :param tensor2: [M, D]
    :return: [N, M]
    """
    t1_squared = tensor1.square().sum(dim=1, keepdim=True)
    t2_squared = tensor2.square().sum(dim=1, keepdim=False)
    out = torch.mm(tensor1, tensor2.T, out=out)  # cross term
    out.mul_(-2.0).add_(t1_squared).add_(t2_squared).sqrt_()
    return out


def _init_kmeans_plus_plus(X: Tensor, n_clusters: int, dist_fn: Callable) -> Tensor:
    """
    1. Randomly select the first centroid from the data points.
    2. For each data point compute its distance from the nearest, previously chosen centroid.
    3. Select the next centroid from the data points such that the probability of choosing a
       point as centroid is directly proportional to its distance from the nearest, previously
       chosen centroid. (i.e. the point having maximum distance from the nearest centroid is most
       likely to be selected next as a centroid)
    4. Repeat steps 2 and 3 until k centroids have been sampled
    """
    centroids = X.new_empty([n_clusters, X.size(1)])

    # 1. Randomly pick a point in X as first centroid
    i_ = torch.randint(X.size(0), [1])
    centroids[0] = X[i_]

    # compute remaining k-1 centroids
    for i in range(1, n_clusters):
        # compute distances of all data points from previously chosen centroids,
        # then find the nearest centroid for each data point and choose the
        # data point which has the largest distance from the nearest centroids
        # as the next centroid
        distance_matrix = dist_fn(X, centroids[:i])
        min_distances = distance_matrix.min(dim=1)[0]
        i_ = min_distances.argmax(dim=0)
        centroids[i] = X[i_]

    return centroids


def _init_random(X: Tensor, n_clusters: int) -> Tensor:
    indices = torch.randperm(X.size(0))[:n_clusters]
    centroids = X.index_select(0, indices)
    return centroids


def _run_kmeans_once(X: Tensor,
                     centroids: Tensor,
                     centroids_new: Tensor,
                     labels: Tensor,
                     centroids_shift: Tensor,
                     dists: Tensor,
                     dists_fn: Callable):
    dists_fn(X, centroids, out=dists)
    torch.argmin(dists, dim=1, out=labels)

    for k in range(centroids.size(0)):
        selected_idxs = torch.nonzero(labels == k).view(-1)
        X_assigned_k = X.index_select(0, selected_idxs)
        if X_assigned_k.size(0) == 0:
            X_assigned_k = X[torch.randint(X.size(0), [1])]
        centroids_new[k] = X_assigned_k.mean(dim=0)
    torch.linalg.norm(centroids_new - centroids, dim=1, out=centroids_shift)


class KMeans(object):
    """
    Simple yet fast pytorch implementation of standard kmeans algorithm.

    The kmeans algorithm works as follows:
        1. First initialize `k` cluster centers randomly;
        2. Then categorize each item from inputs to its closest mean and
           update the mean's coordinates, which are the averages of the
           items categorized in that cluster so far;
        3. Repeat the process for a given number of iterations or until
           the Frobenius norm of the difference in the cluster centers of
           two consecutive iterations is less than some tolerance.
    """

    def __init__(self, n_clusters: int,
                 *,
                 dist_fn: Callable = euclidean_distances,
                 init: str = "k-means++",
                 max_iter: int = 300,
                 tol: float = 1e-4,
                 verbose: bool = False):
        assert n_clusters > 0, f"#clusters must be positive, got {n_clusters}"
        assert callable(dist_fn), type(dist_fn)
        if init == "k-means++":
            self.init_fn = functools.partial(_init_kmeans_plus_plus, dist_fn=dist_fn)
        elif init == "random":
            self.init_fn = _init_random
        else:
            msg = f"`init` must be 'k-means++', 'random' or callable."
            raise RuntimeError(msg)
        self.n_clusters = n_clusters
        self.init = init
        self.dist_fn = dist_fn
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose

        # result tensors
        self.cluster_centers_ = None
        self.labels_ = None
        self.n_iter_ = None
        self.inertia_ = None

    @staticmethod
    def _inertia(X, labels, centroids):
        assigned_clusters = centroids[labels]
        inertia = (X - assigned_clusters).square().sum()
        return inertia

    @torch.no_grad()
    def fit(self, X: Tensor, centroids: Tensor = None):
        assert X.dim() == 2, "kmeans requires input a 2D tensor."
        verbose = self.verbose
        tol = self.tol
        dist_fn = self.dist_fn
        X = X.float()

        if centroids is None:
            centroids = self.init_fn(X, self.n_clusters)
            if verbose: print("Initialization complete")

        centroids_new = torch.zeros_like(centroids)
        labels = X.new_full([X.size(0)], -1, dtype=torch.int64)
        labels_old = labels.clone()
        centroids_shift = X.new_zeros([self.n_clusters])
        dist_matrix = X.new_zeros([X.size(0), self.n_clusters])

        strict_convergence = False
        for self.n_iter_ in range(1, self.max_iter + 1):
            _run_kmeans_once(
                X,
                centroids,
                centroids_new,
                labels,
                centroids_shift,
                dist_matrix,
                dist_fn
            )

            if verbose:
                inertia = self._inertia(X, labels, centroids)
                print(f"Iteration {self.n_iter_}, inertia {inertia}.")

            centroids, centroids_new = centroids_new, centroids

            if torch.equal(labels, labels_old):
                if verbose:
                    print(f"Converged at iteration {self.n_iter_}: strict convergence.")
                strict_convergence = True
                break
            else:
                centroids_shift_tol = centroids_shift.square().sum()
                if centroids_shift_tol <= tol:
                    if verbose:
                        print(
                            f"Converged at iteration {self.n_iter_}: center shift "
                            f"{centroids_shift_tol} within tolerance {tol}."
                        )
                    break

            labels_old.copy_(labels)

        if not strict_convergence:
            _run_kmeans_once(
                X,
                centroids,
                centroids,
                labels,
                centroids_shift,
                dist_matrix,
                self.dist_fn,
            )

        self.inertia_ = self._inertia(X, labels, centroids)
        self.cluster_centers_ = centroids
        self.labels_ = labels
        return self

    def predict(self, X: Tensor, centroids: Tensor = None):
        assert X.dim() == 2, "Input require a 2D tensor."
        if centroids is None and self.cluster_centers_ is None:
            msg = "run kmeans on X first"
            raise RuntimeError(msg)
        if centroids is None:
            centroids = self.cluster_centers_

        X = X.to(centroids)
        dist_mat = self.dist_fn(X, centroids)
        labels = torch.argmin(dist_mat, dim=1)
        return labels

    def fit_predict(self, X: Tensor, centroids: Tensor = None):
        return self.fit(X, centroids).labels_


if __name__ == '__main__':
    X = torch.tensor([[1, 2], [1, 4], [1, 5],
                      [10, 2], [10, 4], [10, 5]], device="cuda")

    kmeans = KMeans(2, dist_fn=euclidean_distances, verbose=True)
    kmeans.fit(X)

    print("cluster centers:", kmeans.cluster_centers_)
    print("labels:", kmeans.labels_)
    print("inertia:", kmeans.inertia_)
    print("n_iter:", kmeans.n_iter_)
    print("predict:", kmeans.predict(torch.tensor([[0, 0], [12, 3]])))
    print()

    from sklearn.cluster import KMeans as KMeansCPU

    kmeans = KMeansCPU(2, n_init="auto", verbose=True)
    kmeans.fit(X.cpu().numpy())

    print("cluster centers:", kmeans.cluster_centers_)
    print("labels:", kmeans.labels_)
    print("inertia:", kmeans.inertia_)
    print("n_iter:", kmeans.n_iter_)
    print("predict:", kmeans.predict([[0, 0], [12, 3]]))
