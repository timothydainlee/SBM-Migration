import numpy as np
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, cophenet


class NMF:
    """
    Non-negative Matrix Factorization class (accounts NaN)

    Parameters:
        num_components: int, default=2
            number of components.
        num_iter: int, default=100
            number of NMF iterations for NMF statistics.
        nmf_iter: int, default=1000
            number of NMF iterations.

    Attributes:
        correlation_coefficient: float
        consensus: np.ndarray of shape (num_input, num_input)

    Methods:
        fit(x, init_w=None, init_h=None): fit model to data. x should be np.ndarray.
            init_w: np.ndarray, default=None
                non-negative w of shape (num_input, num_components)
            init_h: np.ndarray, default=None
                non-negative h of shape (num_components, num_featurs * 2)

    Examples:
        nmf = NMF(x)

    Raises:
        TypeError: if not np.ndarray.
    """
    def __init__(self, num_components=2,
                       num_iter=100,
                       nmf_iter=1000):
        self.num_components = num_components
        self.num_iter = num_iter
        self.nmf_iter = nmf_iter

    def fit(self, x, init_w=None, init_h=None):
        if not isinstance(x, np.ndarray):
            raise TypeError(f"x must be np.ndarray not {type(x)}.")

        x = np.hstack([x, -x])

        x[x < 0] = 0
        mx, _ = x.shape
        c = np.zeros([mx, mx])

        for _ in range(self.num_iter):
            wt, _ = self._nmf(x, init_w, init_h)
            ct = np.zeros([mx, mx])

            idx = np.argmax(wt, axis=1)
            for j in range(mx):
                for k in range(mx):
                    if idx[j] == idx[k]:
                        ct[j, k] = 1
            c += ct

        consensus = c / self.num_iter
        y = pdist(consensus)
        z = linkage(consensus, method="average")

        coph_dist = cophenet(z)
        correlation_coefficient = np.corrcoef(y, coph_dist)[0, 1]

        self.correlation_coefficient = correlation_coefficient
        self.consensus = consensus


    def _nmf(self, x, init_w, init_h):
        mx, nx = x.shape

        if init_w is not None:
            w = init_w.astype("float64")
        else:
            w = np.random.uniform(size=[mx, self.num_components])

        if init_h is not None:
            h = init_h.astype("float64")
        else:
            h = np.random.uniform(size=[self.num_components, nx])
        eps =  np.finfo("float64").eps

        has_nan = np.isnan(np.sum(x))
        if has_nan:
            nan_idx = np.isnan(x)
            x = np.nan_to_num(x)
            mul1 = mx / np.sum(~nan_idx, 0)[:, np.newaxis]
            mul2 = nx / np.sum(~nan_idx, 1)[np.newaxis, :]
        else:
            mul1 = np.zeros([1, 1])
            mul2 = np.zeros([1, 1])

        for _ in range(self.nmf_iter):
            if has_nan:
                h = h * np.sqrt((w.T @ x * mul1.T + eps) / (w.T @ w @ h + eps))
                w = w * np.sqrt((x @ h.T * mul2.T + eps) / ((w @ h) @ (x.T @ w * mul1) + eps))
                w = w / np.sum(w, axis=0)
            else:
                h = h * np.sqrt((w.T @ x + eps) / (w.T @ w @ h + eps))
                w = w * np.sqrt((x @ h.T + eps) / ((w @ h) @ (x.T @ w) + eps))
                w = w / np.sum(w, axis=0)

        return w, h


class PLS:
    """
    Partial Least Square class (single block, accounts NaN)

    Parameters:
        num_components: int, default=2
            number of components.
        center: bool, default=True
            whether to mean-center x and y.
        standardize : bool, default=False
            whether to standardize x and y.
        max_iter: int, default=500
            maximum iteration for NIPALS algorithm.
        tol: float, default=1e-10
            tolerance convergence criteria for NIPALS algorithm.

    Attributes:
        x_loading: np.ndarray of shape (num_features, num_components)
        y_loading: np.ndarray of shape (num_targets, num_components)
        x_weight: np.ndarray of shape (num_features, num_components) y_weight: np.ndarray of shape (num_targets, num_components) x_score: np.ndarray of shape (num_samples, num_components)
        y_score: np.ndarray of shape (num_samples, num_components)
        rel_coeff: np.ndarray of shape (1, num_components)
            inner relation coefficient.
        ssq_diff: np.ndarray of shape (num_components, 2)
            fraction of variance used in the x and y matrices.

    Methods:
        fit(x, y): fit model to data. both x and y should be np.ndarray.
        predict(x, y): predict using fitted data.
        get_vip(): return VIP score.

    Examples:
        pls = PLS()
        pls.fit(x1, y1)
        y2_pred = pls.predict(x2)
        vip = pls.get_vip()

    Raises:
        TypeError: if x or y not np.ndarray.
    """

    def __init__(self, num_components=2,
                       center=True,
                       standardize=False,
                       max_iter=500,
                       tol=1e-10):
        self.num_components = num_components
        self.center = center
        self.standardize = standardize
        self.tol = tol
        self.max_iter = max_iter

        if self.center and self.standardize:
            print("center and standardize are both set True. overriding to standardize.")
            self.center = False


    def fit(self, x, y):
        if not isinstance(x, np.ndarray):
            raise TypeError(f"x must be np.ndarray not {type(x)}.")
        if not isinstance(y, np.ndarray):
            raise TypeError(f"y must be np.ndarray not {type(x)}.")

        mx, nx = x.shape
        my, ny = y.shape

        if self.center:
            x = self._center(x)
            y = self._center(y)
        if self.standardize:
            x = self._standardize(x)
            y = self._standardize(y)

        x_loading = np.zeros([nx, self.num_components])
        y_loading = np.zeros([ny, self.num_components])
        x_weight = np.zeros([nx, self.num_components])
        x_score = np.zeros([mx, self.num_components])
        y_score = np.zeros([my, self.num_components])
        rel_coeff = np.zeros([1, self.num_components])

        ssq = np.zeros([self.num_components, 2])

        ssq_x = np.nansum(x ** 2)
        ssq_y = np.nansum(y ** 2)

        for i in range(self.num_components):
            x_loading_t, y_loading_t, x_weight_t, x_score_t, y_score_t = self._nipals(x, y)

            rel_coeff[0, i] = y_score_t.T @ x_score_t / (x_score_t.T @ x_score_t)
            x = x - x_score_t @ x_loading_t.T
            y = y - rel_coeff[0, i] * x_score_t @ y_loading_t.T

            ssq[i, 0] = np.nansum(x ** 2) * 100 / ssq_x
            ssq[i, 1] = np.nansum(y ** 2) * 100 / ssq_y

            x_loading[:, i] = x_loading_t[:, 0]
            y_loading[:, i] = y_loading_t[:, 0]
            x_weight[:, i] = x_weight_t[:, 0]
            x_score[:, i] = x_score_t[:, 0]
            y_score[:, i] = y_score_t[:, 0]

        ssq_diff = np.zeros([self.num_components, 2])
        ssq_diff[0, 0] = 100 - ssq[0, 0]
        ssq_diff[0, 1] = 100 - ssq[0, 1]
        for i in range(self.num_components-1):
            ssq_diff[i+1, :] = -ssq[i+1, :] + ssq[i, :]

        self.x_loading = x_loading
        self.y_loading = y_loading
        self.x_weight = x_weight
        self.x_score = x_score
        self.y_score = y_score
        self.rel_coeff = rel_coeff
        self.ssq_diff = ssq_diff

        return

    def predict(self, x):
        mx, _ = x.shape
        m_y_loading, _ = self.y_loading.shape

        t_hat = np.zeros([mx, self.num_components])
        y_pred = np.zeros([mx, m_y_loading])

        for i in range(self.num_components):
            t_hat[:, i] = (x @ self.x_weight[:, i][:, np.newaxis]).ravel()
            x = x - t_hat[:, i][:, np.newaxis] @ self.x_loading[:, i][:, np.newaxis].T

        for i in range(self.num_components):
            y_pred = y_pred + self.rel_coeff[0, i] * t_hat[:, i][:, np.newaxis] @ self.y_loading[:, i][:, np.newaxis].T

        return y_pred

    def get_vip(self):
        x_weight_norm = self.x_weight / np.sqrt(np.nansum(self.x_weight ** 2))
        ssq = np.nansum(self.x_score ** 2, axis=0) * np.nansum(self.y_loading ** 2, axis=0)
        vip = np.sqrt(self.x_loading.shape[0] * np.nansum(ssq * x_weight_norm ** 2, axis=1)) / sum(ssq)
        return vip

    def _nipals(self, x, y):
        mx, nx = x.shape
        _, ny = y.shape

        ssq_y = np.nansum(y ** 2, axis=0)
        y_score = y[:, np.argmax(ssq_y)][:, np.newaxis]

        tol = 1.
        x_score_old = x[:, 0][:, np.newaxis]
        count = 1

        x_weight = np.zeros([nx, 1])
        x_score = np.zeros([mx, 1])
        y_loading = np.zeros([1, 1])

        while tol > self.tol:
            count += 1

            for i in range(nx):
                num_not_nan = mx - np.nansum(np.isnan(x[:, i]))
                x_weight[i] = np.nansum(y_score.T * x[:, i]) * mx / num_not_nan
            x_weight = (x_weight.T / np.linalg.norm(x_weight.T)).T

            for i in range(mx):
                num_not_nan = nx - np.sum(np.isnan(x[i, :]))
                x_score[i] = np.nansum(x[i, :] * x_weight.T) * nx / num_not_nan

            if ny == 1:
                y_loading[0] = 1
                break

            y_loading = (x_score.T @ y).T
            y_loading = (y_loading.T / np.linalg.norm(y_loading)).T

            y_score = y @ y_loading
            tol = np.linalg.norm(x_score_old - x_score)
            x_score_old = x_score.copy()

            if count >= self.max_iter:
                raise ValueError(f"Algorithm failed to converge after {self.max_iter} iterations.")

        x_loading = (x_score.T @ x / (x_score.T @ x_score)).T
        x_loading = np.nan_to_num(x_loading, nan=np.nanmean(x_loading))
        x_loading_norm = np.linalg.norm(x_loading)

        x_score = x_score * x_loading_norm
        x_weight = x_weight * x_loading_norm
        x_loading = x_loading / x_loading_norm

        return x_loading, y_loading, x_weight, x_score, y_score

    @staticmethod
    def _standardize(x):
        x = (x - np.nanmean(x, axis=0)) / np.nanstd(x, ddof=1, axis=0)
        return x

    @staticmethod
    def _center(x):
        x = x - np.nanmean(x, axis=0)
        return x


class PCA:
    """
    Principal Component Analysis class (accounts NaN)

    Parameters:
        num_components: int, default=None
            number of components. if not set, all components are kept.
        standardize : bool, default=True
            whether to standardize x.

    Attributes:
        score: np.ndarray of shape (num_features, num_components)
        loading: np.ndarray of shape (num_input, num_components)
        ssq_diff: np.ndarray of shape (num_input, 1)

    Methods:
        fit(x): fit model to data. both x and y should be np.ndarray.

    Examples:
        pca = PCA()
        pca.fit(x)

    Raises:
        TypeError: if x not np.ndarray.
    """

    def __init__(self, num_components=None,
                       standardize=True):
        self.num_components = num_components
        self.standardize = standardize

    def fit(self, x):
        if not isinstance(x, np.ndarray):
            raise TypeError(f"x must be np.ndarray not {type(x)}.")

        mx, nx = x.shape

        if not self.num_components:
            self.num_components = nx
        if self.standardize:
            x = self._standardize(x)

        cov = np.zeros([nx, nx])
        for i in range(self.num_components):
            for j in range(self.num_components):
                x_t = x[:, i]
                y_t = x[:, j]

                x_t_idx = np.where(~np.isnan(x_t))
                y_t_idx = np.where(~np.isnan(y_t))

                t_idx = np.intersect1d(x_t_idx, y_t_idx)
                x_t = x_t[t_idx]
                y_t = y_t[t_idx]
                if len(t_idx) != 1:
                    cov[i, j] = (x_t.T @ y_t) / (len(t_idx)-1)
                else:
                    cov[i, j] = 0

        _, st, loading = np.linalg.svd(cov)
        loading = loading.T
        s = np.zeros([nx, nx])
        np.fill_diagonal(s, st)

        var = np.diag(s) * 100 / sum(np.diag(s))
        ssq = np.cumsum(var)

        ssq_diff = np.zeros([self.num_components, 1])
        ssq_diff[0] = ssq[0]
        for i in range(self.num_components-1):
            ssq_diff[i+1] = ssq[i+1] - ssq[i]

        score = np.zeros([mx, self.num_components])
        for i in range(mx):
            x_t = x[i, :]
            x_t_idx = np.where(~np.isnan(x_t))
            score[i, :] = x_t[x_t_idx] @ loading[x_t_idx, :]

        self.score = score
        self.loading = loading
        self.ssq_diff = ssq_diff

    @staticmethod
    def _standardize(x):
        x = (x - np.nanmean(x, axis=0)) / np.nanstd(x, ddof=1, axis=0)
        return x


if __name__ == "__main__":
    import matlab
    import matlab.engine

    """
    test code for NMF
    """
    eng = matlab.engine.start_matlab()

    x = np.random.randn(128, 32)
    x.ravel()[np.random.choice(x.size, 128, replace=False)] = np.nan
    w = np.random.uniform(size=[128, 4])
    h = np.random.uniform(size=[4, 64])

    nmf = NMF(num_iter=3, nmf_iter=10)
    nmf.fit(x, init_w=w, init_h=h)

    coph_cor_py = nmf.correlation_coefficient
    ave_C_py = nmf.consensus

    x_matlab = matlab.double(x.tolist())
    w_matlab = matlab.double(w.tolist())
    h_matlab = matlab.double(h.tolist())
    out_m = eng.aoNMF_subtyping_NaN3(x_matlab, 3, 10, 3, w_matlab, h_matlab, nargout=4) # type: ignore

    coph_cor_m = np.array(out_m[0])
    ave_C_m = np.array(out_m[1])

    np.testing.assert_array_almost_equal(coph_cor_py, coph_cor_m)
    np.testing.assert_array_almost_equal(ave_C_py, ave_C_m)

    """
    test code for PLS
    """
    eng = matlab.engine.start_matlab()

    x = np.random.randn(64, 64)
    y = np.random.randn(64, 4)

    x.ravel()[np.random.choice(x.size, 128, replace=False)] = np.nan
    x.ravel()[np.random.choice(x.size, 64, replace=False)] = 2.
    y.ravel()[np.random.choice(y.size, 4, replace=False)] = 2.

    pls = PLS(center=False)
    pls.fit(x, y)

    p_py = pls.x_loading
    q_py = pls.y_loading
    w_py = pls.x_weight
    t_py = pls.x_score
    u_py = pls.y_score
    b_py = pls.rel_coeff
    ssqdif_py = pls.ssq_diff

    x_matlab = matlab.double(x.tolist())
    y_matlab = matlab.double(y.tolist())

    out_m = eng.plsmd(x_matlab, y_matlab, 2, 1, nargout=7) # type: ignore
    p_m = np.array(out_m[0])
    q_m = np.array(out_m[1])
    w_m = np.array(out_m[2])
    t_m = np.array(out_m[3])
    u_m = np.array(out_m[4])
    b_m = np.array(out_m[5])
    ssqdif_m = np.array(out_m[6])

    np.testing.assert_array_almost_equal(p_py, p_m)
    np.testing.assert_array_almost_equal(q_py, q_m)
    np.testing.assert_array_almost_equal(w_py, w_m)
    np.testing.assert_array_almost_equal(t_py, t_m)
    np.testing.assert_array_almost_equal(u_py, u_m)
    np.testing.assert_array_almost_equal(b_py, b_m)
    np.testing.assert_array_almost_equal(ssqdif_py, ssqdif_m)


    """
    test code for PCA
    """
    eng = matlab.engine.start_matlab()

    x = np.random.randn(64, 32)
    x.ravel()[np.random.choice(x.size, 128, replace=False)] = np.nan
    x.ravel()[np.random.choice(x.size, 32, replace=False)] = 16

    pca = PCA()
    pca.fit(x)

    score_py = pca.score[:, :4]
    loading_py = pca.loading[:, :4]

    x_matlab = matlab.double(x.tolist())

    out_m = eng.pca_missing_value(x_matlab, 0, 0, 4, nargout=3) # type: ignore
    score_m = np.array(out_m[0])
    loading_m = np.array(out_m[1])

    np.testing.assert_array_almost_equal(score_py, score_m)
    np.testing.assert_array_almost_equal(loading_py, loading_m)

