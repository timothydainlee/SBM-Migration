import numpy as np
from scipy.stats import rankdata
import warnings


def quantilenorm(x, average="mean"):
    """
    Performs 2d quantile normalization. (over columns)

    Arguments:
        x: np.ndarray
            input array.
        average: str, default="mean"
            average method. "mean" or "median".

    Returns:
        x_norm: np.ndarray
            normalized array.

    Raises:
        TypeError: if x not np.ndarray.
        ValueError: if average not "mean" or "median".
    """

    if not isinstance(x, np.ndarray):
        raise TypeError(f"x must be np.ndarray not {type(x)}.")

    if average == "mean":
        average_func = np.mean
    elif average == "median":
        average_func = np.median
    else:
        raise ValueError(f"average must be either 'mean' of 'median' not {average}.")

    x = x.copy()
    x_norm = x.copy()

    r, c = x.shape
    x_nan = np.isnan(x)
    num_nans = np.sum(x_nan, axis=0)

    x[np.isnan(x)] = np.inf

    rr = []
    x_sorted = np.sort(x, axis=0)
    idx_sorted = np.argsort(x, axis=0)
    x_ranked = np.zeros([r, c])
    for i in range(c):
        ranked = rankdata(x[:, i][~x_nan[:, i]])
        rr.append(np.sort(ranked))

        m = r - num_nans[i]
        try:
            x_ranked[:, i] = np.interp(np.arange(1, r + 1),
                                       np.arange(1, r + 1, (r - 1) / (m - 1)),
                                       x_sorted[0:m, i])
        except ValueError:
            warnings.warn(f"{i}th column cannot be interpolated.")
            continue

    mean_val = average_func(x_ranked, axis=1)

    for i in range(c):
        m = r - num_nans[i]
        replace_idx = idx_sorted[:, i][0:m]
        x_norm[:, i][replace_idx] = np.interp(1 + ((r - 1) * (rr[i] - 1) / (m - 1)),
                                              np.arange(1, r + 1),
                                              mean_val)

    return x_norm


def segmented_quantile_normalization(x, segment_size=4, stride=2, error_criterion=1e-4, max_iter=100):
    """
    Performs segmented 2d quantile normalization.
    Arguments:
        x: list of list
            input time series.
        segment_size: int, default=4
            segment size.
        stride: int, default=2
            stride size.
        error_criterion: float, default=1e-4
            stop iteration criterion.
        max_iter: int, default=100
            maximum iteration steps.
    Returns:
        x: list of list
            normalized time series.
        num_iteration: int
            number of iterations. max_iter if not converged.
        error: list
            Root Mean Squared Errors
    """


    x_original = x.copy()

    # remove NaN paddings
    x_clean = []
    for p in x:
        valid = ~np.isnan(p)
        if np.sum(valid) > 0:
            p = p[np.argmax(valid):len(valid)-np.argmax(valid[::-1])]
            x_clean.append(p)

    # padding according to segment_size, stride
    for i, p in enumerate(x_clean):
        if len(p) < segment_size:
            num = segment_size - len(p)
        elif (segment_size-len(p)) % stride != 0:
            num = (segment_size-len(p)) % stride
        else:
            num = 0
        x_clean[i] = p + [np.nan] * num

    # reshape by time
    nums = (((np.array(list(map(len, x_clean)))) - segment_size) / stride + 1).astype(int)
    max_nums = max(nums)

    ts = np.empty([len(x_clean) * segment_size, max_nums])
    ts[:] = np.nan
    for i, p in enumerate(x_clean):
        for j in range(nums[i]):
            ts[segment_size*i:segment_size*(i+1), j] = p[stride*j:stride*j+segment_size]

    # prepare reshape_idx for: reshape by valid
    valid = np.sum(~np.isnan(ts), axis=0)
    size_criterion = valid[0] * .5

    reshape_idx = []
    aug_v, idx = 0, 0
    for i in range(max_nums):
        v = valid[i]
        if v > size_criterion:
            reshape_idx.append(idx)
            idx += 1
        else:
            aug_v += v
            reshape_idx.append(idx)

        if aug_v > size_criterion:
            aug_v = 0
            idx += 1

    # check last idx
    max_idx = max(reshape_idx)
    if max_idx == 0:
        warnings.warn(f"Not enough data for each series to quantile normalize.")
        return x_original, 0, [0]

    v_idx = [i for i, j in enumerate(reshape_idx) if j == max_idx]
    v_count = np.sum(~np.isnan(ts[:, v_idx]))
    if (max_idx != 1) and (v_count <= size_criterion):
        reshape_idx = [max_idx-1 if i == max_idx else i for i in reshape_idx]

    # loop segmented quantilenorm
    num_iteration = 0
    error, errors = 1, []
    m = np.concatenate([np.zeros(stride), np.ones(segment_size-stride)])
    m = np.tile(m, len(x_clean)).astype(bool)
    while error > error_criterion and num_iteration < max_iter:

        # reshape by size
        max_size = max([reshape_idx.count(i) for i in range(max(reshape_idx)+1)])
        ss = np.empty([len(x_clean) * segment_size * max_size, max(reshape_idx)+1])
        ss[:] = np.nan
        for i in range(max(reshape_idx)+1):
            idx = np.where(np.array(reshape_idx) == i)[0]
            ss[:len(ts[:, idx].reshape(-1, order="F")), i] = ts[:, idx].reshape(-1, order="F")

        ss = quantilenorm(ss) # quantilenorm

        # reshape to original
        count = 0
        for i in range(max(reshape_idx)+1):
            idx = np.where(np.array(reshape_idx) == i)[0]
            for j in range(len(idx)):
                ts[:, count] = ss[(len(x_clean)*segment_size)*j:(len(x_clean)*segment_size)*(j+1), i].ravel()
                count += 1

        es = []
        ts_before = ts.copy()
        for i in range(ts.shape[1]-1):
            t1 = ts_before[:, i]
            t2 = ts_before[:, i+1]
            m1 = np.isnan(t1)
            m2 = np.isnan(t2)
            t2 = np.roll(t2, stride)

            # allocate mean
            mean = np.nanmean([t1, t2], axis=0)

            ts[stride:, i][m[stride:]] = mean[stride:][m[stride:]]
            ts[:-stride, i+1][m[stride:]] = mean[stride:][m[stride:]]
            ts[:, i][m1] = np.nan
            ts[:, i+1][m2] = np.nan

            # calculate error
            me1 = ~m1 & m
            me2 = ~np.roll(m2, stride) & m
            me = me1 & me2
            e = np.sqrt((t2 - t1) ** 2)
            es.append(e[me])

        error = np.concatenate(es).mean()
        errors.append(error)
        num_iteration += 1

    # reconstruct
    x_norm = []
    m = np.concatenate([np.ones(stride), np.zeros(segment_size-stride)])
    m = np.tile(m, max_nums)
    m = np.concatenate([np.ones(segment_size-stride), m]).astype(bool)

    idx = 0
    for p in x_original:
        xn = np.array(p.copy())
        valid = ~np.isnan(xn)
        if np.sum(valid) == 0:
            x_norm.append(xn)
        else:
            t = ts[segment_size*idx:segment_size*(idx+1)].reshape(-1, order="F")
            t = t[m[:len(t)]]
            xn[~np.isnan(xn)] = t[~np.isnan(t)]
            x_norm.append(xn)
            idx += 1

    return x_norm, num_iteration, errors)


if __name__ == "__main__":
    import matlab
    import matlab.engine

    """
    test code for quantilenorm
    """
    eng = matlab.engine.start_matlab()

    x = np.random.randn(64, 32)
    x.ravel()[np.random.choice(x.size, 2040, replace=False)] = np.nan
    x.ravel()[np.random.choice(x.size, 512, replace=False)] = 16

    quantilenorm_py = quantilenorm(x)

    x_m = matlab.double(x.tolist())
    quantilenorm_m = eng.quantilenorm(x_m) # type: ignore
    quantilenorm_m = np.array(quantilenorm_m)

    np.testing.assert_array_almost_equal(quantilenorm_m, quantilenorm_py)

