import numpy as np

class ComplexSignal:
  """
  Class: ComplexSignal
  Purpose: Process data with MTF
  """
  def __init__(self, time, theta, r, time_bins = 4, theta_bins = 4, r_bins = 4, image_size = 8, n_order = 4, _class=None):
    n_samples, n_features = time.shape
    self.image_size = image_size
    self.window_size = n_features // image_size
    self.time_series = [time, theta, r]
    self.num_bins = [time_bins, theta_bins, r_bins]
    self.sample = n_samples
    self.features = n_features
    self.mtf = None
    self.remainder = n_features % image_size
    self.n_order = n_order
    self._class = _class

  def _mtf(self, binned_ts, num_bins):
    MTM = np.zeros((num_bins, num_bins))
    
    for i in range(self.n_order):
      temp_MTM = np.zeros((num_bins, num_bins))

      lagged_ts = np.vstack([binned_ts[:-1 - i], binned_ts[1 + i:]])

      np.add.at(temp_MTM, tuple(map(tuple, lagged_ts)), 1)
      temp_MTM *= (.4 ** (i+1))
      MTM += temp_MTM

    non_zero_rows = np.where(MTM.sum(axis=1) != 0)[0]
    MTM = np.multiply(MTM[non_zero_rows][:, non_zero_rows].T,
                      np.sum(MTM[non_zero_rows], axis=1)**(-1)).T

    MTF = np.zeros((self.features, self.features))
    list_values = [np.where(binned_ts == q) for q in non_zero_rows]

    for i in range(non_zero_rows.size):
        for j in range(non_zero_rows.size):
            MTF[tuple(np.meshgrid(list_values[i], list_values[j]))] = MTM[i, j]
    remainder = self.remainder
    if remainder == 0:
      return np.reshape(MTF,
                        (self.image_size, self.window_size,
                          self.image_size, self.window_size)
                        ).mean(axis=(1, 3))
    else:
      self.window_size += 1
      start, end, _ = segmentation(MTF.shape[0], self.window_size,
                                    False, self.image_size)
      AMTF = np.zeros((self.image_size, self.image_size))
      for i in range(self.image_size):
          for j in range(self.image_size):
              AMTF[i, j] = MTF[start[i]:end[i], start[j]:end[j]].mean()

      return AMTF

  def generateMTF(self):
    ts, num_bins = self.time_series, self.num_bins
    mtf = []
    _bit_size = ( 2**16 ) - 1
    for idx in range(len(ts)):
      temp_mtf = None
      bins = self.createQuantiles(ts[idx], num_bins[idx])
      binned_ts = self.valuesToQuantiles(ts[idx], bins)
      temp_mtf = np.apply_along_axis(self._mtf, 1, binned_ts, num_bins[idx])[0]
      interp = np.interp(temp_mtf, (temp_mtf.min(), temp_mtf.max()), (0, _bit_size)).astype(np.uint8)
      mtf.append(interp)

    self.mtf = np.dstack((mtf[1], mtf[2], mtf[0]))
  def createQuantiles(self, data, num_bins):
    return np.percentile(data,
                    np.linspace(0, 100, num_bins + 1)[1:-1],
                    axis=1)

  def valuesToQuantiles(self, data, data_bins):
    n_samples, n_features = data.shape
    mask = np.r_[
        ~np.isclose(0, np.diff(data_bins, axis=0), rtol=0, atol=1e-8),
        np.full((1, n_samples), True)
    ]
    if isinstance(data_bins[0][0], complex):
      binned_ts = []
      data_bins = data_bins.flatten()
      for time_point in data[0]:
        binned = False
        for idx in range(len(data_bins)):
          # print(time_point, data_bins[idx])
          if abs(time_point) <= abs(data_bins[idx]):
            binned_ts.append(idx)
            binned = True
        if not binned:
          binned_ts.append(len(data_bins))
      return np.asarray(binned_ts).reshape(1, -1)
    else:
    	return np.array([np.digitize(data[i], data_bins[:, i][mask[:, i]])
                                for i in range(n_samples)])
def segmentation(ts_size, window_size, overlapping, n_segments=None):
    """Compute the indices for Piecewise Agrgegate Approximation.

    Parameters
    ----------
    ts_size : int
        The size of the time series.

    window_size : int
        The size of the window.

    overlapping : bool
        If True, overlapping windows may be used. If False, non-overlapping
        are used.

    n_segments : int or None (default = None)
        The number of windows. If None, the number is automatically
        computed using `window_size`.

    Returns
    -------
    start : array
        The lower bound for each window.

    end : array
        The upper bound for each window.

    size : int
        The size of `start`.

    """
    if n_segments is None:
        quotient = ts_size // window_size
        remainder = ts_size % window_size
        n_segments = quotient if remainder == 0 else quotient + 1

    bounds = np.linspace(0, ts_size,
                         n_segments + 1, endpoint=True).astype('int64')

    start = bounds[:-1]
    end = bounds[1:]
    size = start.size

    if not overlapping:
        return start, end, size
    else:
        correction = window_size - end + start
        half_size = size // 2
        new_start = start.copy()
        new_start[half_size:] = start[half_size:] - correction[half_size:]
        new_end = end.copy()
        new_end[:half_size] = end[:half_size] + correction[:half_size]
        return new_start, new_end, size