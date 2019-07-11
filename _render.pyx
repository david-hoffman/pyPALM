cdef double [:, :] _gauss(Py_ssize_t yw, Py_ssize_t xw, float y0, float x0, float sy, float sx):
    """Simple normalized 2D gaussian function for rendering"""
    # for this model, x and y are seperable, so we can generate
    # two gaussians and take the outer product
    cdef double amp = 1 / (2 * sy * sx)
    
    result = np.empty((yw, xw), dtype=np.float64)
    cdef double [:, :] result_view = result
    
    cdef Py_ssize_t x, y
    
    for y in range(yw):
        for x in range(xw):
            result_view[y, x] = np.exp(-((y - y0) / sy) ** 2 / 2 - ((x - x0) / sx) ** 2 / 2) * amp
    
    return result