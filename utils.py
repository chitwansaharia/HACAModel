def chunks(l, n):
    for i in range(0, l.shape[1], n):
        yield l[:,i:i+n,:]