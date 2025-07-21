def unpack_meerkat(fn, nsamp, nchan=1024, npol=2, start_heap=0):
    """
    Unpacking of MeerKAT .dada files in 'heaps' of [2pol, 1024chan, 256samp]
    Each complex sample is stored as a 16bit int (8bit+8bit)
    
    Returns complex64 arrays of shape: (nchan, npol, nsamp)

    Parameters:
    fn : str
        The file name of the .dada file to read.
    nsamp : int
        The total number of samples to read, must be a multiple of 256.
    nchan : int, optional
        The number of channels, typically (or perhaps always?) 1024.
    npol : int, optional
        The number of polarizations, typically (or perhaps always?) 2.
    start_heap : int, optional
        Seek start_heap in the file, before reading data.

    
    """
    heap_len = 256
    header_size = 4096

    if nsamp % heap_len != 0:
        raise ValueError("nsamp must be a multiple of 256")

    nheaps = nsamp // heap_len
    samples_per_heap = nchan * npol * heap_len
    total_samples = nheaps * samples_per_heap

    heap_bytes = samples_per_heap * 2  # 2 bytes per int16
    seek_bytes = header_size + start_heap * heap_bytes
    
    # Read raw int16 data (each int16 holds 2 int8s: real, imag)
    with open(fn, 'rb') as f:
        f.seek(seek_bytes, 0)
        raw = np.frombuffer(f.read(total_samples * 2), dtype=np.int16)

    # View each int16 as two int8s: [real, imag]
    from8 = raw.view(np.int8).reshape(-1, 2)

    # Transpose to (nchan, npol, heap_len) and concatenate time across heaps
    out = np.empty((npol, nchan, nsamp), dtype=np.complex64)
    for i in range(nheaps):
        irange = slice(i*samples_per_heap, (i+1)*samples_per_heap)
        re_i = from8[irange, 0]
        im_i = from8[irange, 1]
        re_i = re_i.reshape((npol, nchan, heap_len))
        im_i = im_i.reshape((npol, nchan, heap_len))

        trange = slice(i*heap_len, (i+1)*heap_len)
        out[..., trange] = re_i + 1j*im_i

    # Swap axes to get the final shape (nchan, npol, nsamp)
    return np.swapaxes(out, 0, 1)