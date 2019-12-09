# From John K in a post to pylearn-users: https://groups.google.com/forum/#!topic/pylearn-users/oAFyQLdZUYA

import os, gzip, bz2
import numpy

def load(cls, which_set, filetype):
    """
    Reads and returns a single file as a numpy array.
    """
    assert which_set in ['train', 'test']
    assert filetype in ['dat', 'cat', 'info']

def get_path(which_set, filetype):
    dirname = 'norb-small'
    if which_set == 'train':
        instance_list = '46789'
    elif which_set == 'test':
        instance_list = '01235'
    filename = 'smallnorb-5x%sx9x18x6x2x96x96-%s-%s.mat' % \
        (instance_list, which_set + 'ing', filetype)
    return os.path.join(dirname, filename)


def read_nums(file_handle, num_type, count):
    """
    Reads 4 bytes from file, returns it as a 32-bit integer.
    """
    num_bytes = count * numpy.dtype(num_type).itemsize
    string = file_handle.read(num_bytes)
    return numpy.fromstring(string, dtype = num_type)

def read_header(file_handle, debug=False, from_gzip=None):
    """
    :param file_handle: an open file handle.
    :type file_handle: a file or gzip.GzipFile object

    :param from_gzip: bool or None
    :type from_gzip: if None determine the type of file handle.

    :returns: data type, element size, rank, shape, size
    """

    if from_gzip is None:
        from_gzip = isinstance(file_handle,
                              (gzip.GzipFile, bz2.BZ2File))

    key_to_type = { 0x1E3D4C51 : ('float32', 4),
                    # what is a packed matrix?
                    # 0x1E3D4C52 : ('packed matrix', 0),
                    0x1E3D4C53 : ('float64', 8),
                    0x1E3D4C54 : ('int32', 4),
                    0x1E3D4C55 : ('uint8', 1),
                    0x1E3D4C56 : ('int16', 2) }

    type_key = read_nums(file_handle, 'int32', 1)[0]
    elem_type, elem_size = key_to_type[type_key]

    if elem_type == 'packed matrix':
        raise NotImplementedError('packed matrix not supported')

    num_dims = read_nums(file_handle, 'int32', 1)[0]
    if debug:
        print('# of dimensions, according to header: ', num_dims)

    if from_gzip:
        shape = read_nums(file_handle,
                         'int32',
                         max(num_dims, 3))[:num_dims]
    else:
        shape = numpy.fromfile(file_handle,
                               dtype='int32',
                               count=max(num_dims, 3))[:num_dims]

    if debug:
        print('Tensor shape, as listed in header:', shape)

    return elem_type, elem_size, shape


def parse_NORB_file(file_handle, subtensor=None, debug=False, filetype='dat'):
    """
    Load all or part of file 'f' into a numpy ndarray
    :param file_handle: file from which to read file can be opended with
      open(), gzip.open() and bz2.BZ2File() @type f: file-like
      object. Can be a gzip open file.

    :param subtensor: If subtensor is not None, it should be like the
      argument to numpy.ndarray.__getitem__.  The following two
      expressions should return equivalent ndarray objects, but the one
      on the left may be faster and more memory efficient if the
      underlying file f is big.

       read(f, subtensor) <===> read(f)[*subtensor]

      Support for subtensors is currently spotty, so check the code to
      see if your particular type of subtensor is supported.
      """

    elem_type, elem_size, shape = read_header(file_handle,debug)
    beginning = file_handle.tell()

    num_elems = numpy.prod(shape)

    result = None
    if isinstance(file_handle, (gzip.GzipFile, bz2.BZ2File)):
        assert subtensor is None, \
            "Subtensors on gzip files are not implemented."
        result = read_nums(file_handle,
                           elem_type,
                           num_elems*elem_size).reshape(shape)
    elif subtensor is None:
        result = numpy.fromfile(file_handle,
                                dtype = elem_type,
                                count = num_elems).reshape(shape)
    elif isinstance(subtensor, slice):
        if subtensor.step not in (None, 1):
            raise NotImplementedError('slice with step', subtensor.step)
        if subtensor.start not in (None, 0):
            bytes_per_row = numpy.prod(shape[1:]) * elem_size
            file_handle.seek(beginning+subtensor.start * bytes_per_row)
        shape[0] = min(shape[0], subtensor.stop) - subtensor.start
        result = numpy.fromfile(file_handle,
                                dtype=elem_type,
                                count=num_elems).reshape(shape)
    else:
        raise NotImplementedError('subtensor access not written yet:',
                                  subtensor)
    if filetype == 'dat':
        return result.reshape((result.shape[0] * 2, -1))
    else:
        return result

def norb_data(dataset='train', filetype='dat'):
    infile = open(get_path(dataset, filetype), mode='rb')
    return parse_NORB_file(infile, filetype='dat')

def norb_labels(dataset='train'):
    infile = open(get_path(dataset, 'cat'), mode='rb')
    return parse_NORB_file(infile, filetype='cat')

