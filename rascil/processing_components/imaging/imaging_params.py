import numpy

from rascil.data_models.memory_data_models import Image, BlockVisibility
from rascil.data_models.polarisation import PolarisationFrame
from rascil.processing_components.image.operations import image_is_canonical

__all__ = ['get_rowmap', 'get_polarisation_map', 'get_frequency_map']

def get_frequency_map(vis, im: Image = None):
    """ Map channels from visibilities to image

    """

    # Find the unique frequencies in the visibility
    ufrequency = numpy.unique(vis.frequency)
    vnchan = len(ufrequency)

    if im is None:
        spectral_mode = 'channel'
        if vis.frequency_map is None:
            vfrequencymap = get_rowmap(vis.frequency, ufrequency)
            vis.frequencymap = vfrequencymap
        else:
            vfrequencymap = vis.frequency_map

        assert min(vfrequencymap) >= 0, "Invalid frequency map: visibility channel < 0: %s" % str(vfrequencymap)

    elif im["pixels"].data.shape[0] == 1 and vnchan >= 1:
        assert image_is_canonical(im)

        spectral_mode = 'mfs'
        vfrequencymap = numpy.zeros_like(vis.frequency, dtype='int')

    else:
        assert image_is_canonical(im)

        # We can map these to image channels
        v2im_map = im.wcs.sub(['spectral']).wcs_world2pix(ufrequency, 0)[0].astype('int')

        spectral_mode = 'channel'
        nrows = len(vis.frequency)
        row2vis = numpy.array(get_rowmap(vis.frequency, ufrequency))
        vfrequencymap = [v2im_map[row2vis[row]] for row in range(nrows)]

        assert min(vfrequencymap) >= 0, "Invalid frequency map: image channel < 0 %s" % str(vfrequencymap)
        assert max(vfrequencymap) < im["pixels"].data.shape[0], "Invalid frequency map: image channel > number image channels %s" % \
                                                 str(vfrequencymap)

    return spectral_mode, vfrequencymap


def get_polarisation_map(vis: BlockVisibility, im: Image = None):
    """ Get the mapping of visibility polarisations to image polarisations

    """
    assert image_is_canonical(im)

    if vis.blockvisibility_acc.polarisation_frame == im.image_acc.polarisation_frame:
        if vis.blockvisibility_acc.polarisation_frame == PolarisationFrame('stokesI'):
            return "stokesI->stokesI", lambda pol: 0
        elif vis.blockvisibility_acc.polarisation_frame == PolarisationFrame('stokesIQUV'):
            return "stokesIQUV->stokesIQUV", lambda pol: pol

    return "unknown", lambda pol: pol


def get_rowmap(col, ucol=None):
    """ Map to unique cols

    :param col: Data column
    :param ucol: Unique values in col
    """
    pdict = {}

    def phash(f):
        return numpy.round(f).astype('int')

    if ucol is None:
        ucol = numpy.unique(col)

    for i, f in enumerate(ucol):
        pdict[phash(f)] = i
    # vmap = []
    # vmap = [pdict[phash(p)] for p in col]
    # for p in col:
    #     vmap.append(pdict[phash(p)])

    n_ucol = numpy.round(col).astype(('int'))
    vmap = numpy.vectorize(pdict.__getitem__)(n_ucol)

    return vmap.tolist()
