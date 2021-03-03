"""Coordinate support

We follow the casa definition of coordinate systems http://casa.nrao.edu/Memos/CoordConvention.pdf :

UVW is a right-handed coordinate system, with W pointing towards the
source, and a baseline convention of :math:`ant2 - ant1` where
:math:`index(ant1) < index(ant2)`.  Consider an XYZ Celestial
coordinate system centered at the location of the interferometer, with
:math:`X` towards the East, :math:`Z` towards the NCP and :math:`Y` to
complete a right-handed system. The UVW coordinate system is then
defined by the hour-angle and declination of the phase-reference
direction such that

1. when the direction of observation is the NCP (`ha=0,dec=90`),
   the UVW coordinates are aligned with XYZ,

2. V, W and the NCP are always on a Great circle,

3. when W is on the local meridian, U points East

4. when the direction of observation is at zero declination, an
   hour-angle of -6 hours makes W point due East.

The :math:`(l,m,n)` coordinates are parallel to :math:`(u,v,w)` such
that :math:`l` increases with Right-Ascension (or increasing longitude
coordinate), :math:`m` increases with Declination, and :math:`n` is
towards the source. With this convention, images will have Right
Ascension increasing from Right to Left, and Declination increasing
from Bottom to Top.

"""

__all__ = ["uvw_ha_dec"]

import numpy


def uvw_ha_dec(antxyz, ha, dec):
    """Calculate uvw for hour angle and declination

    :param antxyz:
    :param ha:
    :param dec:
    :return:
    """
    #     implicit real*8 (a-h,o-z), integer*4 (i-n)
    #     real*8 lx(*),ly(*),lz(*),uant(*),vant(*),want(*)
    #     sh=sin(h)
    #     ch=cos(h)
    #     sd=sin(delta)
    #     cd=cos(delta)
    #     do 10 i=1,n
    #        uant(i)=sh*lx(i)+ch*ly(i)
    #        vant(i)=-sd*(ch*lx(i)-sh*ly(i))+cd*lz(i)
    #        want(i)=cd*(ch*lx(i)-sh*ly(i))+sd*lz(i)
    # 10  continue

    cosdec = numpy.cos(dec)
    sindec = numpy.sin(dec)
    cosha = numpy.cos(ha)
    sinha = numpy.sin(ha)

    x, y, z = numpy.hsplit(antxyz, 3)

    u = sinha * x + cosha * y
    u0 = cosha * x - sinha * y
    v = -sindec * u0 + cosdec * z
    w = cosdec * u0 + sindec * z

    return numpy.hstack([u, v, w])
