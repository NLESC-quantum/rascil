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

__all__ = ['xyz_at_latitude', 'xyz_to_baselines', 'xyz_to_uvw', 'uvw_to_xyz',
           'uvw_transform', 'simulate_point', 'skycoord_to_lmn', 'lmn_to_skycoord',
           'visibility_shift', 'azel_to_hadec', 'hadec_to_azel', 'baselines',
           'pa_z', 'parallactic_angle', 'enu_to_xyz', 'lla_to_ecef', 'ecef_to_lla',
           'enu_to_ecef', 'ecef_to_enu', 'eci_to_uvw', 'uvw_to_eci', 'eci_to_enu', 'enu_to_eci']

import numpy
from astropy.coordinates import SkyCoord, CartesianRepresentation
from astropy import units


def lla_to_ecef(lat, lon, alt):
    """Convert WGS84 spherical coordinates to ECEF cartesian coordinates.
    :param lat:
    :param lon:
    :param alt:
    :result ecef:
    """
    WGS84_a = 6378137.00000000
    WGS84_b = 6356752.31424518
    N = WGS84_a ** 2 / numpy.sqrt(WGS84_a ** 2 * numpy.cos(lat) ** 2 + WGS84_b ** 2 * numpy.sin(lat) ** 2)

    x = (N + alt) * numpy.cos(lat) * numpy.cos(lon)
    y = (N + alt) * numpy.cos(lat) * numpy.sin(lon)
    z = ((WGS84_b ** 2 / WGS84_a ** 2) * N + alt) * numpy.sin(lat)

    return x, y, z


def ecef_to_lla(x, y, z):
    """
    Convert earth-centered, earth-fixed coordinates to (rad), longitude
    (rad), elevation (m) using Bowring's method.
    """

    WGS84_a = 6378137.00000000
    WGS84_b = 6356752.31424518
    e2 = (WGS84_a ** 2 - WGS84_b ** 2) / WGS84_a ** 2
    ep2 = (WGS84_a ** 2 - WGS84_b ** 2) / WGS84_b ** 2

    # Distance from rotation axis
    p = numpy.sqrt(x ** 2 + y ** 2)

    # Longitude
    lon = numpy.arctan2(y, x)
    p = numpy.sqrt(x ** 2 + y ** 2)

    # Latitude (first approximation)
    lat = numpy.arctan2(z, p)

    # Latitude (refined using Bowring's method)
    psi = numpy.arctan2(WGS84_a * z, WGS84_b * p)
    num = z + WGS84_b * ep2 * numpy.sin(psi) ** 3
    den = p - WGS84_a * e2 * numpy.cos(psi) ** 3
    lat = numpy.arctan2(num, den)

    # Elevation
    N = WGS84_a ** 2 / numpy.sqrt(WGS84_a ** 2 * numpy.cos(lat) ** 2 + WGS84_b ** 2 * numpy.sin(lat) ** 2)
    alt = p / numpy.cos(lat) - N

    return lat, lon, alt


def enu_to_eci(enu, lat):
    """
    Converts a baseline in [east, north, elevation] to earth-centered inertial coordinates
    for that baseline [x, y, z].

    """
    e, n, u = numpy.hsplit(enu, 3)  # pylint: disable=unbalanced-tuple-unpacking

    x = -numpy.sin(lat) * n + u * numpy.cos(lat)
    y = e
    z = n * numpy.cos(lat) + u * numpy.sin(lat)

    return numpy.hstack([x, y, z])

    # rot_matrix = numpy.array([[0.0, -numpy.sin(lat), numpy.cos(lat)],
    #                     [1.0, 0.0, 0.0],
    #                     [0.0, numpy.cos(lat), numpy.sin(lat)]])
    # return numpy.einsum('ij,jk->ik',enu, numpy.transpose(rot_matrix))


def eci_to_enu(eci, lat):
    """
    Converts a baseline in earth-centered inertial coordinates [x, y, z] to [east, north,
    elevation] for that baseline.

    """

    x, y, z = numpy.hsplit(eci, 3)  # pylint: disable=unbalanced-tuple-unpacking

    e = y
    n = -numpy.sin(lat) * x + z * numpy.cos(lat)
    u = numpy.cos(lat) * x + z * numpy.sin(lat)

    return numpy.hstack([e, n, u])

    # rot_matrix = numpy.array([[0.0, 1.0, 0.0],
    #                     [-numpy.sin(lat), 0.0, numpy.cos(lat)],
    #                     [numpy.cos(lat), 0.0, numpy.sin(lat)]])
    # if len(eci.shape) == 1:
    #     enu = eci.reshape(1, -1)
    # return numpy.einsum('ij,jk->ik', eci, numpy.transpose(rot_matrix))


def enu_to_ecef(location, enu):
    """Convert ENU coordinates relative to reference location to ECEF coordinates.
    :param location: Current WGS84 coordinate
    :param enu: local xyz coordinate
    :result : ECEF
    """
    # ECEF coordinates of reference point

    e, n, u = numpy.hsplit(enu, 3)  # pylint: disable=unbalanced-tuple-unpacking

    lon = location.geodetic[0].to(units.rad).value
    lat = location.geodetic[1].to(units.rad).value
    alt = location.geodetic[2].to(units.m).value

    x, y, z = lla_to_ecef(lat, lon, alt)
    sin_lat, cos_lat = numpy.sin(lat), numpy.cos(lat)
    sin_lon, cos_lon = numpy.sin(lon), numpy.cos(lon)

    X = x - sin_lon * e - sin_lat * cos_lon * n + cos_lat * cos_lon * u
    Y = y + cos_lon * e - sin_lat * sin_lon * n + cos_lat * sin_lon * u
    Z = z + cos_lat * n + sin_lat * u

    return numpy.hstack([X, Y, Z])


def ecef_to_enu(location, xyz):
    """Convert ECEF coordinates to ENU coordinates relative to reference location.
    :param location: Current WGS84 coordinate
    :param xyz: ECEF coordinate
    :result : enu
    """
    # ECEF coordinates of reference point
    lon = location.geodetic[0].to(units.rad).value
    lat = location.geodetic[1].to(units.rad).value
    alt = location.geodetic[2].to(units.m).value
    x, y, z = numpy.hsplit(xyz, 3)  # pylint: disable=unbalanced-tuple-unpacking

    center_x, center_y, center_z = lla_to_ecef(lat, lon, alt)

    delta_x, delta_y, delta_z = x - center_x, y - center_y, z - center_z
    sin_lat, cos_lat = numpy.sin(lat), numpy.cos(lat)
    sin_lon, cos_lon = numpy.sin(lon), numpy.cos(lon)

    e = -sin_lon * delta_x + cos_lon * delta_y
    n = -sin_lat * cos_lon * delta_x - sin_lat * sin_lon * delta_y + cos_lat * delta_z
    u = cos_lat * cos_lon * delta_x + cos_lat * sin_lon * delta_y + sin_lat * delta_z

    return numpy.hstack([e, n, u])


def enu_to_xyz(e, n, u, lat):
    """Convert ENU to XYZ coordinates.
    
       [TMS] Thompson, Moran, Swenson, "Interferometry and Synthesis in Radio
       Astronomy," 2nd ed., Wiley-VCH, 2004, pp. 86-89.

    :param e: East
    :param n: North
    :param u: Up
    :param lat: Latitude
    :result : xyz
    """
    sin_lat, cos_lat = numpy.sin(lat), numpy.cos(lat)
    return -sin_lat * n + cos_lat * u, e, cos_lat * n + sin_lat * u


def xyz_at_latitude(local_xyz, lat):
    """
    Rotate local XYZ coordinates into celestial XYZ coordinates. These
    coordinate systems are very similar, with X pointing towards the
    geographical east in both cases. However, before the rotation Z
    points towards the zenith, whereas afterwards it will point towards
    celestial north (parallel to the earth axis).

    :param lat: target latitude (radians or astropy quantity)
    :param local_xyz: Array of local XYZ coordinates
    :return: Celestial XYZ coordinates
    """

    # return enu_to_eci(local_xyz, lat)
    x, y, z = numpy.hsplit(local_xyz, 3)  # pylint: disable=unbalanced-tuple-unpacking

    lat2 = numpy.pi / 2 - lat
    y2 = -z * numpy.sin(lat2) + y * numpy.cos(lat2)
    z2 = z * numpy.cos(lat2) + y * numpy.sin(lat2)

    return numpy.hstack([x, y2, z2])


def eci_to_uvw(xyz, ha, dec):
    """
    Rotate :math:`(x,y,z)` positions in earth coordinates to
    :math:`(u,v,w)` coordinates relative to astronomical source
    position :math:`(ha, dec)`. Can be used for both antenna positions
    as well as for baselines.

    Hour angle and declination can be given as single values or arrays
    of the same length. Angles can be given as radians or astropy
    quantities with a valid conversion.

    :param xyz: :math:`(x,y,z)` co-ordinates of antennas in array
    :param ha: hour angle of phase tracking centre (:math:`ha = ra - lst`)
    :param dec: declination of phase tracking centre.
    """

    # trans = numpy.array([[ numpy.sin(ha),                  numpy.cos(ha),                 0],
    #                 [-numpy.sin(dec)*numpy.cos(ha),  numpy.sin(dec)*numpy.sin(ha), numpy.cos(dec)],
    #                 [ numpy.cos(dec)*numpy.cos(ha), -numpy.cos(dec)*numpy.sin(ha), numpy.sin(dec)]])

    # return numpy.einsum('ij,jk->ik', xyz, numpy.transpose(trans))

    x, y, z = numpy.hsplit(xyz, 3)  # pylint: disable=unbalanced-tuple-unpacking
    u = numpy.sin(ha) * x + numpy.cos(ha) * y
    v = -numpy.sin(dec) * numpy.cos(ha) * x + numpy.sin(dec) * numpy.sin(ha) * y + numpy.cos(dec) * z
    w = numpy.cos(dec) * numpy.cos(ha) * x - numpy.cos(dec) * numpy.sin(ha) * y + numpy.sin(dec) * z
    return numpy.hstack([u, v, w])
    # Two rotations:
    #  1. by 'ha' along the z axis
    #  2. by '90-dec' along the u axis
    # u = x * numpy.cos(ha) - y * numpy.sin(ha)
    # v0 = x * numpy.sin(ha) + y * numpy.cos(ha)
    # w = z * numpy.sin(dec) - v0 * numpy.cos(dec)
    # v = z * numpy.cos(dec) + v0 * numpy.sin(dec)
    # return numpy.hstack([u, v, w])

    # trans = numpy.array([[ numpy.sin(ha),                  numpy.cos(ha),                 0],
    #                 [-numpy.sin(dec)*numpy.cos(ha),  numpy.sin(dec)*numpy.sin(ha), numpy.cos(dec)],
    #                 [ numpy.cos(dec)*numpy.cos(ha), -numpy.cos(dec)*numpy.sin(ha), numpy.sin(dec)]])


def uvw_to_eci(uvw, ha, dec):
    """
    Rotate :math:`(x,y,z)` positions relative to a sky position at
    :math:`(ha, dec)` to earth coordinates. Can be used for both
    antenna positions as well as for baselines.

    Hour angle and declination can be given as single values or arrays
    of the same length. Angles can be given as radians or astropy
    quantities with a valid conversion.

    :param uvw: :math:`(u,v,w)` co-ordinates of antennas in array
    :param ha: hour angle of phase tracking centre (:math:`ha = ra - lst`)
    :param dec: declination of phase tracking centre
    """
    # trans = numpy.array([[numpy.sin(ha), -numpy.cos(ha) * numpy.sin(dec), numpy.cos(ha) * numpy.cos(dec)],
    #                        [numpy.cos(ha), numpy.sin(dec) * numpy.sin(ha), -numpy.sin(ha) * numpy.cos(dec)],
    #                        [0, numpy.cos(dec), numpy.sin(dec)]])
    #

    u, v, w = numpy.hsplit(uvw, 3)  # pylint: disable=unbalanced-tuple-unpacking
    e = numpy.sin(ha) * u -numpy.cos(ha) * numpy.sin(dec) * v + numpy.cos(ha) * numpy.cos(dec) * w
    c = numpy.cos(ha) * u + numpy.sin(dec) * numpy.sin(ha) * v -numpy.sin(ha) * numpy.cos(dec) * w
    i = numpy.cos(dec) * v + numpy.sin(dec) * w
    return numpy.hstack([u, v, w])

    # trans = numpy.array([[numpy.sin(ha), numpy.cos(ha), 0],
    #                      [-numpy.sin(dec) * numpy.cos(ha), numpy.sin(dec) * numpy.sin(ha), numpy.cos(dec)],
    #                      [numpy.cos(dec) * numpy.cos(ha), -numpy.cos(dec) * numpy.sin(ha), numpy.sin(dec)]])
    #
    # return numpy.einsum('ij,jk->ik', uvw, numpy.transpose(trans))

    # u, v, w = numpy.hsplit(uvw, 3)  # pylint: disable=unbalanced-tuple-unpacking

    # Two rotations:
    #  1. by 'dec-90' along the u axis
    #  2. by '-ha' along the z axis
    # v0 = v * numpy.sin(dec) - w * numpy.cos(dec)
    # z = v * numpy.cos(dec) + w * numpy.sin(dec)
    # x = u * numpy.cos(ha) + v0 * numpy.sin(ha)
    # y = -u * numpy.sin(ha) + v0 * numpy.cos(ha)
    # return numpy.hstack([x, y, z])


def xyz_to_uvw(xyz, ha, dec):
    """
    Rotate :math:`(x,y,z)` positions in earth coordinates to
    :math:`(u,v,w)` coordinates relative to astronomical source
    position :math:`(ha, dec)`. Can be used for both antenna positions
    as well as for baselines.

    Hour angle and declination can be given as single values or arrays
    of the same length. Angles can be given as radians or astropy
    quantities with a valid conversion.

    :param xyz: :math:`(x,y,z)` co-ordinates of antennas in array
    :param ha: hour angle of phase tracking centre (:math:`ha = ra - lst`)
    :param dec: declination of phase tracking centre.
    """

    # return eci_to_uvw(xyz, ha, dec)
    x, y, z = numpy.hsplit(xyz, 3)  # pylint: disable=unbalanced-tuple-unpacking

    # Two rotations:
    #  1. by 'ha' along the z axis
    #  2. by '90-dec' along the u axis
    u = x * numpy.cos(ha) - y * numpy.sin(ha)
    v0 = x * numpy.sin(ha) + y * numpy.cos(ha)
    w = z * numpy.sin(dec) - v0 * numpy.cos(dec)
    v = z * numpy.cos(dec) + v0 * numpy.sin(dec)

    return numpy.hstack([u, v, w])


def uvw_to_xyz(uvw, ha, dec):
    """
    Rotate :math:`(x,y,z)` positions relative to a sky position at
    :math:`(ha, dec)` to earth coordinates. Can be used for both
    antenna positions as well as for baselines.

    Hour angle and declination can be given as single values or arrays
    of the same length. Angles can be given as radians or astropy
    quantities with a valid conversion.

    :param uvw: :math:`(u,v,w)` co-ordinates of antennas in array
    :param ha: hour angle of phase tracking centre (:math:`ha = ra - lst`)
    :param dec: declination of phase tracking centre
    """
    # return uvw_to_eci(uvw, ha, dec)
    u, v, w = numpy.hsplit(uvw, 3)  # pylint: disable=unbalanced-tuple-unpacking

    # Two rotations:
    #  1. by 'dec-90' along the u axis
    #  2. by '-ha' along the z axis
    v0 = v * numpy.sin(dec) - w * numpy.cos(dec)
    z = v * numpy.cos(dec) + w * numpy.sin(dec)
    x = u * numpy.cos(ha) + v0 * numpy.sin(ha)
    y = -u * numpy.sin(ha) + v0 * numpy.cos(ha)

    return numpy.hstack([x, y, z])


def baselines(ants_uvw):
    """
    Compute baselines in uvw co-ordinate system from
    uvw co-ordinate system station positions

    :param ants_uvw: `(u,v,w)` co-ordinates of antennas in array
    """

    res = []
    nants = ants_uvw.shape[0]
    for a1 in range(nants):
        for a2 in range(a1 + 1, nants):
            res.append(ants_uvw[a2] - ants_uvw[a1])

    basel_uvw = numpy.array(res)

    return basel_uvw


def xyz_to_baselines(ants_xyz, ha_range, dec):
    """
    Calculate baselines in :math:`(u,v,w)` co-ordinate system
    for a range of hour angles (i.e. non-snapshot observation)
    to create a uvw sampling distribution

    :param ants_xyz: :math:`(x,y,z)` co-ordinates of antennas in array
    :param ha_range: list of hour angle values for astronomical source as function of time
    :param dec: declination of astronomical source [constant, not :math:`f(t)`]
    """

    dist_uvw = numpy.concatenate([baselines(xyz_to_uvw(ants_xyz, hax, dec)) for hax in ha_range])
    return dist_uvw


def skycoord_to_lmn(pos: SkyCoord, phasecentre: SkyCoord):
    """
    Convert astropy sky coordinates into the l,m,n coordinate system
    relative to a phase centre.

    The l,m,n is a RHS coordinate system with
    * its origin on the sky sphere
    * m,n and the celestial north on the same plane
    * l,m a tangential plane of the sky sphere

    Note that this means that l increases east-wards
    """

    # Determine relative sky position
    todc = pos.transform_to(phasecentre.skyoffset_frame())
    dc = todc.represent_as(CartesianRepresentation)

    # Do coordinate transformation - astropy's relative coordinates do
    # not quite follow imaging conventions
    return dc.y.value, dc.z.value, dc.x.value - 1


def lmn_to_skycoord(lmn, phasecentre: SkyCoord):
    """
    Convert l,m,n coordinate system + phascentre to astropy sky coordinate
    relative to a phase centre.

    The l,m,n is a RHS coordinate system with
    * its origin on the sky sphere
    * m,n and the celestial north on the same plane
    * l,m a tangential plane of the sky sphere

    Note that this means that l increases east-wards
    """

    # Convert l,m,n to SkyCoord convention, also enforce celestial sphere
    n = numpy.sqrt(1 - lmn[0] ** 2 - lmn[1] ** 2) - 1.0
    dc = n + 1, lmn[0], lmn[1]
    target = SkyCoord(x=dc[0], y=dc[1], z=dc[2], representation_type='cartesian', frame=phasecentre.skyoffset_frame())
    return target.transform_to(phasecentre.frame)


def simulate_point(dist_uvw, l, m):
    """
    Simulate visibilities for unit amplitude point source at
    direction cosines (l,m) relative to the phase centre.

    This includes phase tracking to the centre of the field (hence the minus 1
    in the exponent.)

    Note that point source is delta function, therefore the
    FT relationship becomes an exponential, evaluated at
    (uvw.lmn)

    :param dist_uvw: :math:`(u,v,w)` distribution of projected baselines (in wavelengths)
    :param l: horizontal direction cosine relative to phase tracking centre
    :param m: orthogonal directon cosine relative to phase tracking centre
    """

    # vector direction to source
    s = numpy.array([l, m, numpy.sqrt(1 - l ** 2 - m ** 2) - 1.0])
    # complex valued Visibility data_models
    return numpy.exp(-2j * numpy.pi * numpy.einsum("...fs,s->...f", dist_uvw, s))


def visibility_shift(uvw, vis, dl, dm):
    """
    Shift visibilities by the given image-space distance. This is
    based on simple FFT laws. It will require kernels to be suitably
    shifted as well to work correctly.

    :param uvw:
    :param vis: :math:`(u,v,w)` distribution of projected baselines (in wavelengths)
    :param vis: Input visibilities
    :param dl: Horizontal shift distance as directional cosine
    :param dm: Vertical shift distance as directional cosine
    :return: New visibilities

    """

    s = numpy.array([dl, dm])
    return vis * numpy.exp(-2j * numpy.pi * numpy.dot(uvw[:, 0:2], s))


def uvw_transform(uvw, transform_matrix):
    """
    Transforms UVW baseline coordinates such that the image is
    transformed with the given matrix. Will require kernels to be
    suitably transformed to work correctly.

    Reference: Sault, R. J., L. Staveley-Smith, and W. N. Brouw. "An
    approach to interferometric mosaicing." Astronomy and Astrophysics
    Supplement Series 120 (1996): 375-384.

    :param uvw: :math:`(u,v,w)` distribution of projected baselines (in wavelengths)
    :param transform_matrix: 2x2 matrix for image transformation
    :return: New baseline coordinates
    """

    # Apply to uv coordinates
    uv1 = numpy.dot(uvw[:, 0:2], transform_matrix)
    # Restack with original w values
    return numpy.hstack([uv1, uvw[:, 2:3]])


def parallactic_angle(ha, dec, lat):
    """Calculate parallactic angle of source at ha, dec observed from site at latitude dec
    
    H = t - α
    sin(a) = sin(δ) sin(φ) + cos(δ) cos(φ) cos(H)
    sin(A) = - sin(H) cos(δ) / cos(a)
    cos(A) = { sin(δ) - sin(φ) sin(a) } / cos(φ) cos(a)
    
    :param ha: Hour angle (radians)
    :param dec: Declination (radians)
    :param lat: Site latitude (radians)
    :return:
    """
    return numpy.arctan2(numpy.cos(lat) * numpy.sin(ha),
                         (numpy.sin(lat) * numpy.cos(dec) - numpy.cos(lat) * numpy.sin(dec) * numpy.cos(ha)))


def pa_z(ha, dec, lat):
    """Calculate parallactic angle and zenith angle of source at ha, dec observed from site at latitude dec

    H = t - α
    sin(a) = sin(δ) sin(φ) + cos(δ) cos(φ) cos(H)
    sin(A) = - sin(H) cos(δ) / cos(a)
    cos(A) = { sin(δ) - sin(φ) sin(a) } / cos(φ) cos(a)
    
    :param ha: Hour angle (radians)
    :param dec: Declination (radians)
    :param lat: Site latitude (radians)
    :return:
    """
    sinz = numpy.sin(dec) * numpy.sin(lat) + numpy.cos(dec) * numpy.cos(lat) * numpy.cos(ha)
    return numpy.arctan2(numpy.cos(lat) * numpy.sin(ha),
                         (numpy.sin(lat) * numpy.cos(dec) - numpy.cos(lat) * numpy.sin(dec) * numpy.cos(ha))), \
           numpy.arcsin(sinz)


def hadec_to_azel(ha, dec, latitude):
    """ Convert HA Dec to Az El
    
    TMS Appendix 4.1
    
    sinel = sinlat sindec + coslat cosdec cosha
    cosel cosaz = coslat sindec - sinlat cosdec cosha
    cosel sinaz = - cosdec sinha
    
    :param ha:
    :param dec:
    :param latitude:
    :return: az, el
    """
    coslat = numpy.cos(latitude)
    sinlat = numpy.sin(latitude)
    cosdec = numpy.cos(dec)
    sindec = numpy.sin(dec)
    cosha = numpy.cos(ha)
    sinha = numpy.sin(ha)

    az = numpy.arctan2(- cosdec * sinha, (coslat * sindec - sinlat * cosdec * cosha))
    el = numpy.arcsin(sinlat * sindec + coslat * cosdec * cosha)
    return az, el


def azel_to_hadec(az, el, latitude):
    """Converting Az El to HA Dec
    
    TMS Appendix 4.1
    
    sindec = sinlat sinel + coslat cosel cosaz
    cosdec cosha = coslat sinel - sinlat cosel cosaz
    cosdec sinha = -cosel sinaz
    
    :param az:
    :param el:
    :param latitude:
    :return: ha, dec
    """
    cosel = numpy.cos(el)
    sinel = numpy.sin(el)
    coslat = numpy.cos(latitude)
    sinlat = numpy.sin(latitude)
    cosaz = numpy.cos(az)
    sinaz = numpy.sin(az)

    ha = numpy.arctan2(-cosel * sinaz, coslat * sinel - sinlat * cosel * cosaz)
    dec = numpy.arcsin(sinlat * sinel + coslat * cosel * cosaz)
    return ha, dec
