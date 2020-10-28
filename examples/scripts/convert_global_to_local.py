# Convert ECEF to local
# See https://gist.github.com/govert/1b373696c9a27ff4c72a
import numpy
import math

from rascil.processing_components.simulation import create_named_configuration

def ecef_to_enu( x,  y,  z, lat0, lon0, h0):
    #  Convert to radians in notation consistent with the paper:
    a = 6378137.0         # WGS-84 Earth semimajor axis (m)
    b = 6356752.314245     # Derived Earth semiminor axis (m)
    f = (a - b) / a           # Ellipsoid Flatness
    f_inv = 1.0 / f       #Inverse flattening
    a_sq = a * a
    b_sq = b * b
    e_sq = f * (2 - f)    # Square of Eccentricity
    n_lambda = lat0*numpy.pi/180.
    phi = lon0*numpy.pi/180.
    s = numpy.sin(n_lambda)
    N = a / numpy.sqrt(1 - e_sq * s * s)
    sin_lambda = math.sin(n_lambda)
    cos_lambda = math.cos(n_lambda)
    cos_phi = math.cos(phi)
    sin_phi = math.sin(phi)
    x0 = (h0 + N) * cos_lambda * cos_phi
    y0 = (h0 + N) * cos_lambda * sin_phi
    z0 = (h0 + (1 - e_sq) * N) * sin_lambda
    xd = x - x0
    yd = y - y0
    zd = z - z0
    # This is the matrix multiplication
    xEast = -sin_phi * xd + cos_phi * yd
    yNorth = -cos_phi * sin_lambda * xd - sin_lambda * sin_phi * yd + cos_lambda * zd
    zUp = cos_lambda * cos_phi * xd + cos_lambda * sin_phi * yd + sin_lambda * zd
    return xEast,yNorth,zUp

if __name__ == "__main__":
    
    for telescope in ["MEERKAT+"]:
        config = create_named_configuration(telescope)
        nants = len(config.names)
        print(config.location)
        print(config.location.geodetic)
        lon0 = config.location.geodetic[0].to('rad').value
        lat0 = config.location.geodetic[1].to('rad').value
        h0 = config.location.geodetic[2].to('m').value

        x0, y0, z0 = config.location.value
        print(x0, y0, z0)
        for i, xyz in enumerate(config.xyz):
            x, y, z = list(xyz)
            east, north, z = ecef_to_enu(x - x0, y - y0, z - z0, lat0, lon0, h0)
            print(east, north, h0, config.diameter[i], config.names[i])
