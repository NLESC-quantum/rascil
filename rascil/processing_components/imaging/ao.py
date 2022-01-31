"""
Functions that implement creating Zernike polynomials to be applied as phase across the dish.

Note that we copied the implementation of zernike_noll and its related functions from package aotools (Ver 1.0.5)
to simplify the RASCIL installation. Copyright of function zernIndex and zernike_noll belongs to its 
respectful owner

AOTools: https://github.com/AOtools/aotools version 1.0.5 Releases 22 

"""

__all__ = ["zernike_noll"]


import numpy


def zernIndex(j):
    """
    Find the [n,m] list giving the radial order n and azimuthal order
    of the Zernike polynomial of Noll index j.

    Parameters:
        j (int): The Noll index for Zernike polynomials

    Returns:
        list: n, m values
    """
    n = int((-1.0 + numpy.sqrt(8 * (j - 1) + 1)) / 2.0)
    p = j - (n * (n + 1)) / 2.0
    k = n % 2
    m = int((p + k) / 2.0) * 2 - k

    if m != 0:
        if j % 2 == 0:
            s = 1
        else:
            s = -1
        m *= s

    return [n, m]


def circle(radius, size, circle_centre=(0, 0), origin="middle"):
    """
    Create a 2-D array: elements equal 1 within a circle and 0 outside.

    The default centre of the coordinate system is in the middle of the array:
    circle_centre=(0,0), origin="middle"
    This means:
    if size is odd  : the centre is in the middle of the central pixel
    if size is even : centre is in the corner where the central 4 pixels meet

    origin = "corner" is used e.g. by psfAnalysis:radialAvg()

    Examples: ::

        circle(1,5) circle(0,5) circle(2,5) circle(0,4) circle(0.8,4) circle(2,4)
          00000       00000       00100       0000        0000          0110
          00100       00000       01110       0000        0110          1111
          01110       00100       11111       0000        0110          1111
          00100       00000       01110       0000        0000          0110
          00000       00000       00100

        circle(1,5,(0.5,0.5))   circle(1,4,(0.5,0.5))
           .-->+
           |  00000               0000
           |  00000               0010
          +V  00110               0111
              00110               0010
              00000

    Parameters:
        radius (float)       : radius of the circle
        size (int)           : size of the 2-D array in which the circle lies
        circle_centre (tuple): coords of the centre of the circle
        origin (str)  : where is the origin of the coordinate system
                               in which circle_centre is given;
                               allowed values: {"middle", "corner"}

    Returns:
        ndarray (float64) : the circle array
    """
    # (2) Generate the output array:
    C = numpy.zeros((size, size))

    # (3.a) Generate the 1-D coordinates of the pixel's centres:
    # coords = numpy.linspace(-size/2.,size/2.,size) # Wrong!!:
    # size = 5: coords = array([-2.5 , -1.25,  0.  ,  1.25,  2.5 ])
    # size = 6: coords = array([-3. , -1.8, -0.6,  0.6,  1.8,  3. ])
    # (2015 Mar 30; delete this comment after Dec 2015 at the latest.)

    # Before 2015 Apr 7 (delete 2015 Dec at the latest):
    # coords = numpy.arange(-size/2.+0.5, size/2.-0.4, 1.0)
    # size = 5: coords = array([-2., -1.,  0.,  1.,  2.])
    # size = 6: coords = array([-2.5, -1.5, -0.5,  0.5,  1.5,  2.5])

    coords = numpy.arange(0.5, size, 1.0)
    # size = 5: coords = [ 0.5  1.5  2.5  3.5  4.5]
    # size = 6: coords = [ 0.5  1.5  2.5  3.5  4.5  5.5]

    # (3.b) Just an internal sanity check:
    if len(coords) != size:
        raise Exception(
            "len(coords) = {0}, ".format(len(coords))
            + "size = {0}. They must be equal.".format(size)
            + '\n           Debug the line "coords = ...".'
        )

    # (3.c) Generate the 2-D coordinates of the pixel's centres:
    x, y = numpy.meshgrid(coords, coords)

    # (3.d) Move the circle origin to the middle of the grid, if required:
    if origin == "middle":
        x -= size / 2.0
        y -= size / 2.0

    # (3.e) Move the circle centre to the alternative position, if provided:
    x -= circle_centre[0]
    y -= circle_centre[1]

    # (4) Calculate the output:
    # if distance(pixel's centre, circle_centre) <= radius:
    #     output = 1
    # else:
    #     output = 0
    mask = x * x + y * y <= radius * radius
    C[mask] = 1

    # (5) Return:
    return C


def zernikeRadialFunc(n, m, r):
    """
    Fucntion to calculate the Zernike radial function

    Parameters:
        n (int): Zernike radial order
        m (int): Zernike azimuthal order
        r (ndarray): 2-d array of radii from the centre the array

    Returns:
        ndarray: The Zernike radial function
    """

    R = numpy.zeros(r.shape)
    for i in range(0, int((n - m) / 2) + 1):

        R += numpy.array(
            r ** (n - 2 * i)
            * (((-1) ** (i)) * numpy.math.factorial(n - i))
            / (
                numpy.math.factorial(i)
                * numpy.math.factorial(0.5 * (n + m) - i)
                * numpy.math.factorial(0.5 * (n - m) - i)
            ),
            dtype="float",
        )
    return R


def zernike_nm(n, m, N):
    """
    Creates the Zernike polynomial with radial index, n, and azimuthal index, m.

    Args:
       n (int): The radial order of the zernike mode
       m (int): The azimuthal order of the zernike mode
       N (int): The diameter of the zernike more in pixels
    Returns:
       ndarray: The Zernike mode
    """
    coords = (numpy.arange(N) - N / 2.0 + 0.5) / (N / 2.0)
    X, Y = numpy.meshgrid(coords, coords)
    R = numpy.sqrt(X**2 + Y**2)
    theta = numpy.arctan2(Y, X)

    if m == 0:
        Z = numpy.sqrt(n + 1) * zernikeRadialFunc(n, 0, R)
    else:
        if m > 0:  # j is even
            Z = (
                numpy.sqrt(2 * (n + 1))
                * zernikeRadialFunc(n, m, R)
                * numpy.cos(m * theta)
            )
        else:  # i is odd
            m = abs(m)
            Z = (
                numpy.sqrt(2 * (n + 1))
                * zernikeRadialFunc(n, m, R)
                * numpy.sin(m * theta)
            )

    # clip
    Z = Z * numpy.less_equal(R, 1.0)

    return Z * circle(N / 2.0, N)


def zernike_noll(j, N):
    """
    Creates the Zernike polynomial with mode index j,
    where j = 1 corresponds to piston.

    Args:
       j (int): The noll j number of the zernike mode
       N (int): The diameter of the zernike more in pixels
    Returns:
       ndarray: The Zernike mode
    """

    n, m = zernIndex(j)
    return zernike_nm(n, m, N)
