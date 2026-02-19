import math


# 这个DistAz类用于计算地球上两点（以经纬度表示）之间的大圆距离、方位角和反方位角等信息。
# 它基于大地测量学中的相关公式进行计算，提供了多种方法来获取计算结果，如获取距离（以千米为单位）、方位角、反方位角等。
class DistAz:
    """c
    c Subroutine to calculate the Great Circle Arc distance
    c    between two sets of geographic coordinates
    c
    c Equations take from Bullen, pages 154, 155
    c
    c T. Owens, September 19, 1991
    c           Sept. 25 -- fixed az and baz calculations
    c
    P. Crotwell, Setember 27, 1995
    Converted to c to fix annoying problem of fortran giving wrong
    answers if the input doesn't contain a decimal point.
    
    H. P. Crotwell, September 18, 1997
    Java version for direct use in java programs.
    *
    * C. Groves, May 4, 2004
    * Added enough convenience constructors to choke a horse and made public double
    * values use accessors so we can use this class as an immutable

    H.P. Crotwell, May 31, 2006
    Port to python, thus adding to the great list of languages to which
    distaz has been ported from the origin fortran: C, Tcl, Java and now python
    and I vaguely remember a perl port. Long live distaz! 
    """

    def __init__(self, stalat, stalon, evtlat, evtlon):
        # 纬度     经度   目标点的纬度  目标点的经度
        lat1 = stalat
        lon1 = stalon
        lat2 = evtlat
        lon2 = evtlon

        self.stalat = lat1
        self.stalon = lon1
        self.evtlat = lat2
        self.evtlon = lon2
        # 如果起始点和目标点相同（经纬度都相等），则将距离delta、方位角az和反方位角baz都设置为 0.0 并直接返回。
        if (lat1 == lat2) and (lon1 == lon2):
            self.delta = 0.0
            self.az = 0.0
            self.baz = 0.0
            return

        rad = 2. * math.pi / 360.0  # 角度转换为弧度
        """
	c
	c scolat and ecolat are the geocentric colatitudes
	c as defined by Richter (pg. 318)
	c
	c Earth Flattening of 1/298.257 take from Bott (pg. 3)
	c
        """
        # 根据地球扁率（sph = 1.0 / 298.257）计算两点的地心余纬度（scolat和ecolat）以及经度的弧度值（slon和elon）。
        sph = 1.0 / 298.257
        scolat = math.pi / 2.0 - math.atan((1. - sph) * (1. - sph) * math.tan(lat1 * rad))
        ecolat = math.pi / 2.0 - math.atan((1. - sph) * (1. - sph) * math.tan(lat2 * rad))
        slon = lon1 * rad
        elon = lon2 * rad
        # 接着按照 Bullen 书中（第 154 页，10.2 节）定义的公式计算一系列中间变量（a - e和aa - ee），
        # 这些变量用于后续计算距离、方位角和反方位角。
        """
	c
	c  a - e are as defined by Bullen (pg. 154, Sec 10.2)
	c     These are defined for the pt. 1
	c
        """
        a = math.sin(scolat) * math.cos(slon)
        b = math.sin(scolat) * math.sin(slon)
        c = math.cos(scolat)
        d = math.sin(slon)
        e = -math.cos(slon)
        g = -c * e
        h = c * d
        k = -math.sin(scolat)
        """
	c
	c  aa - ee are the same as a - e, except for pt. 2
	c
        """
        aa = math.sin(ecolat) * math.cos(elon)
        bb = math.sin(ecolat) * math.sin(elon)
        cc = math.cos(ecolat)
        dd = math.sin(elon)
        ee = -math.cos(elon)
        gg = -cc * ee
        hh = cc * dd
        kk = -math.sin(ecolat)

        # 根据相关公式计算出两点之间的距离delta（以度为单位）、反方位角baz（以度为单位）和方位角az（以度为单位），
        # 并确保0.0就是精确的0.0而不是接近360.0的值。

        """
	c
	c  Bullen, Sec 10.2, eqn. 4
	c
        """
        # 两点之间的距离delta（以度为单位）
        delrad = math.acos(a * aa + b * bb + c * cc)
        self.delta = delrad / rad
        """
	c
	c  Bullen, Sec 10.2, eqn 7 / eqn 8
	c
	c    pt. 1 is unprimed, so this is technically the baz
	c
	c  Calculate baz this way to avoid quadrant problems
	c
        """

        # 反方位角baz（以度为单位）
        rhs1 = (aa - d) * (aa - d) + (bb - e) * (bb - e) + cc * cc - 2.
        rhs2 = (aa - g) * (aa - g) + (bb - h) * (bb - h) + (cc - k) * (cc - k) - 2.
        dbaz = math.atan2(rhs1, rhs2)
        if (dbaz < 0.0):
            dbaz = dbaz + 2 * math.pi

        self.baz = dbaz / rad

        """
	c
	c  Bullen, Sec 10.2, eqn 7 / eqn 8
	c
	c    pt. 2 is unprimed, so this is technically the az
	c
	"""
        # 方位角az（以度为单位）
        rhs1 = (a - dd) * (a - dd) + (b - ee) * (b - ee) + c * c - 2.
        rhs2 = (a - gg) * (a - gg) + (b - hh) * (b - hh) + (c - kk) * (c - kk) - 2.
        daz = math.atan2(rhs1, rhs2)
        if daz < 0.0:
            daz = daz + 2 * math.pi

        self.az = daz / rad
        """
	c
	c   Make sure 0.0 is always 0.0, not 360.
	c
	"""
        # 确保0.0就是精确的0.0而不是接近360.0的值
        if (abs(self.baz - 360.) < .00001):
            self.baz = 0.0
        if (abs(self.az - 360.) < .00001):
            self.az = 0.0

    # 两点之间的距离（以度为单位）
    def getDelta(self):
        return self.delta

    # 获取两点之间的方位角
    def getAz(self):
        return self.az

    # 获取两点之间的反方位角
    def getBaz(self):
        return self.baz

    # 通过将距离（以度为单位）乘以 111.19 来得到两点之间的距离（以千米为单位），
    # 这里假设 1 度经度或纬度大约等于 111.19 千米。
    def getDistanceKm(self):
        return self.delta * 111.19

    # degreesToKilometers和kilometersToDegrees方法提供了度与千米之间的转换功能。
    def degreesToKilometers(self, degrees):
        return degrees * 111.19

    def kilometersToDegrees(self, kilometers):
        return kilometers / 111.19


distaz = DistAz(stalon=140.345, stalat=36.191, evtlon=140.412, evtlat=36.235)
# print("%f  azimuth:%f  backAzimth:%f" % (distaz.getDelta(), distaz.getAz(), distaz.getBaz()))
# from obspy.clients.iris import Client
# client=Client()
# print(client.distaz(stalon=60,stalat=10,evtlon=51,evtlat=11))
print(distaz.getDistanceKm())
