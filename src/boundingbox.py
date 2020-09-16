
class BoundingBox:

    xmin: float
    ymin: float
    zmin: float
    xmax: float
    ymax: float
    zmax: float


    def __init__(self, xmin, ymin, zmin, xmax, ymax, zmax):
        self.xmin = xmin
        self.ymin = ymin
        self.zmin = zmin
        self.xmax = xmax
        self.ymax = ymax
        self.zmax = zmax


    def set_bbox(self,xmin, ymin, zmin, xmax, ymax, zmax):
        self.xmin = xmin
        self.ymin = ymin
        self.zmin = zmin
        self.xmax = xmax
        self.ymax = ymax
        self.zmax = zmax

    def get_bbox(self) -> [float]:
        return [self.xmin, self.ymin, self.zmin, self.xmax, self.ymax, self.zmax]

    def set_xmin(self, xmin):
        self.xmin = xmin

    def get_xmin(self) -> float:
        return self.xmin

    def set_ymin(self, ymin):
        self.ymin = ymin

    def get_ymin(self) -> float:
        return self.ymin

    def set_zmin(self, zmin):
        self.zmin = zmin

    def get_zmin(self) -> float:
        return self.zmin

    def set_xmax(self, xmax):
        self.xmax = xmax

    def get_xmax(self) -> float:
        return self.xmax

    def set_ymax(self, ymax):
        self.ymax = ymax

    def get_ymax(self) -> float:
        return self.ymax

    def set_zmax(self, zmax):
        self.zmax = zmax

    def get_zmax(self) -> float:
        return self.zmax