from dataclasses import dataclass
import numpy as np

@dataclass
class Camera:
    fx: float = 800.; fy: float = 800.; cx: float = 640.; cy: float = 360.
    def project(self, points):
        p=np.asarray(points,float); z=np.maximum(p[:,2],1e-9)
        return np.column_stack((self.fx*p[:,0]/z+self.cx,self.fy*p[:,1]/z+self.cy))
