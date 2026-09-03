"""Optional live Open3D viewer for FFEM results."""
from __future__ import annotations
import numpy as np
class Open3DViewer:
    def __init__(self, title='FFEM Adaptive LiDAR Map', enabled=True):
        self.enabled=enabled; self.vis=None; self.geoms={}
        if enabled:
            try:
                import open3d as o3d
                self.o3d=o3d; self.vis=o3d.visualization.Visualizer(); self.vis.create_window(window_name=title,width=1280,height=720)
            except ImportError as exc: raise RuntimeError('Install Open3D with: pip install open3d') from exc
    def _cloud(self,name,points,colors=None,size=1.0):
        o3d=self.o3d; cloud=self.geoms.get(name)
        if cloud is None: cloud=o3d.geometry.PointCloud(); self.geoms[name]=cloud; self.vis.add_geometry(cloud)
        cloud.points=o3d.utility.Vector3dVector(np.asarray(points,dtype=np.float64).reshape(-1,3))
        if colors is not None: cloud.colors=o3d.utility.Vector3dVector(np.asarray(colors,dtype=np.float64).reshape(-1,3)/255.0)
        return cloud
    def update(self,result,map_points,map_colors,levels):
        if not self.enabled:return
        pts=np.asarray(result['points']); moving=np.asarray(result['moving'],dtype=bool); self._cloud('raw',pts,np.tile([0.35,0.55,0.95],(len(pts),1))*255); self._cloud('moving',pts[moving],np.tile([1.0,0.1,0.1],(moving.sum(),1))*255); self._cloud('elevation',map_points,map_colors)
        self.vis.poll_events(); self.vis.update_renderer()
    def close(self):
        if self.vis:self.vis.destroy_window()
