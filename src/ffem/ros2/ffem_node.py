"""Integrated ROS 2 FFEM node: PointCloud2 -> TF -> FFEM -> outputs/Rerun."""
from __future__ import annotations
import json, time
import numpy as np
try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
    from sensor_msgs.msg import PointCloud2
    from std_msgs.msg import Float32MultiArray, String
    from tf2_ros import Buffer, TransformListener, LookupException, ConnectivityException, ExtrapolationException
    from geometry_msgs.msg import TransformStamped
    ROS_AVAILABLE=True
except ImportError:
    ROS_AVAILABLE=False
from ffem.pipeline import FFEMPipeline, FFEMConfig
from ffem.perception.segmentation import NumpyFallbackSegmenter, TorchRangeSegmenter, ProjectionConfig
from ffem.ros2.pointcloud2_codec import decode_pointcloud2, encode_pointcloud2
from ffem.ros2.transforms import transform_points, transform_from_ros_transform
try:
    import rerun as rr
except ImportError: rr=None

if ROS_AVAILABLE:
    class FFEMNode(Node):
        def __init__(self):
            super().__init__('ffem_mapper')
            self.declare_parameter('input_topic','/points'); self.declare_parameter('map_frame','map'); self.declare_parameter('use_tf',True); self.declare_parameter('queue_depth',5); self.declare_parameter('max_points_per_frame',150000); self.declare_parameter('enable_rerun',False); self.declare_parameter('recording','outputs/ros_ffem.rrd'); self.declare_parameter('model_backend','fallback'); self.declare_parameter('checkpoint',''); self.declare_parameter('range_height',32); self.declare_parameter('range_width',1024); self.declare_parameter('max_range',80.0)
            self.declare_parameter('map_topic','~/map/elevation'); self.declare_parameter('semantic_topic','~/map/semantic'); self.declare_parameter('moving_topic','~/map/moving_points'); self.declare_parameter('metrics_topic','~/metrics'); self.declare_parameter('events_topic','~/refinement_events')
            cfg=FFEMConfig(base_cell_size=float(self.declare_parameter('base_cell_size',1.0).value),finest_cell_size=float(self.declare_parameter('finest_cell_size',0.25).value),max_level=int(self.declare_parameter('max_level',2).value),max_active_cells=int(self.declare_parameter('max_active_cells',20000).value),max_topology_changes=int(self.declare_parameter('max_topology_changes',32).value))
            backend=str(self.get_parameter('model_backend').value); checkpoint=str(self.get_parameter('checkpoint').value); segmenter=None
            if backend == 'torch_range':
                if not checkpoint: raise RuntimeError('model_backend=torch_range requires checkpoint=<path>')
                segmenter=TorchRangeSegmenter(checkpoint, ProjectionConfig(height=int(self.get_parameter('range_height').value),width=int(self.get_parameter('range_width').value),max_range=float(self.get_parameter('max_range').value)),num_classes=cfg.num_classes)
            elif backend == 'fallback': segmenter=NumpyFallbackSegmenter(cfg.num_classes)
            else: raise ValueError(f'Unknown model_backend: {backend}')
            self.pipeline=FFEMPipeline(cfg,segmenter=segmenter); self.frame=0; self.map_frame=str(self.get_parameter('map_frame').value); self.max_points=int(self.get_parameter('max_points_per_frame').value); self.use_tf=bool(self.get_parameter('use_tf').value); self.tf_buffer=Buffer(); self.tf_listener=TransformListener(self.tf_buffer,self)
            depth=int(self.get_parameter('queue_depth').value); qos=QoSProfile(depth=depth,history=HistoryPolicy.KEEP_LAST,reliability=ReliabilityPolicy.BEST_EFFORT)
            self.sub=self.create_subscription(PointCloud2,str(self.get_parameter('input_topic').value),self.callback,qos)
            self.map_pub=self.create_publisher(PointCloud2,str(self.get_parameter('map_topic').value),5); self.semantic_pub=self.create_publisher(PointCloud2,str(self.get_parameter('semantic_topic').value),5); self.moving_pub=self.create_publisher(PointCloud2,str(self.get_parameter('moving_topic').value),5); self.metrics_pub=self.create_publisher(Float32MultiArray,str(self.get_parameter('metrics_topic').value),5); self.events_pub=self.create_publisher(String,str(self.get_parameter('events_topic').value),5)
            self.rerun_enabled=bool(self.get_parameter('enable_rerun').value) and rr is not None
            if self.rerun_enabled: rr.init('ffem-ros2',spawn=True)
            self.get_logger().info(f'FFEM integrated node listening on {self.get_parameter("input_topic").value}; backend={backend}; TF={self.use_tf}; Rerun={self.rerun_enabled}')
        def _lookup(self,msg):
            if not self.use_tf or msg.header.frame_id==self.map_frame: return np.eye(4,dtype=np.float64)
            try:
                t=self.tf_buffer.lookup_transform(self.map_frame,msg.header.frame_id,msg.header.stamp)
                return transform_from_ros_transform(t.transform)
            except (LookupException,ConnectivityException,ExtrapolationException) as exc:
                self.get_logger().warning(f'TF unavailable {msg.header.frame_id}->{self.map_frame}: {exc}', throttle_duration_sec=5.0); return None
        def callback(self,msg):
            try:
                decoded=decode_pointcloud2(msg,remove_invalid=True)
                if not len(decoded.points): return
                idx=np.linspace(0,len(decoded.points)-1,min(len(decoded.points),self.max_points),dtype=int) if len(decoded.points)>self.max_points else slice(None)
                points=decoded.points[idx]; intensity=None if decoded.intensity is None else decoded.intensity[idx]
                matrix=self._lookup(msg)
                if matrix is None: return
                points=transform_points(points,matrix); result=self.pipeline.process_points(points,intensity=intensity,frame=self.frame); self._publish(result,msg.header.stamp); self._log_rerun(result); self.frame+=1
            except Exception as exc: self.get_logger().error(f'FFEM callback failed: {exc}')
        def _publish(self,result,stamp):
            mp,_,_=self.pipeline.mapping.arrays(); self.map_pub.publish(encode_pointcloud2(mp,frame_id=self.map_frame,stamp=stamp)); labels=np.argmax(result['semantic_probs'],axis=1).astype(np.float32); self.semantic_pub.publish(encode_pointcloud2(result['points'],frame_id=self.map_frame,stamp=stamp,intensity=labels)); moving=result['points'][result['moving']]; self.moving_pub.publish(encode_pointcloud2(moving,frame_id=self.map_frame,stamp=stamp)); s=result['stats']; m=Float32MultiArray(); m.data=[float(s[k]) for k in ('total_ms','map_ms','active_cells','topology_changes','points','moving_points')]; self.metrics_pub.publish(m)
            if self.pipeline.mapping.events: e=String(); e.data=json.dumps(self.pipeline.mapping.events[-1]); self.events_pub.publish(e)
        def _log_rerun(self,result):
            if not self.rerun_enabled:return
            s=result['stats']; mp,col,levels=self.pipeline.mapping.arrays(); rr.set_time('frame',sequence=self.frame); rr.log('world/lidar/input',rr.Points3D(result['points'])); rr.log('world/lidar/semantic',rr.Points3D(result['points'],colors=np.array([[int(30+30*(i%7)),int(200-20*(i%7)),int(80+20*(i%7))] for i in np.argmax(result['semantic_probs'],axis=1)],dtype=np.uint8))); rr.log('world/dynamics/moving',rr.Points3D(result['points'][result['moving']],colors=[240,60,50])); rr.log('world/map/elevation',rr.Points3D(mp,colors=col)); rr.log('world/map/adaptive_cells',rr.Points3D(mp,radii=0.04+0.03*levels,colors=col)); rr.log('metrics/total_ms',rr.Scalars([s['total_ms']])); rr.log('metrics/active_cells',rr.Scalars([s['active_cells']]))
    def main(args=None):
        rclpy.init(args=args); node=FFEMNode()
        try:rclpy.spin(node)
        except KeyboardInterrupt:pass
        finally:
            if node.rerun_enabled: rr.save(str(node.get_parameter('recording').value))
            node.destroy_node();rclpy.shutdown()
else:
    class FFEMNode:
        def __init__(self): raise RuntimeError('ROS 2 is not installed.')
    def main(args=None): raise RuntimeError('ROS 2 is not installed.')
if __name__=='__main__':main()
