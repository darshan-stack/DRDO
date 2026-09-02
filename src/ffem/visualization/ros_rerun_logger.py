"""Optional ROS 2 to Rerun logger for FFEM output topics."""
from __future__ import annotations
try:
 import rclpy
 from rclpy.node import Node
 from sensor_msgs.msg import PointCloud2
 from std_msgs.msg import Float32MultiArray
 ROS_AVAILABLE=True
except ImportError: ROS_AVAILABLE=False
from ffem.ros2.pointcloud2_codec import decode_pointcloud2
try: import rerun as rr
except ImportError: rr=None
if ROS_AVAILABLE:
 class FFEMRerunLogger(Node):
  def __init__(self):
   super().__init__('ffem_rerun_logger'); self.declare_parameter('input_topic','/ffem_mapper/map/elevation'); self.declare_parameter('moving_topic','/ffem_mapper/map/moving_points'); self.declare_parameter('metrics_topic','/ffem_mapper/metrics'); self.declare_parameter('recording','outputs/ffem_ros.rrd'); self.frame=0; self.enabled=rr is not None
   if self.enabled: rr.init('ffem-ros2-logger',spawn=True)
   self.create_subscription(PointCloud2,str(self.get_parameter('input_topic').value),self.map_cb,5); self.create_subscription(PointCloud2,str(self.get_parameter('moving_topic').value),self.moving_cb,5); self.create_subscription(Float32MultiArray,str(self.get_parameter('metrics_topic').value),self.metrics_cb,5)
  def map_cb(self,msg):
   if not self.enabled:return
   p=decode_pointcloud2(msg); rr.set_time('frame',sequence=self.frame); rr.log('world/map/elevation',rr.Points3D(p.points)); self.frame+=1
  def moving_cb(self,msg):
   if self.enabled: rr.log('world/dynamics/moving',rr.Points3D(decode_pointcloud2(msg).points,colors=[240,60,50]))
  def metrics_cb(self,msg):
   if self.enabled and msg.data: rr.log('metrics/values',rr.Scalars([float(msg.data[0])]))
 def main(args=None):
  rclpy.init(args=args); n=FFEMRerunLogger()
  try:rclpy.spin(n)
  except KeyboardInterrupt:pass
  finally:
   if n.enabled:rr.save(str(n.get_parameter('recording').value))
   n.destroy_node();rclpy.shutdown()
else:
 class FFEMRerunLogger:
  def __init__(self):raise RuntimeError('ROS 2 is not installed.')
 def main(args=None):raise RuntimeError('ROS 2 is not installed.')
