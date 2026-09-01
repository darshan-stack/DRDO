import struct
from types import SimpleNamespace
import numpy as np
from ffem.ros2.pointcloud2_codec import decode_pointcloud2, FLOAT32, UINT16


def make_msg(points, *, big=False, row_padding=0, with_ring=False):
    endian = '>' if big else '<'
    fields = [SimpleNamespace(name='x', offset=0, datatype=FLOAT32, count=1), SimpleNamespace(name='y', offset=4, datatype=FLOAT32, count=1), SimpleNamespace(name='z', offset=8, datatype=FLOAT32, count=1)]
    step = 12
    if with_ring:
        fields.append(SimpleNamespace(name='ring', offset=12, datatype=UINT16, count=1)); step = 14
    width, height = 2, 2
    rows=[]
    for row in range(height):
        payload=b''
        for col in range(width):
            x,y,z=points[row*width+col]
            payload += struct.pack(endian+'fff', x,y,z)
            if with_ring: payload += struct.pack(endian+'H', row*width+col)
        rows.append(payload + b'P'*row_padding)
    return SimpleNamespace(width=width,height=height,point_step=step,row_step=step*width+row_padding,is_bigendian=big,data=b''.join(rows),fields=fields,header=SimpleNamespace(frame_id='lidar',stamp=SimpleNamespace(sec=2,nanosec=3)))


def test_decode_little_endian_with_padding_and_ring():
    msg=make_msg([(1,2,3),(4,5,6),(7,8,9),(10,11,12)], with_ring=True, row_padding=3)
    out=decode_pointcloud2(msg)
    np.testing.assert_allclose(out.points, [[1,2,3],[4,5,6],[7,8,9],[10,11,12]])
    assert out.ring.tolist()==[0,1,2,3] and out.frame_id=='lidar' and out.stamp_ns==2000000003


def test_decode_big_endian():
    msg=make_msg([(1.5,2.5,3.5),(4,5,6),(7,8,9),(10,11,12)], big=True)
    out=decode_pointcloud2(msg)
    np.testing.assert_allclose(out.points[0], [1.5,2.5,3.5])


def test_invalid_points_are_removed():
    msg=make_msg([(float('nan'),2,3),(4,5,6),(7,8,9),(10,11,12)])
    out=decode_pointcloud2(msg)
    assert len(out.points)==3
