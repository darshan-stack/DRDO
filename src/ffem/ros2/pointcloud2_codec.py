"""Robust PointCloud2 binary codec.

The decoder is ROS-independent and can be unit-tested with a lightweight
message-like object. It honors row_step, point_step, field offsets, datatype,
endianness, organized clouds, and invalid-point filtering.
"""
from __future__ import annotations
from dataclasses import dataclass
import struct
import numpy as np

# sensor_msgs/PointField datatype constants
INT8, UINT8, INT16, UINT16, INT32, UINT32, FLOAT32, FLOAT64 = range(1, 9)
_FORMATS = {INT8:'b', UINT8:'B', INT16:'h', UINT16:'H', INT32:'i', UINT32:'I', FLOAT32:'f', FLOAT64:'d'}
_SIZES = {k: struct.calcsize(v) for k, v in _FORMATS.items()}

@dataclass
class DecodedPointCloud:
    points: np.ndarray
    intensity: np.ndarray | None = None
    ring: np.ndarray | None = None
    time: np.ndarray | None = None
    width: int = 0
    height: int = 1
    frame_id: str = ''
    stamp_ns: int = 0


def _field_map(msg):
    return {str(f.name).lower(): f for f in msg.fields}


def _read_value(data: bytes, offset: int, field, endian: str):
    datatype = int(field.datatype)
    if datatype not in _FORMATS:
        raise ValueError(f'Unsupported PointField datatype: {datatype}')
    count = int(getattr(field, 'count', 1) or 1)
    fmt = endian + (str(count) if count != 1 else '') + _FORMATS[datatype]
    return struct.unpack_from(fmt, data, offset)[0 if count == 1 else slice(None)]


def decode_pointcloud2(msg, *, remove_invalid: bool = True, fields: tuple[str, ...] = ('x','y','z','intensity','ring','time')) -> DecodedPointCloud:
    """Decode a sensor_msgs/PointCloud2-like message into NumPy arrays.

    The implementation avoids dtype reinterpretation assumptions and therefore
    works with padding, non-contiguous rows, big-endian payloads, and arbitrary
    field offsets. It returns points in the message's native frame.
    """
    fmap = _field_map(msg)
    missing = [name for name in ('x','y','z') if name not in fmap]
    if missing: raise ValueError(f'PointCloud2 is missing required fields: {missing}')
    width, height = int(msg.width), int(msg.height)
    point_step, row_step = int(msg.point_step), int(msg.row_step)
    raw = bytes(msg.data)
    expected = row_step * height
    if len(raw) < expected: raise ValueError(f'PointCloud2 data truncated: {len(raw)} < {expected}')
    endian = '>' if bool(getattr(msg, 'is_bigendian', False)) else '<'
    values = {name: [] for name in fields if name in fmap}
    for row in range(height):
        row_base = row * row_step
        for col in range(width):
            base = row_base + col * point_step
            for name, field in fmap.items():
                if name in values:
                    values[name].append(_read_value(raw, base + int(field.offset), field, endian))
    xyz = np.asarray(values['x'], dtype=np.float32), np.asarray(values['y'], dtype=np.float32), np.asarray(values['z'], dtype=np.float32)
    points = np.column_stack(xyz).astype(np.float32, copy=False)
    valid = np.isfinite(points).all(axis=1)
    if remove_invalid:
        points = points[valid]
    def optional(name, dtype=np.float32):
        if name not in values: return None
        arr = np.asarray(values[name], dtype=dtype)
        return arr[valid] if remove_invalid else arr
    stamp = getattr(getattr(msg, 'header', None), 'stamp', None)
    stamp_ns = 0 if stamp is None else int(getattr(stamp, 'sec', 0))*1_000_000_000 + int(getattr(stamp, 'nanosec', 0))
    return DecodedPointCloud(points, optional('intensity'), optional('ring', np.int32), optional('time'), width, height, getattr(getattr(msg, 'header', None), 'frame_id', ''), stamp_ns)


def encode_pointcloud2(points: np.ndarray, *, frame_id: str = 'base_link', stamp=None, intensity=None, ros_types=None):
    """Create a sensor_msgs/PointCloud2 message when ROS message types exist."""
    if ros_types is None:
        from sensor_msgs.msg import PointCloud2, PointField
    else:
        PointCloud2, PointField = ros_types
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    intensity = None if intensity is None else np.asarray(intensity, dtype=np.float32).reshape(-1)
    if intensity is not None and len(intensity) != len(points): raise ValueError('intensity length must match points')
    fields = [PointField(name=n, offset=o, datatype=PointField.FLOAT32, count=1) for n,o in [('x',0),('y',4),('z',8)]]
    if intensity is not None: fields.append(PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1))
    step = 16 if intensity is not None else 12
    payload = np.empty((len(points), step//4), dtype='<f4'); payload[:, :3] = points
    if intensity is not None: payload[:, 3] = intensity
    msg = PointCloud2(); msg.header.frame_id = frame_id
    if stamp is not None: msg.header.stamp = stamp
    msg.height=1; msg.width=len(points); msg.fields=fields; msg.is_bigendian=False; msg.point_step=step; msg.row_step=step*len(points); msg.is_dense=bool(np.isfinite(payload).all()); msg.data=payload.tobytes()
    return msg
