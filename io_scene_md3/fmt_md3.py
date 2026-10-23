from math import pi, sin, cos, atan2, acos

from .utils import AnyStruct

import numpy as np


def string_from_bytes(b):
    return b.rstrip(b'\0').decode('utf-8', errors='ignore')


def string_to_bytes(s):
    return s.encode('utf-8')


# byte order follows Quake 3 (q3map2 NormalToLatLong):
# b[0] = polar angle from +Z, b[1] = azimuth around Z
def decode_normal(b):
    polar = b[0] / 255.0 * 2 * pi
    azimuth = b[1] / 255.0 * 2 * pi
    x = cos(azimuth) * sin(polar)
    y = sin(azimuth) * sin(polar)
    z = cos(polar)
    return (x, y, z)


def encode_normal(n):
    x, y, z = n
    if x == 0 and y == 0:
        return bytes((0, 0)) if z > 0 else bytes((128, 0))
    azimuth = int(atan2(y, x) * 255 / (2 * pi)) & 255
    polar = int(acos(np.clip(z, -1.0, 1.0)) * 255 / (2 * pi)) & 255
    return bytes((polar, azimuth))


VERTEX_SCALE = 64.0


def decode_vertex(v):
    return v / VERTEX_SCALE


def encode_vertex(v):
    return int(v * VERTEX_SCALE)


def texcoord_inverted(v):
    return 1.0 - v


Header = AnyStruct('Header', (
    ('magic', '4s'),
    ('version', 'i'),
    ('modelname', '64s', 1, string_from_bytes, string_to_bytes),
    ('flags', 'i'),
    ('nFrames', 'i'),
    ('nTags', 'i'),
    ('nSurfaces', 'i'),
    ('nSkins', 'i'),
    ('offFrames', 'i'),
    ('offTags', 'i'),
    ('offSurfaces', 'i'),
    ('offEnd', 'i'),
))

Surface = AnyStruct('Surface', (
    ('magic', '4s'),
    ('name', '64s', 1, string_from_bytes, string_to_bytes),
    ('flags', 'i'),
    ('nFrames', 'i'),
    ('nShaders', 'i'),
    ('nVerts', 'i'),
    ('nTris', 'i'),
    ('offTris', 'i'),
    ('offShaders', 'i'),
    ('offST', 'i'),
    ('offVerts', 'i'),
    ('offEnd', 'i'),
))

Frame = AnyStruct('Frame', (
    ('minBounds', '3f', 3),
    ('maxBounds', '3f', 3),
    ('localOrigin', '3f', 3),
    ('radius', 'f'),
    ('name', '16s', 1, string_from_bytes, string_to_bytes),
))

Tag = AnyStruct('Tag', (
    ('name', '64s', 1, string_from_bytes, string_to_bytes),
    ('origin', '3f', 3),
    ('axis', '9f', 9),
))

Shader = AnyStruct('Shader', (
    ('name', '64s', 1, string_from_bytes, string_to_bytes),
    ('index', 'i'),
))

Triangle = AnyStruct('Triangle', (
    ('a', 'i'),
    ('b', 'i'),
    ('c', 'i'),
))

TexCoord = AnyStruct('TexCoord', (
    ('s', 'f'),
    ('t', 'f', 1, texcoord_inverted, texcoord_inverted),
))

Vertex = AnyStruct('Vertex', (
    ('x', 'h', 1, decode_vertex, encode_vertex),
    ('y', 'h', 1, decode_vertex, encode_vertex),
    ('z', 'h', 1, decode_vertex, encode_vertex),
    ('normal', '2s', 1, decode_normal, encode_normal),
))


MAGIC = b'IDP3'
VERSION = 15
