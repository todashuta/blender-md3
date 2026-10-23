# grouping to surfaces must done by UV maps also, not only normals
# TODO: merge surfaces with same uv maps and texture
# TODO: check bounding sphere calculation


import re
from collections import defaultdict
from math import sqrt

import bpy
import mathutils

from . import fmt_md3 as fmt
from .utils import OffsetBytesIO

nums = re.compile(r'\.\d{3}$')

# md3 names are char[64]; Quake 3 reads them as NUL-terminated strings
MAX_NAME_BYTES = 63


class MD3ExportError(Exception):
    pass


def prepare_name(name):
    if nums.findall(name):
        return name[:-4]  # cut off blender's .001 .002 etc
    return name

def get_textures(material):
    textures = []
    if material.node_tree:
        for tex in material.node_tree.nodes:
            if tex.type=='TEX_IMAGE':
                textures.append(tex)
    return textures


def return_single_mat(uv_maps,materials):
    if len(uv_maps) == 1:
        return uv_maps.active.name, materials[0]
    elif len(uv_maps) > 1:
        print('Warning: Multiple Shaders or/and UV maps found, only first will be chosen')
        return uv_maps.active.name, materials[0]


def gather_shader_info(mesh):
    'Returning uvmap name, texture name list'
    uv_maps = mesh.uv_layers
    materials = [] 
    for mat in mesh.materials:
        materials.append(mat.name)

    if len(uv_maps) <= 0:
        print('Warning: No UV maps found, zero filling will be used')
        return None, []
    
    if len(materials) <= 0:
        print('Warning: No Materials found, zero filling will be used')
        return None, []

    if len(materials) == 1:
        return return_single_mat(uv_maps,materials)
    else:
        print('Warning: Multiple Shaders found, only first will be chosen')
        return return_single_mat(uv_maps,materials)

def get_normal_matrix(mesh_matrix):
    # 4x4 @ Vector3 treats the vector as a point (adds translation, keeps scale),
    # so normals need the inverse-transpose of the 3x3 part instead
    return mesh_matrix.to_3x3().inverted_safe().transposed()


def gather_vertices(mesh, uvmap_data=None):
    md3vert_to_loop_map = []
    loop_to_md3vert_map = []
    index = {}
    for i, loop in enumerate(mesh.loops):
        # the key only needs to tell normals apart, so transforming them here is unnecessary
        key = (
            loop.vertex_index,
            tuple(loop.normal),
            None if uvmap_data is None else tuple(uvmap_data[i].uv),
        )
        md3id = index.get(key, None)
        if md3id is None:
            md3id = len(md3vert_to_loop_map)
            index[key] = md3id
            md3vert_to_loop_map.append(i)
        loop_to_md3vert_map.append(md3id)

    return md3vert_to_loop_map, loop_to_md3vert_map


def interp(a, b, t):
    return (b - a) * t + a


def find_interval(vs, t):
    a, b = 0, len(vs) - 1
    if t < vs[a]:
        return None, a
    if t > vs[b]:
        return b, None
    while b - a > 1:
        c = (a + b) // 2
        if vs[c] > t:
            b = c
        else:
            a = c
    assert vs[a] <= t <= vs[b]
    return a, b


class MD3Exporter:
    def __init__(self, context):
        self.context = context

    @property
    def scene(self):
        return self.context.scene

    def get_mesh_matrix(self, obj):
        from mathutils import Matrix
        from math import radians
        M = obj.matrix_world.copy()
        if self.enable_convert:
            M = Matrix.Scale(39.3701, 4) @ Matrix.Rotation(radians(90), 4, 'Z') @ M
        if self.adjust_scale_bug:
            M = Matrix.Scale(0.254/0.25, 4) @ M
        return M

    def check_name(self, kind, name):
        # an error rather than silent truncation: .skin files look surfaces up by name,
        # and cutting UTF-8 in the middle of a character would break the match
        size = len(fmt.string_to_bytes(name))
        if size > MAX_NAME_BYTES:
            raise MD3ExportError('{}名 "{}" が長すぎます（{} バイト、上限 {} バイト）'.format(
                kind, name, size, MAX_NAME_BYTES))

    def check_duplicates(self, kind, obj_names):
        # only a warning: sharing a name (e.g. Body and Body.001) may be intended
        seen = defaultdict(list)
        for n in obj_names:
            seen[prepare_name(n)].append(n)
        for exported, originals in seen.items():
            if len(originals) > 1:
                self.warnings.append('{}名 "{}" が重複しています: {}'.format(
                    kind, exported, ', '.join(originals)))

    def validate_names(self):
        self.check_name('モデル', self.scene.name)
        for n in self.surfNames:
            self.check_name('サーフェス', prepare_name(n))
        for n in self.tagNames:
            self.check_name('タグ', prepare_name(n))
        self.check_duplicates('サーフェス', self.surfNames)
        self.check_duplicates('タグ', self.tagNames)

    def pack_tag(self, name):
        tag = self.scene.objects[name]
        # same matrix as meshes so tags stay attached to them after conversion;
        # scale is dropped because md3 tag axes must be unit vectors
        loc, rot, _ = self.get_mesh_matrix(tag).decompose()
        axes = rot.to_matrix()
        return fmt.Tag.pack(
            name=prepare_name(tag.name),
            origin=tuple(loc),
            axis=sum([tuple(axes.col[j]) for j in range(3)], ()),
        )

    def pack_animated_tags(self):
        tags_bin = []
        for frame in range(self.nFrames):
            self.switch_frame(frame)
            for name in self.tagNames:
                tags_bin.append(self.pack_tag(name))
        return b''.join(tags_bin)

    def pack_surface_shader(self, i):
        return fmt.Shader.pack(
            name=prepare_name(self.mesh_shader_list),
            index=i,
        )

    def pack_surface_triangle(self, i):
        assert self.mesh.polygons[i].loop_total == 3
        start = self.mesh.polygons[i].loop_start
        a, b, c = (self.mesh_loop_to_md3vert[j] for j in range(start, start + 3))
        return fmt.Triangle.pack(a, c, b)  # swapped c/b

    def get_evaluated_vertex_co(self, frame, i):
        co = self.mesh.vertices[i].co.copy()

        if self.mesh_sk_rel is not None:
            bco = co.copy()
            for ki, k in enumerate(self.mesh.shape_keys.key_blocks):
                co += (k.data[i].co - bco) * self.mesh_sk_rel[ki]
        elif self.mesh_sk_abs is not None:
            kbs = self.mesh.shape_keys.key_blocks
            a, b, t = self.mesh_sk_abs
            co = interp(kbs[a].data[i].co, kbs[b].data[i].co, t)

        co = self.mesh_matrix @ co
        self.mesh_vco[frame].append(co)
        return co

    def pack_surface_vert(self, frame, i):
        loop_id = self.mesh_md3vert_to_loop[i]
        vert_id = self.mesh.loops[loop_id].vertex_index
        return fmt.Vertex.pack(
            *self.get_evaluated_vertex_co(frame, vert_id),
            normal=tuple((self.normal_matrix @ self.mesh.loops[loop_id].normal).normalized()))

    def pack_surface_ST(self, i):
        if self.mesh_uvmap_name is None:
            s, t = 0.0, 0.0
        else:
            loop_idx = self.mesh_md3vert_to_loop[i]
            s, t = self.mesh.uv_layers[self.mesh_uvmap_name].data[loop_idx].uv
        return fmt.TexCoord.pack(s, t)

    def switch_frame(self, i):
        self.scene.frame_set(self.scene.frame_start + i)

    def surface_start_frame(self, i):
        self.switch_frame(i)
        obj = bpy.context.view_layer.objects.active

        self.mesh_matrix = self.get_mesh_matrix(obj)
        self.normal_matrix = get_normal_matrix(self.mesh_matrix)
        #print(self.enable_convert, self.mesh_matrix.to_euler(), self.mesh_matrix.to_scale())

        obj.update_from_editmode()
        dg = bpy.context.evaluated_depsgraph_get()
        self.mesh_owner = obj.evaluated_get(dg)
        self.mesh = self.mesh_owner.to_mesh()

        self.mesh_sk_rel = None
        self.mesh_sk_abs = None

        shape_keys = self.mesh.shape_keys
        if shape_keys is not None:
            kblocks = shape_keys.key_blocks
            if shape_keys.use_relative:
                self.mesh_sk_rel = [k.value for k in kblocks]
            else:
                e = shape_keys.eval_time / 100.0
                a, b = find_interval([k.frame for k in kblocks], e)
                if a is None:
                    self.mesh_sk_abs = (b, b, 0.0)
                elif b is None:
                    self.mesh_sk_abs = (a, a, 0.0)
                else:
                    self.mesh_sk_abs = (a, b, (e - kblocks[a].frame) / (kblocks[b].frame - kblocks[a].frame))

    def pack_surface(self, surf_name):
        self.scene.frame_set(self.scene.frame_start)
        obj = self.scene.objects[surf_name]
        bpy.context.view_layer.objects.active = obj
        tris_mod = obj.modifiers.new(name="Triangulate", type='TRIANGULATE')  # no 4-gons or n-gons
        tris_mod.quad_method = 'FIXED'
        self.mesh_owner = None
        # the temporary modifier must not be left on the user's object when packing fails
        # (struct.error on out-of-range vertices is an expected failure)
        try:
            return self.pack_surface_data(obj)
        finally:
            # release here, to_mesh used for every frame
            if self.mesh_owner is not None:
                self.mesh_owner.to_mesh_clear()
                self.mesh_owner = None
            # not bpy.ops.object.modifier_remove: it depends on context and
            # may pick a user's own modifier with the same name
            obj.modifiers.remove(tris_mod)

    def pack_surface_data(self, obj):
        self.mesh_matrix = self.get_mesh_matrix(obj)

        obj.update_from_editmode()
        dg = bpy.context.evaluated_depsgraph_get()
        self.mesh_owner = obj.evaluated_get(dg)
        self.mesh = self.mesh_owner.to_mesh()

        self.mesh_uvmap_name, self.mesh_shader_list = gather_shader_info(self.mesh)
        if self.mesh_shader_list:
            self.check_name('シェーダー', prepare_name(self.mesh_shader_list))
        self.mesh_md3vert_to_loop, self.mesh_loop_to_md3vert = gather_vertices(
            self.mesh,
            None if self.mesh_uvmap_name is None else self.mesh.uv_layers[self.mesh_uvmap_name].data)

        nShaders = len(self.mesh_shader_list)
        nVerts = len(self.mesh_md3vert_to_loop)
        nTris = len(self.mesh.polygons)

        f = OffsetBytesIO(start_offset=fmt.Surface.size)
        f.mark('offShaders')
        f.write(b''.join([self.pack_surface_shader(i) for i in range(nShaders)]))
        f.mark('offTris')
        f.write(b''.join([self.pack_surface_triangle(i) for i in range(nTris)]))
        f.mark('offST')
        f.write(b''.join([self.pack_surface_ST(i) for i in range(nVerts)]))
        f.mark('offVerts')

        for frame in range(self.nFrames):
            self.surface_start_frame(frame)
            f.write(b''.join([self.pack_surface_vert(frame, i) for i in range(nVerts)]))

        f.mark('offEnd')

        print('Surface {}: nVerts={}{} nTris={}{} nShaders={}{}'.format(
            obj.name,
            nVerts, ' (Too many!)' if nVerts > 4096 else '',
            nTris, ' (Too many!)' if nTris > 8192 else '',
            nShaders, ' (Too many!)' if nShaders > 256 else '',
        ))

        return fmt.Surface.pack(
            magic=fmt.MAGIC,
            name=prepare_name(obj.name),
            flags=0,  # ignored
            nFrames=self.nFrames,
            nShaders=nShaders,
            nVerts=nVerts,
            nTris=nTris,
            **f.getoffsets()
        ) + f.getvalue()


    def get_frame_data(self, i):
        center = mathutils.Vector((0.0, 0.0, 0.0))
        x1, x2, y1, y2, z1, z2 = [0.0] * 6
        first = True
        for co in self.mesh_vco[i]:
            if first:
                x1, x2 = co.x, co.x
                y1, y2 = co.y, co.y
                z1, z2 = co.z, co.z
            else:
                x1, y1, z1 = min(co.x, x1), min(co.y, y1), min(co.z, z1)
                x2, y2, z2 = max(co.x, x2), max(co.y, y2), max(co.z, z2)
            first = False
            center += co
        if len(self.mesh_vco[i]):  # issue #9
            center /= len(self.mesh_vco[i])  # TODO: can be very distorted
        r = 0.0
        for co in self.mesh_vco[i]:
            r = max(r, (co - center).length_squared)
        r = sqrt(r)
        return {
            'minBounds': (x1, y1, z1),
            'maxBounds': (x2, y2, z2),
            'radius': r,  # TODO: not sure the radius is measured from center, and not localOrigin
        }

    def pack_frame(self, i):
        return fmt.Frame.pack(
            localOrigin=(0.0, 0.0, 0.0),
            name='',  # frame name, ignored, TODO:
            **self.get_frame_data(i)
        )

    def __call__(self, filename, enable_convert, adjust_scale_bug):
        self.enable_convert = enable_convert
        self.adjust_scale_bug = adjust_scale_bug
        self.nFrames = self.scene.frame_end - self.scene.frame_start + 1
        self.surfNames = []
        self.tagNames = []
        for o in bpy.context.selected_objects:
            if o.hide_get():  # skip hidden objects
                continue
            if o.type == 'MESH':
                self.surfNames.append(o.name)
            elif o.type == 'EMPTY' and o.empty_display_type == 'ARROWS':
                self.tagNames.append(o.name)
        self.warnings = []
        # before packing, so nothing is written or modified for a bad name
        self.validate_names()
        for w in self.warnings:
            print('WARNING:', w)
        self.mesh_vco = defaultdict(list)

        tags_bin = self.pack_animated_tags()
        surfaces_bin = [self.pack_surface(name) for name in self.surfNames]
        frames_bin = [self.pack_frame(i) for i in range(self.nFrames)]

        if len(surfaces_bin) == 0:
            print("WARNING: There're no visible surfaces to export")

        f = OffsetBytesIO(start_offset=fmt.Header.size)
        f.mark('offFrames')
        f.write(b''.join(frames_bin))
        f.mark('offTags')
        f.write(tags_bin)
        f.mark('offSurfaces')
        f.write(b''.join(surfaces_bin))
        f.mark('offEnd')

        with open(filename, 'wb') as file:
            file.write(fmt.Header.pack(
                magic=fmt.MAGIC,
                version=fmt.VERSION,
                modelname=self.scene.name,
                flags=0,  # ignored
                nFrames=self.nFrames,
                nTags=len(self.tagNames),
                nSurfaces=len(surfaces_bin),
                nSkins=0,  # count of skins, ignored
                **f.getoffsets()
            ))
            file.write(f.getvalue())
            print('nFrames={} nSurfaces={}'.format(self.nFrames, len(surfaces_bin)))
