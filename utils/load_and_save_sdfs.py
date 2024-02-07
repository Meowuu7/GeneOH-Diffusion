from mesh_to_sdf import mesh_to_voxels
import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'

import trimesh
import skimage
import numpy as np

obj_mesh_path =  "/home/xueyi/sim/arctic/data/arctic_data/data/meta/object_vtemplates/box/mesh.obj"

mesh = trimesh.load(obj_mesh_path)

verts = mesh.vertices
maxx_verts = np.max(verts, axis=0)
minn_verts = np.min(verts, axis=0)
print(maxx_verts, minn_verts)
maxx_verts = np.max(maxx_verts).item()
minn_verts = np.min(minn_verts).item()
extent = maxx_verts - minn_verts
verts = (verts - minn_verts) / extent

voxels = mesh_to_voxels(mesh, 64, pad=True, sign_method='depth')

voxels_sv_path =  "/home/xueyi/sim/arctic/data/arctic_data/data/meta/object_vtemplates/box/voxels.npy"
np.save(voxels_sv_path, voxels)

# 
# vertices, faces, normals, _ = skimage.measure.marching_cubes(voxels, level=0)
# mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals)
# mesh.show()