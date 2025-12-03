import shutil
import os
import json
import copy
import multiprocessing as mp
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import trimesh
import igraph
from PIL import Image
from omegaconf import OmegaConf
from trimesh.base import Trimesh, Scene
from tqdm import tqdm
from natsort import natsorted

from samesh.data.common import NumpyTensor
from samesh.data.loaders import read_scene, remove_texture, scene2mesh
from samesh.renderer.renderer import Renderer, render_multiview, colormap_faces, colormap_norms
from samesh.models.sam import SamModel, Sam2Model, combine_bmasks, colormap_mask, remove_artifacts, point_grid_from_mask
from samesh.utils.cameras import *
from samesh.utils.timer import timer
from samesh.utils.mesh import duplicate_verts
from samesh.models.shape_diameter_function import *


@timer
def colormap_faces_mesh(mesh: Trimesh, face2label: dict[int, int], background=np.array([0, 0, 0])) -> Trimesh:
    """
    """
    label_max = max(face2label.values())
    palette = RandomState(0).randint(0, 255, (label_max + 1, 3)) # +1 for unlabeled faces
    palette[0] = background
    mesh = remove_texture(mesh, visual_kind='vertex')
    mesh = duplicate_verts(mesh) # needed to prevent face color interpolation
    faces_colored = set()
    for face, label in face2label.items():
        mesh.visual.face_colors[face, :3] = palette[label]
        faces_colored.add(face)
    #print(np.unique(mesh.visual.face_colors, axis=0, return_counts=True))
    '''
    for face in range(len(mesh.faces)):
        if face not in faces_colored:
            mesh.visual.face_colors[face, :3] = background
            print('Unlabeled face ', face)
    '''
    return mesh


@timer
def split_mesh_by_label(mesh: trimesh.Trimesh,
                        face2label: dict[int, int]) -> trimesh.Scene:
    """
    Split a mesh into multiple sub-meshes according to face2label.
    Each label becomes a separate trimesh.Trimesh inside a Scene.
    """

    # Collect face indices for each label
    label2faces = defaultdict(list)
    for f, label in face2label.items():
        label2faces[label].append(f)

    scene = trimesh.Scene()

    for label, face_ids in label2faces.items():
        face_ids = np.array(face_ids, dtype=int)

        # extract submesh by face index
        submesh = mesh.submesh([face_ids], append=True, repair=False)

        # Give each submesh a name
        submesh_name = f"label_{label}"

        # Add to scene
        scene.add_geometry(submesh, node_name=submesh_name)

    return scene

@timer
def split_mesh_by_label_optimized(mesh: trimesh.Trimesh,
                                 face2label: Dict[int, int]) -> trimesh.Scene:
    """
    Optimized version: Split a mesh into multiple sub-meshes according to face2label.
    """
    # 优化1: 使用数组操作替代字典循环
    labels = np.array([face2label[i] for i in range(len(face2label))])
    unique_labels = np.unique(labels)
    
    # 优化2: 预分配数据结构
    label2faces = {}
    for label in unique_labels:
        label2faces[label] = np.where(labels == label)[0]
    
    scene = trimesh.Scene()
    
    # 优化3: 批量处理并避免重复的内存分配
    for label, face_ids in label2faces.items():
        if len(face_ids) == 0:
            continue
            
        # 优化4: 直接使用面片索引提取子网格，避免不必要的复制
        # 使用更高效的提取方法
        submesh = extract_submesh_fast(mesh, face_ids)
        
        submesh_name = f"label_{label}"
        scene.add_geometry(submesh, node_name=submesh_name)
    
    return scene

def extract_submesh_fast(mesh: trimesh.Trimesh, face_indices: np.ndarray) -> trimesh.Trimesh:
    """
    快速提取子网格的辅助函数，避免submesh方法的开销
    """
    # 获取选中的面片
    selected_faces = mesh.faces[face_indices]
    
    # 找到这些面片使用的唯一顶点索引
    unique_vertices = np.unique(selected_faces.flatten())
    
    # 创建顶点索引映射：旧索引 -> 新索引
    vertex_map = np.zeros(len(mesh.vertices), dtype=np.int32)
    vertex_map[unique_vertices] = np.arange(len(unique_vertices))
    
    # 重新映射面片顶点索引
    new_faces = vertex_map[selected_faces]
    
    # 提取对应的顶点
    new_vertices = mesh.vertices[unique_vertices]
    
    # 创建新网格，避免不必要的处理
    return trimesh.Trimesh(vertices=new_vertices, 
                           faces=new_faces,
                           process=False,  # 禁用自动处理以提升速度
                           validate=False)  # 禁用验证以提升速度

def norms_mask(norms: NumpyTensor['h w 3'], cam2world: NumpyTensor['4 4'], threshold=0.0) -> NumpyTensor['h w 3']:
    """
    Mask pixels that are directly facing camera
    """
    lookat = cam2world[:3, :3] @ np.array([0, 0, 1])
    return np.abs(np.dot(norms, lookat)) > threshold


@timer
def load_items(path: Path) -> dict[str, list]:
    """
    """
    print('Loading items from cache...')

    filenames = list(path.glob('matte_*.png'))
    filenames = natsorted(filenames, key=lambda x: int(x.stem.split('_')[-1]))
    items = {
        'matte' : list(map(Image.open, filenames)),
        'faces' : [np.load(path / f'faces_{i}.npy') for i in range(len(filenames))],
        'norms' : [np.load(path / f'norms_{i}.npy') for i in range(len(filenames))],
        'bmasks': [np.load(path / f'bmask_{i}.npy') for i in range(len(filenames))],
        'cmasks': [np.load(path / f'cmask_{i}.npy') for i in range(len(filenames))],
        'norms_masked': [np.load(path / f'norms_mask_{i}.npy') for i in range(len(filenames))],
        'poses':  np.load(path / 'poses.npy')
    }
    try:
        filenames = list(path.glob('sdf_*.png'))
        filenames = natsorted(filenames, key=lambda x: int(x.stem.split('_')[-1]))
        items['sdf'] = list(map(Image.open, filenames))
    except FileNotFoundError:
        pass
    return items


@timer
def save_items(items: dict, path: Path) -> None:
    """
    """
    print('Saving items to cache...')

    for i, image in enumerate(items['matte']):
        image.save(path / f'matte_{i}.png')
    if 'sdf' in items:
        for i, image in enumerate(items['sdf']):
            image.save(path / f'sdf_{i}.png')
    
    for i, (faces, bmask, cmask, norms, norms_masked) in enumerate(
        zip(items['faces'], items['bmasks'], items['cmasks'], items['norms'], items['norms_masked'])
    ):
        np.save(path / f'faces_{i}.npy', faces)
        np.save(path / f'bmask_{i}.npy', bmask)
        np.save(path / f'cmask_{i}.npy', cmask)
        np.save(path / f'norms_{i}.npy', norms)
        np.save(path / f'norms_mask_{i}.npy', norms_masked)
    np.save(path / 'poses.npy', items['poses'])


@timer
def visualize_items(items: dict, path: Path) -> None:
    """
    """
    os.makedirs(path, exist_ok=True)

    for i, image in tqdm(enumerate(items['matte']), 'Visualizing images'):
        image.save(f'{path}/matte_{i}.png')
    if 'sdf' in items:
        for i, image in tqdm(enumerate(items['sdf']), 'Visualizing SDF'):
            image.save(f'{path}/sdf_{i}.png')
    
    for i, faces in tqdm(enumerate(items['faces']), 'Visualizing FaceIDs'):
        colormap_faces(faces).save(f'{path}/faces_{i}.png')
    for i, cmask in tqdm(enumerate(items['cmasks']), 'Visualizing SAM Masks'):
        colormap_mask (cmask).save(f'{path}/masks_{i}.png')
    for i, norms in tqdm(enumerate(items['norms']), 'Visualizing Normals'):
        colormap_norms(norms).save(f'{path}/norms_{i}.png')
    for i, norms_masked in tqdm(enumerate(items['norms_masked']), 'Visualizing Normals Mask'):
        colormap_norms(norms_masked).save(f'{path}/norms_mask_{i}.png')


"""
Multiprocessing functions SamModelMesh
"""
def compute_face2label(
    labels: NumpyTensor['l'],
    faceid: NumpyTensor['h w'], 
    mask  : NumpyTensor['h w'],
    norms : NumpyTensor['h w 3'],
    pose  : NumpyTensor['4 4'], 
    label_sequence_count: int, threshold_counts: int=16
):
    """
    """
    #print(f'Computing face2label starting with {label_sequence_count}')

    normal_mask = norms_mask(norms, pose)

    face2label = defaultdict(Counter)
    for j, label in enumerate(labels):
        label_sequence = label_sequence_count + j
        faces_mask = (mask == label) & normal_mask
        faces, counts = np.unique(faceid[faces_mask], return_counts=True)
        faces = faces[counts > threshold_counts]
        faces = faces[faces != -1] # remove background
        for face in faces:
            face2label[int(face)][label_sequence] += np.sum(faces_mask & (faceid == face))
    return face2label


def compute_face2label_full_vectorized(
    labels: NumpyTensor['l'],
    faceid: NumpyTensor['h w'], 
    mask  : NumpyTensor['h w'],
    norms : NumpyTensor['h w 3'],
    pose  : NumpyTensor['4 4'], 
    label_sequence_count: int, threshold_counts: int=16
):
    # normal_mask = norms_mask(norms, pose)
    face2label = defaultdict(Counter)
    
    for j, label in enumerate(labels):
        label_sequence = label_sequence_count + j
        # faces_mask = (mask == label) & normal_mask
        faces_mask = (mask == label)
        
        valid_faces = faceid[faces_mask]
        valid_faces = valid_faces[valid_faces != -1]
        
        if len(valid_faces) > 0:
            face_counts = np.bincount(valid_faces)
            valid_faces = np.where(face_counts > threshold_counts)[0]
            
            for face in valid_faces:
                if face != -1 and face_counts[face] > threshold_counts:
                    face2label[face][label_sequence] += face_counts[face]
    
    return face2label


def compute_component2label_full_vectorized(
    labels: NumpyTensor['l'],
    faceid: NumpyTensor['h w'], 
    mask  : NumpyTensor['h w'],
    norms : NumpyTensor['h w 3'],
    pose  : NumpyTensor['4 4'], 
    label_sequence_count: int, face2component: dict, threshold_counts: int=16
):
    # normal_mask = norms_mask(norms, pose)
    component2label = defaultdict(Counter)
    
    # 一次性处理所有labels
    for j, label in enumerate(labels):
        label_sequence = label_sequence_count + j
        # faces_mask = (mask == label) & normal_mask
        faces_mask = (mask == label)
        
        valid_faces = faceid[faces_mask]
        valid_faces = valid_faces[valid_faces != -1]
        
        if len(valid_faces) > 0:
            face_counts = np.bincount(valid_faces)
            valid_faces = np.where(face_counts > threshold_counts)[0]
            
            for face in valid_faces:
                if face != -1 and face_counts[face] > threshold_counts:
                    component_id = face2component.get(face)
                    if component_id is not None:
                        component2label[component_id][label_sequence] += face_counts[face]
    
    for key, value in component2label.items():
        component2label[key] = value.most_common(1)[0]

    return component2label


def compute_connections(i: int, j: int, face2label1: dict, face2label2: dict, counter_threshold=32):
    """
    """
    #print(f'Computing partial connection ratios for {i} and {j}')

    connections = defaultdict(Counter)
    face2label1_common = {face: counter.most_common(1)[0][0] for face, counter in face2label1.items()}
    face2label2_common = {face: counter.most_common(1)[0][0] for face, counter in face2label2.items()}
    for face1, label1 in face2label1_common.items():
        for face2, label2 in face2label2_common.items():
            if face1 != face2:
                continue
            connections[label1][label2] += 1
            connections[label2][label1] += 1
    # remove connections where # overlapping faces is below threshold
    for label1, counter in connections.items():
        connections[label1] = {k: v for k, v in counter.items() if v > counter_threshold}
    return connections


def compute_connections_optimized(i: int, j: int, face2label1: dict, face2label2: dict, counter_threshold=32):
    connections = defaultdict(Counter)
    common_faces = set(face2label1.keys()) & set(face2label2.keys())
    
    for face in common_faces:
        label1 = face2label1[face].most_common(1)[0][0]
        label2 = face2label2[face].most_common(1)[0][0]
        connections[label1][label2] += 1
        connections[label2][label1] += 1
    
    return {label: {k: v for k, v in counter.items() if v > counter_threshold} 
            for label, counter in connections.items()}


class SamModelMesh(nn.Module):
    """
    """
    def __init__(self, config: OmegaConf, device='cuda', use_sam=True):
        """
        """
        super().__init__()
        self.config = config
        self.sam = Sam2Model(config.sam, device=device)
        self.renderer = Renderer(config.renderer)

    def load(self, scene: Scene, mesh_graph=True):
        """
        """
        self.renderer.set_object(scene)
        self.renderer.set_camera(self.config.renderer.get('camera_params', None))

        if mesh_graph:
            self.mesh_edges = trimesh.graph.face_adjacency(mesh=self.renderer.tmesh)
            self.mesh_graph = defaultdict(set)
            for face1, face2 in self.mesh_edges:
                self.mesh_graph[face1].add(face2)
                self.mesh_graph[face2].add(face1)

    @timer
    def render(self, scene: Scene, cache_path=None, visualize_path=None) -> dict[str, NumpyTensor]:
        """
        """
        if cache_path is not None and cache_path.exists():
            if self.config.cache_overwrite:
                shutil.rmtree(cache_path)
            else:
                return load_items(cache_path)
        @timer
        def render_func(uv_map=False):
            renderer_args = self.config.renderer.renderer_args.copy()
            if uv_map:
                renderer_args['uv_map'] = True # handle cases like sdf
            return render_multiview(
                self.renderer,
                camera_generation_method=self.config.renderer.camera_generation_method,
                renderer_args=renderer_args,
                sampling_args=self.config.renderer.sampling_args,
                lighting_args=self.config.renderer.lighting_args,
            )

        def compute_norms_masked(norms: NumpyTensor['h w 3'], pose: NumpyTensor['4 4']):
            """
            """
            valid = norms_mask(norms, pose)
            norms_masked = norms.copy()
            norms_masked[~valid] = np.array([1, 1, 1])
            return norms_masked
        
        renders = render_func()
        renders['norms_masked'] = [
            compute_norms_masked(norms, pose) for norms, pose in zip(renders['norms'], renders['poses'])
        ]

        def call_sam(image: Image, mask: NumpyTensor['h w']):
            """
            """
            self.sam.engine.point_grids = \
                [point_grid_from_mask(mask, self.config.sam.sam.engine_config.points_per_side ** 2)]
            return self.sam(image)

        bmasks_list = []

        if 'norms' in self.config.sam_mesh.use_modes:
            images1 = [colormap_norms(norms) for norms in renders['norms']]
            bmasks1 = [
                call_sam(image, faces != -1) for image, faces in \
                    tqdm(zip(images1, renders['faces']), 'Computing SAM Masks for norms')
            ]
            bmasks_list.extend(bmasks1)

        if 'sdf' in self.config.sam_mesh.use_modes:
            #scene_sdf = remove_texture(scene)
            tmesh_sdf = prep_mesh_shape_diameter_function(scene)
            tmesh_sdf = colormap_shape_diameter_function(tmesh_sdf, sdf_values=shape_diameter_function(tmesh_sdf))
            self.load(tmesh_sdf)
            renders_sdf = render_func(uv_map=True)
            images2 = renders_sdf['matte']

            bmasks2 = [
                call_sam(image, faces != -1) for image, faces in \
                    tqdm(zip(images2, renders['faces']), 'Computing SAM Masks for sdf')
            ]
            bmasks_list.extend(bmasks2)
            renders['sdf'] = renders_sdf['matte']

        if 'matte' in self.config.sam_mesh.use_modes: # default matte render
            images3 = renders['matte']
            # breakpoint()
            # self.sam.engine.generate_batch([np.array(image) for image in images3])
            bmasks3 = [
                call_sam(image, faces != -1) for image, faces in \
                    tqdm(zip(images3, renders['faces']), 'Computing SAM Masks for matte')
            ]
            bmasks_list.extend(bmasks3)

        self.load(scene) # restore original scene
        
        n = len(renders['faces'])
        m = len(bmasks_list) // n
        bmasks = [
            np.concatenate([bmasks_list[j * n + i] for j in range(m)], axis=0) 
            for i in range(n)
        ]
        cmasks = [combine_bmasks(masks, sort=True) for masks in bmasks]
        # sometimes SAM doesn't separate body from background, so we have extra step to remove background using faceids
        for cmask, faces in zip(cmasks, renders['faces']):
            cmask += 1
            cmask[faces == -1] = 0
        min_area = self.config.sam_mesh.get('min_area', 1024)
        cmasks = [remove_artifacts(mask, mode='islands', min_area=min_area) for mask in cmasks]
        cmasks = [remove_artifacts(mask, mode='holes'  , min_area=min_area) for mask in cmasks]
        renders['bmasks'] = bmasks
        renders['cmasks'] = cmasks

        if cache_path is not None:
            cache_path.mkdir(parents=True)
            save_items(renders, cache_path)
        if visualize_path is not None:
            visualize_items(renders, visualize_path)
        return renders

    @timer
    def lift(self, renders: dict[str, NumpyTensor]) -> dict:
        """
        """
        be, en = 0, len(renders['faces'])
        renders = {k: [v[i] for i in range(be, en) if len(v)] for k, v in renders.items()}

        print('Computing face2label for each view on ', mp.cpu_count(), ' cores')
        label_sequence_count = 1 # background is 0
        args = []
        for faceid, cmask, norms, pose in zip(
            renders['faces'],
            renders['cmasks'],
            renders['norms'],
            renders['poses'],
        ):
            labels = np.unique(cmask)
            labels = labels[labels != 0] # remove background
            args.append((labels, faceid, cmask, norms, pose, label_sequence_count, self.config.sam_mesh.get('face2label_threshold', 16)))
            label_sequence_count += len(labels)
        
        with mp.Pool(mp.cpu_count()) as pool:
            face2label_views = pool.starmap(compute_face2label_full_vectorized, args)
        
        print('Building match graph on ', mp.cpu_count(), ' cores')
        args = []
        for i, face2label1 in enumerate(face2label_views):
            for j, face2label2 in enumerate(face2label_views):
                if i < j:
                    args.append((i, j, face2label1, face2label2, self.config.sam_mesh.get('connections_threshold', 32)))
        
        with mp.Pool(mp.cpu_count()) as pool:
            partial_connections = pool.starmap(compute_connections_optimized, args)

        connections_ratios = defaultdict(Counter)
        for c in partial_connections:
            for label1, counter in c.items():
                connections_ratios[label1].update(counter)
    
        # normalize ratios
        if len(connections_ratios) > 0:
            for label1, counter in connections_ratios.items():
                total = sum(counter.values())
                connections_ratios[label1] = {k: v / total for k, v in counter.items()}
            
            counter_lens = [len(counter) for counter in connections_ratios.values()]
            counter_lens = sorted(counter_lens)
            counter_lens_threshold = max(np.percentile(counter_lens, 95), self.config.sam_mesh.get('counter_lens_threshold_min', 16))
            print('Counter lens threshold: ', counter_lens_threshold)
            removed = []
            for label, counter in connections_ratios.items():
                if len(counter) > counter_lens_threshold:
                    removed.append(label)
            for label in removed:
                connections_ratios.pop(label)
                for counter in connections_ratios.values():
                    if label in counter:
                        counter.pop(label)
        '''
        print('Count ratios:')
        for label1, counter in connections_ratios.items():
            print(label1)
            for label2, count in counter.items():
                print(label2, count)
        exit()
        '''

        bins_resolution = self.config.sam_mesh.connections_bin_resolution
        bins = np.zeros(bins_resolution + 1)
        for label1, counter in connections_ratios.items():
            #print(label1)
            for label2, ratio in counter.items():
                #print(label2, ratio)
                bins[int(ratio * bins_resolution)] += 1
        cutoff = self.config.sam_mesh.connections_bin_threshold_percentage * np.sum(bins) # more connections means higher threshold
        accum = 0
        accum_bin = 0
        while accum < cutoff:
            accum += bins[accum_bin]
            accum_bin += 1

        '''
        import matplotlib.pyplot as plt
        plt.clf()
        plt.bar(range(bins_resolution + 1), bins)
        plt.xlabel(f'Cutoff bin: {accum_bin}')
        plt.axvline(x=accum_bin, color='r')
        plt.savefig(f'ratios_{self.config.cache.stem}.png')
        '''

        # construct match graph edges
        connections = []
        connections_ratio_threshold = max(accum_bin / bins_resolution, 0.075)
        print('Connections ratio threshold: ', connections_ratio_threshold)
        for label1, counter in connections_ratios.items():
            for label2, ratio12 in counter.items():
                ratio21 = connections_ratios[label2][label1]
                # best buddy match above threshold
                if ratio12 > connections_ratio_threshold and \
                   ratio21 > connections_ratio_threshold:
                    connections.append((label1, label2))
        print('Found ', len(connections), ' connections')

        connection_graph = igraph.Graph(edges=connections, directed=False)
        connection_graph.simplify()
        communities = connection_graph.community_leiden(resolution_parameter=0)
        # for comm in communities:
        #     print(comm)
        # exit()
        label2label_consistent = {}
        comm_count = 0
        for comm in communities:
            if len(comm) > 1:
                label2label_consistent.update({label: comm[0] for label in comm if label != comm[0]})
                comm_count += 1
        print('Found ', comm_count, ' communities')

        print('Merging labels')
        face2label_combined = defaultdict(Counter)
        for face2label in face2label_views:
            face2label_combined.update(face2label)
        face2label_consistent = {}
        for face, labels in face2label_combined.items():
            hook = labels.most_common(1)[0][0]
            if hook in label2label_consistent:
                hook = label2label_consistent[hook]
            face2label_consistent[face] = hook
        #print(sorted(face2label_consistent.values()))
        return face2label_consistent

    @timer
    def lift_components(self, renders: dict[str, NumpyTensor], face2component: dict) -> dict:
        """
        """
        be, en = 0, len(renders['faces'])
        renders = {k: [v[i] for i in range(be, en) if len(v)] for k, v in renders.items()}

        print('Computing face2label for each view on ', mp.cpu_count(), ' cores')
        label_sequence_count = 1 # background is 0
        args = []
        for faceid, cmask, norms, pose in zip(
            renders['faces'],
            renders['cmasks'],
            renders['norms'],
            renders['poses'],
        ):
            labels = np.unique(cmask)
            labels = labels[labels != 0] # remove background
            args.append((labels, faceid, cmask, norms, pose, label_sequence_count, face2component, self.config.sam_mesh.get('face2label_threshold', 16)))
            label_sequence_count += len(labels)
        
        with mp.Pool(mp.cpu_count()) as pool:
            component2label_views = pool.starmap(compute_component2label_full_vectorized, args)
        
        component2label_merged = defaultdict(Counter)
        for component2label in component2label_views:
            for component, (label, count) in component2label.items():
                component2label_merged[component][label] = count
        
        label2label = {}
        for component, counter in component2label_merged.items():
            if len(counter) > 1:
                label = counter.most_common(1)[0][0]
                label2label.update({l: label for l, c in counter.items()})
        
        face2label = {}
        for face, component in face2component.items():
            if component in component2label_merged:
                label = component2label_merged[component].most_common(1)[0][0]
                face2label[face] = label2label.get(label, label)
            else:
                face2label[face] = 0
        return face2label
    
    @timer
    def smooth(self, face2label_consistent: dict) -> dict:
        """
        """
        # remove holes
        components = self.label_components(face2label_consistent)

        threshold_percentage_size = self.config.sam_mesh.smoothing_threshold_percentage_size
        threshold_percentage_area = self.config.sam_mesh.smoothing_threshold_percentage_area
        components = sorted(components, key=lambda x: len(x), reverse=True)
        components_area = [
            sum([float(self.renderer.tmesh.area_faces[face]) for face in comp]) for comp in components
        ]
        max_size = max([len(comp) for comp in components])
        max_area = max(components_area)

        remove_comp_size = set()
        remove_comp_area = set()
        for i, comp in enumerate(components):
            if len(comp)          < max_size * threshold_percentage_size:
                remove_comp_size.add(i)
            if components_area[i] < max_area * threshold_percentage_area:
                remove_comp_area.add(i)
        remove_comp = remove_comp_size.intersection(remove_comp_area)
        print('Removing ', len(remove_comp), ' components')
        for i in remove_comp:
            for face in components[i]:
                face2label_consistent.pop(face)
        
        # fill islands
        print('Smoothing labels')
        smooth_iterations = self.config.sam_mesh.smoothing_iterations
        for iteration in range(smooth_iterations):
            count = 0
            changes = {}
            for face in range(len(self.renderer.tmesh.faces)):
                if face in face2label_consistent:
                    continue
                labels_adj = Counter()
                for adj in self.mesh_graph[face]:
                    if adj in face2label_consistent:
                        label = face2label_consistent[adj]
                        if label != 0:
                            labels_adj[label] += 1
                if len(labels_adj):
                    count += 1
                    changes[face] = labels_adj.most_common(1)[0][0]
            for face, label in changes.items():
                face2label_consistent[face] = label
            print('Smoothing iteration ', iteration, ' changed ', count, ' faces')

            if count == 0:
                break

        return face2label_consistent

    @timer
    def split(self, face2label_consistent: dict) -> dict:
        """
        """
        components = self.label_components(face2label_consistent)

        labels_seen = set()
        labels_curr = max(face2label_consistent.values()) + 1
        labels_orig = labels_curr
        for comp in components:
            face = comp.pop()
            label = face2label_consistent[face]
            comp.add(face)
            if label == 0 or label in labels_seen: # background or repeated label
                face2label_consistent.update({face: labels_curr for face in comp})
                labels_curr += 1
            labels_seen.add(label)
        print('Split', (labels_curr - labels_orig), 'times') # account for background

        return face2label_consistent

    @timer
    def split_components(self) -> dict:
        components = []
        visited = set()

        def dfs(source: int):
            stack = [source]
            components.append({source})
            visited.add(source)
            
            while stack:
                node = stack.pop()
                for adj in self.mesh_graph[node]:
                    if adj not in visited:
                        stack.append(adj)
                        components[-1].add(adj)
                        visited.add(adj)

        for face in range(len(self.renderer.tmesh.faces)):
            if face not in visited:
                dfs(face)
        
        face2label_consistent = {}
        labels_curr = 1
        for comp in components:            
            face2label_consistent.update({face: labels_curr for face in comp})
            labels_curr += 1
        print('Generate', labels_curr, 'components') # account for background
        return face2label_consistent

    @timer
    def smooth_repartition_faces(self, face2label_consistent: dict, target_labels=None) -> dict:
        """
        """
        tmesh = self.renderer.tmesh

        partition = np.array([face2label_consistent[face] for face in range(len(tmesh.faces))])
        # # before
        # cost_data = np.zeros((len(tmesh.faces), np.max(partition) + 1))
        # for f in range(len(tmesh.faces)):
        #     for l in np.unique(partition):
        #         cost_data[f, l] = 0 if partition[f] == l else 1
        # cost_smoothness = -np.log(tmesh.face_adjacency_angles / np.pi + 1e-20)
        # after
        cost_data = np.zeros((len(tmesh.faces), np.max(partition) + 1))
        labels_range = np.arange(np.max(partition) + 1)
        partition_expanded = partition[:, np.newaxis]  # 将partition转换为列向量以便广播
        cost_data = (partition_expanded != labels_range).astype(float)
        cost_smoothness = -np.log(tmesh.face_adjacency_angles / np.pi + 1e-20)
        
        lambda_seed = self.config.sam_mesh.repartition_lambda
        if target_labels is None:
            # # before
            # refined_partition = repartition(tmesh, partition, cost_data, cost_smoothness, self.config.sam_mesh.repartition_iterations, lambda_seed)
            # # after
            refined_partition = repartition_with_pymaxflow(tmesh, partition, cost_data, cost_smoothness, self.config.sam_mesh.repartition_iterations, lambda_seed)
            return {
                face: refined_partition[face] for face in range(len(tmesh.faces))
            }
    
        lambda_range=(
            self.config.sam_mesh.repartition_lambda_lb, 
            self.config.sam_mesh.repartition_lambda_ub
        )
        lambdas = np.linspace(*lambda_range, num=mp.cpu_count())
        chunks = [
            (tmesh, partition, cost_data, cost_smoothness, self.config.sam_mesh.repartition_iterations, _lambda) 
            for _lambda in lambdas
        ]
        with mp.Pool(mp.cpu_count() // 2) as pool:
            refined_partitions = pool.starmap(repartition, chunks)

        def compute_cur_labels(part, noise_threshold=10):
            """
            """
            values, counts = np.unique(part, return_counts=True)
            return values[counts > noise_threshold]

        # lambda crawling algorithm when target_labels is specified i.e. Princeton Mesh Segmentation Benchmark
        max_iteration = 8
        cur_iteration = 0
        cur_lambda_index = np.searchsorted(lambdas, lambda_seed)
        cur_labels = len(compute_cur_labels(refined_partitions[cur_lambda_index]))
        while not (
            target_labels - self.config.sam_mesh.repartition_lambda_tolerance <= cur_labels and
            target_labels + self.config.sam_mesh.repartition_lambda_tolerance >= cur_labels
        ) and cur_iteration < max_iteration:
            
            if cur_labels < target_labels and cur_lambda_index > 0:
                # want more labels so decrease lambda
                cur_lambda_index -= 1
            if cur_labels > target_labels and cur_lambda_index < len(refined_partitions) - 1:
                # want less labels so increase lambda
                cur_lambda_index += 1

            cur_labels = len(compute_cur_labels(refined_partitions[cur_lambda_index]))
            cur_iteration += 1

        print('Repartitioned with ', cur_labels, ' labels aiming for ', target_labels, 'target labels using lambda ', lambdas[cur_lambda_index], ' in ', cur_iteration, ' iterations')
        
        refined_partition = refined_partitions[cur_lambda_index]
        return {
            face: refined_partition[face] for face in range(len(tmesh.faces))
        }

    @timer
    def forward(self, scene: Scene, cache_path=None, visualize_path=None, target_labels=None) -> tuple[dict, Trimesh]:
        """
        """
        self.load(scene)
        renders = self.render(scene, cache_path=cache_path, visualize_path=visualize_path)
        face2label_consistent = self.lift(renders)
        face2label_consistent = self.smooth(face2label_consistent)
        # inject unlabeled faces after smoothing
        for face in range(len(self.renderer.tmesh.faces)):
            if face not in face2label_consistent:
                face2label_consistent[face] = 0
        face2label_consistent = self.smooth_repartition_faces(face2label_consistent, target_labels=target_labels)
        face2label_consistent = self.split (face2label_consistent) # needed to label all faces for repartition
        face2label_consistent = {int(k): int(v) for k, v in face2label_consistent.items()} # ensure serialization
        assert self.renderer.tmesh.faces.shape[0] == len(face2label_consistent)
        return face2label_consistent, self.renderer.tmesh

    @timer
    def dfs_component_by_color(self) -> dict:
        face_color = fetch_face_color(self.renderer.tmesh).astype(np.int32)

        components = []
        visited    = set()
        def dfs(source: int):
            stack = [source]
            components.append({source})
            visited.add(source)
            source_color = face_color[source]

            while stack:
                node = stack.pop()
                for adj in self.mesh_graph[node]:
                    if (adj not in visited and 
                        np.abs(source_color - face_color[adj]).sum() < self.config.sam_mesh.color_res
                        ):
                        
                        stack.append(adj)
                        components[-1].add(adj)
                        visited.add(adj)

        for face in range(len(self.renderer.tmesh.faces)):
            if face not in visited:
                dfs(face)

        face2component = {}
        for label, component in enumerate(components):
            face2component.update({face: label for face in component})
        return face2component

    def find_and_correct_discrete_faces(self, face2label):
        """
        查找并校正离散面。
        Args:
            face2label: 字典，面ID -> 标签
            face_neighbors: 字典，面ID -> 邻居面ID列表
        Returns:
            校正后的 face2label 字典
        """
        face2label_updated = face2label.copy()
        discrete_faces = []

        for face_id, label in face2label.items():
            neighbor_labels = []
            for neighbor_id in self.mesh_graph.get(face_id, []):
                if neighbor_id in face2label and face2label[neighbor_id] != 0:  # 确保邻居面在字典中
                    neighbor_labels.append(face2label[neighbor_id])
            
            if not neighbor_labels:  # 如果该面没有邻居，则跳过
                continue

            # 找到邻居标签中的众数（出现最频繁的标签）
            label_counter = Counter(neighbor_labels)
            most_common_label, most_common_count = label_counter.most_common(1)[0]

            # 判断是否为离散面：如果当前面的标签不在邻居标签的众数中，或者其标签与众数不同
            if label != most_common_label or label == 0:
                discrete_faces.append(face_id)
                face2label_updated[face_id] = most_common_label  # 进行校正

        print(f"发现并校正了 {len(discrete_faces)} 个离散面")
        return face2label_updated

    
    @timer
    def forward_component(self, scene: Scene, cache_path=None, visualize_path=None, target_labels=None) -> tuple[dict, Trimesh]:
        """
        """
        self.load(scene)
        renders = self.render(scene, cache_path=cache_path, visualize_path=visualize_path)
        face2component = self.dfs_component_by_color() # needed to label all faces for repartition
        face2label_consistent = face2component
        face2label_consistent = self.lift_components(renders, face2component)
        face2label_consistent = self.smooth(face2label_consistent)
        # inject unlabeled faces after smoothing
        for face in range(len(self.renderer.tmesh.faces)):
            if face not in face2label_consistent:
                face2label_consistent[face] = 0
        face2label_consistent = self.smooth_repartition_faces(face2label_consistent, target_labels=target_labels)
        face2label_consistent = self.split (face2label_consistent) # needed to label all faces for repartition

        face2label_consistent = {int(k): int(v) for k, v in face2label_consistent.items()} # ensure serialization
        assert self.renderer.tmesh.faces.shape[0] == len(face2label_consistent)
        return face2label_consistent, self.renderer.tmesh

    @timer
    def label_components(self, face2label: dict) -> list[set]:
        """
        """
        components = []
        visited = set()

        def dfs(source: int):
            stack = [source]
            components.append({source})
            visited.add(source)
            
            while stack:
                node = stack.pop()
                for adj in self.mesh_graph[node]:
                    if adj not in visited and adj in face2label and face2label[adj] == face2label[node]:
                        stack.append(adj)
                        components[-1].add(adj)
                        visited.add(adj)

        for face in range(len(self.renderer.tmesh.faces)):
            if face not in visited and face in face2label:
                dfs(face)
        return components


def fetch_face_color(mesh: Trimesh):
    color_kind = mesh.visual.kind
    if color_kind == 'face':
        face_colors = mesh.visual.face_colors
    elif color_kind == 'vertex':
        vertex_colors = mesh.visual.vertex_colors
        face_colors = vertex_colors[mesh.faces].mean(axis=1).astype(int)
    elif color_kind == 'texture' and hasattr(mesh.visual, 'uv'):
        uv = mesh.visual.uv
        uv[:, 1] = 1 - uv[:, 1]
        texture_image = mesh.visual.material.image
        tex_array = np.array(texture_image)
        
        uv_coords = (uv * [tex_array.shape[1]-1, tex_array.shape[0]-1]).astype(int)

        face_colors_list = []
        for face in mesh.faces:
            face_uvs = uv_coords[face]
            sampled_colors = tex_array[face_uvs[:, 1], face_uvs[:, 0]]

            avg_color = sampled_colors.mean(axis=0).astype(int)
            face_colors_list.append(avg_color)
        face_colors = np.array(face_colors_list)
    return face_colors

def segment_mesh(filename: Path | str, config: OmegaConf, visualize=False, extension='glb', target_labels=None, texture=False) -> Trimesh:
    """
    """
    print('Segmenting mesh with SAMesh: ', filename)
    filename = Path(filename)
    config = copy.deepcopy(config)
    config.cache  = Path(config.cache)  / filename.stem if "cache" in config else None
    config.output = Path(config.output) / filename.stem

    model = SamModelMesh(config)
    tmesh = read_mesh(filename, norm=True)
    if not texture:
        tmesh = remove_texture(tmesh, visual_kind='vertex')
    
    # run sam grounded mesh and optionally visualize renders
    visualize_path = f'{config.output}/{filename.stem}_visualized' if visualize else None
    faces2label, _ = model(tmesh, visualize_path=visualize_path, target_labels=target_labels)

    # colormap and save mesh
    os.makedirs(config.output, exist_ok=True)
    tmesh_colored = colormap_faces_mesh(tmesh, faces2label)
    tmesh_colored.export       (f'{config.output}/{filename.stem}_segmented.{extension}')
    json.dump(faces2label, open(f'{config.output}/{filename.stem}_face2label.json', 'w'))
    return tmesh_colored


if __name__ == '__main__':
    import glob
    from natsort import natsorted

    def read_filenames(pattern: str):
        """
        """
        filenames = glob.glob(pattern)
        filenames = map(Path, filenames)
        filenames = natsorted(list(set(filenames)))
        print('Segmenting ', len(filenames), ' meshes')
        return filenames

    filenames = read_filenames('/home/gtangg12/data/samesh/backflip-benchmark-remeshed-processed/*.glb')
    config = OmegaConf.load('/home/gtangg12/samesh/configs/mesh_segmentation.yaml')
    for i, filename in enumerate(filenames):
        segment_mesh(filename, config, visualize=False)
    
    config_original = OmegaConf.load('/home/gtangg12/samesh/configs/mesh_segmentation_coseg.yaml')
    categories = ['candelabra', 'chairs', 'fourleg', 'goblets', 'guitars', 'irons', 'lamps', 'vases']
    for cat in categories:
        filenames = read_filenames(f'/home/gtangg12/data/samesh/coseg/{cat}/*.off')
        for filename in filenames:
            config = copy.deepcopy(config_original)
            config.output = Path(config.output) / cat
            if "cache" in config:
                config.cache  = Path(config.cache) / cat
            segment_mesh(filename, config, visualize=False)

    config = OmegaConf.load('/home/gtangg12/samesh/configs/mesh_segmentation_princeton.yaml')
    filenames = read_filenames('/home/gtangg12/data/samesh/MeshsegBenchmark-1.0/data/off/*.off')
    for i, filename in enumerate(filenames):
        name, extension = filename.stem, filename.suffix[1:]
        category = (int(name) - 1) // 20 + 1
        if category in [14]: #[4, 8, 13, 14, 17]:
            continue
        segment_mesh(filename, config, visualize=False)

    with open('/home/gtangg12/data/samesh/MeshsegBenchmark-1.0/util/parameters/nSeg-ByModel.txt') as f:
        target_labels_dict = {str(i): int(line) for i, line in enumerate(f.readlines(), 1)}
    
    config = OmegaConf.load('/home/gtangg12/samesh/configs/mesh_segmentation_princeton_dynamic.yaml')
    filenames = read_filenames('/home/gtangg12/data/samesh/MeshsegBenchmark-1.0/data/off/*.off')
    for i, filename in enumerate(filenames):
        name, extension = filename.stem, filename.suffix[1:]
        category = (int(name) - 1) // 20 + 1
        if category in [14]: #[4, 8, 13, 14, 17]:
            continue
        segment_mesh(filename, config, visualize=False, target_labels=target_labels_dict[name])