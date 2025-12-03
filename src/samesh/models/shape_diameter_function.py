import glob
import json
import os
import copy
from pathlib import Path
from collections import defaultdict

import numpy as np
import pymeshlab
import trimesh
import networkx as nx
import igraph
import maxflow
from numpy.random import RandomState
from trimesh.base import Trimesh, Scene
from sklearn.mixture import GaussianMixture
from tqdm import tqdm
from igraph import Graph
from omegaconf import OmegaConf

from samesh.data.common import NumpyTensor
from samesh.data.loaders import scene2mesh, read_mesh
from samesh.utils.mesh import duplicate_verts


EPSILON = 1e-20
SCALE = 1e6


def partition_cost(
    mesh           : Trimesh,
    partition      : NumpyTensor['f'],
    cost_data      : NumpyTensor['f num_components'],
    cost_smoothness: NumpyTensor['e']
) -> float:
    """
    """
    cost = 0
    for f in range(len(partition)):
        cost += cost_data[f, partition[f]]
    for i, edge in enumerate(mesh.face_adjacency):
        f1, f2 = int(edge[0]), int(edge[1])
        if partition[f1] != partition[f2]:
            cost += cost_smoothness[i]
    return cost

def partition_cost_optimized(
    mesh: Trimesh,
    partition: NumpyTensor['f'],
    cost_data: NumpyTensor['f num_components'],
    cost_smoothness: NumpyTensor['e']
) -> float:
    # 向量化计算数据项成本
    data_cost = cost_data[np.arange(len(partition)), partition].sum()
    
    # 向量化计算平滑项成本
    edges = mesh.face_adjacency
    f1 = edges[:, 0].astype(int)
    f2 = edges[:, 1].astype(int)
    
    # 只对标签不同的边添加平滑成本
    mask = partition[f1] != partition[f2]
    smooth_cost = cost_smoothness[mask].sum()
    
    return data_cost + smooth_cost


def construct_expansion_graph(
    label          : int,
    mesh           : Trimesh,
    partition      : NumpyTensor['f'],
    cost_data      : NumpyTensor['f num_components'],
    cost_smoothness: NumpyTensor['e']
) -> nx.Graph:
    """
    """
    G = nx.Graph() # undirected graph
    A = 'alpha'
    B = 'alpha_complement'

    node2index = {}
    G.add_node(A)
    G.add_node(B)
    node2index[A] = 0
    node2index[B] = 1
    for i in range(len(mesh.faces)):
        G.add_node(i)
        node2index[i] = 2 + i

    aux_count = 0
    for i, edge in enumerate(mesh.face_adjacency): # auxillary nodes
        f1, f2 = int(edge[0]), int(edge[1])
        if partition[f1] != partition[f2]:
            a = (f1, f2)
            if a in node2index: # duplicate edge
                continue
            G.add_node(a)
            node2index[a] = len(mesh.faces) + 2 + aux_count
            aux_count += 1

    for f in range(len(mesh.faces)):
        G.add_edge(A, f, capacity=cost_data[f, label])
        G.add_edge(B, f, capacity=float('inf') if partition[f] == label else cost_data[f, partition[f]])

    for i, edge in enumerate(mesh.face_adjacency):
        f1, f2 = int(edge[0]), int(edge[1])
        a = (f1, f2)
        if partition[f1] == partition[f2]:
            if partition[f1] != label:
                G.add_edge(f1, f2, capacity=cost_smoothness[i])
        else:
            G.add_edge(a, B, capacity=cost_smoothness[i])
            if partition[f1] != label:
                G.add_edge(f1, a, capacity=cost_smoothness[i])
            if partition[f2] != label:
                G.add_edge(a, f2, capacity=cost_smoothness[i])
    
    return G, node2index


def construct_expansion_graph_igraph(label, mesh, partition, cost_data, cost_smoothness):

    n_faces = len(mesh.faces)
    nodes = []          # int faces + A + B + aux nodes
    A, B = "alpha", "alpha_complement"

    node2index = {A:0, B:1}
    nodes.extend([A, B])

    # faces
    for f in range(n_faces):
        node2index[f] = len(nodes)
        nodes.append(f)

    # aux nodes
    for i, (f1, f2) in enumerate(mesh.face_adjacency):
        if partition[f1] != partition[f2]:
            a = f"aux_{f1}_{f2}"
            node2index[(f1, f2)] = len(nodes)
            nodes.append((f1,f2))

    g = Graph(directed=False)
    g.add_vertices(len(nodes))
    g.vs["name"] = nodes

    # add edges (precompute then bulk add)
    edges = []
    capacities = []

    # A/B -> faces
    for f in range(n_faces):
        edges.append((node2index[A], node2index[f]))
        capacities.append(cost_data[f, label])

        capB = float("inf") if partition[f]==label else cost_data[f, partition[f]]
        edges.append((node2index[B], node2index[f]))
        capacities.append(capB)

    # smoothness
    for i, (f1, f2) in enumerate(mesh.face_adjacency):
        cap = cost_smoothness[i]
        if partition[f1] == partition[f2]:
            if partition[f1] != label:
                edges.append((node2index[f1], node2index[f2]))
                capacities.append(cap)
        else:
            a = (f1,f2)
            ia = node2index[a]
            edges.append((ia, node2index[B])); capacities.append(cap)
            if partition[f1] != label:
                edges.append((node2index[f1], ia)); capacities.append(cap)
            if partition[f2] != label:
                edges.append((node2index[f2], ia)); capacities.append(cap)

    g.add_edges(edges)
    g.es["capacity"] = capacities

    return g, node2index


class MeshTopology:
    def __init__(self, mesh):
        self.mesh = mesh
        self.n_faces = len(mesh.faces)

        # f1,f2 arrays
        adj = mesh.face_adjacency
        self.f1 = adj[:, 0].astype(int)
        self.f2 = adj[:, 1].astype(int)

        self.n_edges = len(adj)


class ExpansionGraphCache:
    def __init__(self, topo: MeshTopology):
        self.topo = topo
        self.A = "alpha"
        self.B = "alpha_complement"

        # 基础节点（固定）
        self.base_nodes = [self.A, self.B] + list(range(topo.n_faces))

    def build_aux_indices(self, partition):
        """
        根据 partition 创建 aux 节点索引（轻量操作）
        """
        aux_list = []
        f1, f2 = self.topo.f1, self.topo.f2
        mask = partition[f1] != partition[f2]
        # 避免 tuple → string → tuple
        aux_list = list(zip(f1[mask], f2[mask]))

        # node2index
        node2index = {self.A:0, self.B:1}
        for i in range(self.topo.n_faces):
            node2index[i] = 2 + i

        start = 2 + self.topo.n_faces
        for i,a in enumerate(aux_list):
            node2index[a] = start + i

        return aux_list, node2index


def construct_graph_fast(cache: ExpansionGraphCache, label, partition, cost_data, cost_smoothness):

    topo = cache.topo
    aux_list, node2index = cache.build_aux_indices(partition)

    n_nodes = 2 + topo.n_faces + len(aux_list)
    g = Graph(directed=False)
    g.add_vertices(n_nodes)

    edges = []
    capacities = []

    A = cache.A
    B = cache.B

    # ------------ A/B → face edges ------------
    for f in range(topo.n_faces):
        iA = node2index[A]
        iB = node2index[B]
        iF = node2index[f]

        # A → f
        edges.append((iA, iF))
        capacities.append(cost_data[f, label])

        # B → f
        capB = float("inf") if partition[f] == label else cost_data[f, partition[f]]
        edges.append((iB, iF))
        capacities.append(capB)

    # ------------ smoothness edges ------------
    f1 = topo.f1
    f2 = topo.f2

    idx = 0
    for (u, v), cap in zip(zip(f1, f2), cost_smoothness):
        same = (partition[u] == partition[v])

        if same:
            # no aux
            if partition[u] != label:
                edges.append((node2index[u], node2index[v]))
                capacities.append(cap)
        else:
            # aux node
            a = (u, v)
            ia = node2index[a]

            edges.append((ia, node2index[B])); capacities.append(cap)
            if partition[u] != label:
                edges.append((node2index[u], ia)); capacities.append(cap)
            if partition[v] != label:
                edges.append((node2index[v], ia)); capacities.append(cap)

    g.add_edges(edges)
    g.es["capacity"] = capacities

    return g, node2index


def repartition(
    mesh: trimesh.Trimesh,
    partition      : NumpyTensor['f'],
    cost_data      : NumpyTensor['f num_components'],
    cost_smoothness: NumpyTensor['e'],
    smoothing_iterations: int,
    _lambda=1.0,
):
    A = 'alpha'
    B = 'alpha_complement'
    labels = np.unique(partition)

    cost_smoothness = cost_smoothness * _lambda

    # networkx broken for float capacities
    #cost_data       = np.round(cost_data       * SCALE).astype(int)
    #cost_smoothness = np.round(cost_smoothness * SCALE).astype(int)

    topo = MeshTopology(mesh)
    cache = ExpansionGraphCache(topo)

    cost_min = partition_cost(mesh, partition, cost_data, cost_smoothness)

    for i in range(smoothing_iterations):

        #print('Repartition iteration ', i)
        
        for label in tqdm(labels):
            # G, node2index = construct_expansion_graph(label, mesh, partition, cost_data, cost_smoothness)
            # G, node2index = construct_expansion_graph_igraph(label, mesh, partition, cost_data, cost_smoothness)
            G, node2index = construct_graph_fast(cache, label, partition, cost_data, cost_smoothness)
            index2node = {v: k for k, v in node2index.items()}

            '''
            _, (S, T) = nx.minimum_cut(G, A, B)
            assert A in S and B in T
            S = np.array([v for v in S if isinstance(v, int)]).astype(int)
            T = np.array([v for v in T if isinstance(v, int)]).astype(int)
            '''

            # G = igraph.Graph.from_networkx(G)
            outputs = G.st_mincut(source=node2index[A], target=node2index[B], capacity='capacity')
            S = outputs.partition[0]
            T = outputs.partition[1]
            assert node2index[A] in S and node2index[B] in T
            S = np.array([index2node[v] for v in S if isinstance(index2node[v], int)]).astype(int)
            T = np.array([index2node[v] for v in T if isinstance(index2node[v], int)]).astype(int)

            assert (partition[S] == label).sum() == 0 # T consists of those assigned 'alpha' and S 'alpha_complement' (see paper)
            partition[T] = label

            cost = partition_cost_optimized(mesh, partition, cost_data, cost_smoothness)
            if cost > cost_min:
                raise ValueError('Cost increased. This should not happen because the graph cut is optimal.')
            cost_min = cost
    
    return partition


def construct_graph_pymaxflow(cache: ExpansionGraphCache, label, partition, cost_data, cost_smoothness):
    """
    Build a PyMaxflow graph for the expansion move of `label`.
    Returns:
      g: the pymaxflow.Graph object (after maxflow() it's ready)
      node2index: mapping (same as cache.build_aux_indices)
    """
    topo = cache.topo
    aux_list, node2index = cache.build_aux_indices(partition)

    n_nodes = 2 + topo.n_faces + len(aux_list)  # A, B, faces, aux
    # create graph
    g = maxflow.Graph[float](n_nodes, max(0, len(topo.f1) * 3))  # hint sizes (optional)
    g.add_nodes(n_nodes)

    # terminal/source node is A, sink is B (we don't need explicit A/B nodes in pymaxflow terminals;
    # we implement A-terminal capacity as source capacity, B-terminal capacity as sink capacity)
    # But to follow your indexing we still allocate nodes for A and B (unused for edges except indices).
    iA = node2index[cache.A]
    iB = node2index[cache.B]

    # --- A/B terminal edges for faces (use add_tedge: (node, source_cap, sink_cap)) ---
    for f in range(topo.n_faces):
        iF = node2index[f]
        source_cap = float(cost_data[f, label])
        sink_cap = float("inf") if partition[f] == label else float(cost_data[f, partition[f]])
        # pymaxflow supports large floats; if you prefer integers, scale and cast to int.
        g.add_tedge(iF, source_cap, sink_cap)

    # --- smoothness / pairwise edges ---
    f1 = topo.f1
    f2 = topo.f2
    # iterate edges in order so cost_smoothness aligns
    for (u, v), cap in zip(zip(f1, f2), cost_smoothness):
        u = int(u); v = int(v); cap = float(cap)
        same = (partition[u] == partition[v])
        if same:
            # no aux node used; only add if neither side is alpha (i.e., partition[u] != label)
            if partition[u] != label:
                g.add_edge(node2index[u], node2index[v], cap, cap)
        else:
            # there is an aux node a = (u,v)
            a = (u, v)
            ia = node2index[a]  # index of aux node
            # connect aux node to sink (B) -> this is a terminal (sink) capacity
            # add_tedge(node, source_cap, sink_cap)
            g.add_tedge(ia, 0.0, cap)
            # connect face -> aux if that face is not already label (i.e., can be cut)
            if partition[u] != label:
                g.add_edge(node2index[u], ia, cap, cap)
            if partition[v] != label:
                g.add_edge(node2index[v], ia, cap, cap)

    return g, node2index


# Example replacement inside repartition loop:
def repartition_with_pymaxflow(
    mesh: trimesh.Trimesh,
    partition,
    cost_data,
    cost_smoothness,
    smoothing_iterations: int,
    _lambda=1.0,
):
    topo = MeshTopology(mesh)
    cache = ExpansionGraphCache(topo)
    labels = np.unique(partition)
    cost_smoothness = cost_smoothness * _lambda

    cost_min = partition_cost_optimized(mesh, partition, cost_data, cost_smoothness)

    for it in range(smoothing_iterations):
        print("Repartition iteration", it)
        for label in tqdm(labels):
            g, node2index = construct_graph_pymaxflow(cache, label, partition, cost_data, cost_smoothness)
            # run maxflow
            flow_val = g.maxflow()

            # collect which face nodes belong to source set (i.e., will be assigned 'label')
            # NOTE: pymaxflow.Graph.get_segment(node) returns 1 if node is in the SOURCE set, 0 if in SINK set.
            # (If you see reversed behavior in your pymaxflow version, swap checks.)
            new_label_nodes = []
            # face nodes indices are 2 .. 2 + n_faces - 1
            for f in range(topo.n_faces):
                idx = node2index[f]
                seg = g.get_segment(idx)
                # seg == 1 means connected to source (A) after mincut
                if seg == 1:
                    new_label_nodes.append(f)

            # apply expansion move: assign label to all nodes in source set that are faces
            partition[new_label_nodes] = label

            # verify cost monotonicity
            cost = partition_cost_optimized(mesh, partition, cost_data, cost_smoothness)
            if cost > cost_min + 1e-12:
                # Shouldn't happen for correct construction; keep a helpful error message
                raise ValueError(f"Cost increased after graph cut for label {label} (was {cost_min}, now {cost})")
            cost_min = cost

    return partition


def prep_mesh_shape_diameter_function(source: Trimesh | Scene) -> Trimesh:
    """
    """
    if isinstance(source, trimesh.Scene):
        source = scene2mesh(source)
    source.merge_vertices(merge_tex=True, merge_norm=True)
    return source


def colormap_shape_diameter_function(mesh: Trimesh, sdf_values: NumpyTensor['f']) -> Trimesh:
    """
    """
    assert len(mesh.faces) == len(sdf_values)
    mesh = duplicate_verts(mesh) # needed to prevent face color interpolation
    mesh.visual.face_colors = trimesh.visual.interpolate(sdf_values, color_map='plasma')
    return mesh


def colormap_partition(mesh: Trimesh, partition: NumpyTensor['f']) -> Trimesh:
    """
    """
    assert len(mesh.faces) == len(partition)
    palette = RandomState(0).randint(0, 255, (np.max(partition) + 1, 3)) # must init every time to get same colors
    mesh = duplicate_verts(mesh) # needed to prevent face color interpolation
    mesh.visual.face_colors = palette[partition]
    return mesh


def shape_diameter_function(mesh: Trimesh, norm=True, alpha=4, rays=64, cone_amplitude=120) -> NumpyTensor['f']:
    """
    """
    mesh = pymeshlab.Mesh(mesh.vertices, mesh.faces)
    meshset = pymeshlab.MeshSet()
    meshset.add_mesh(mesh)
    meshset.compute_scalar_by_shape_diameter_function_per_vertex(rays=rays, cone_amplitude=cone_amplitude)

    sdf_values = meshset.current_mesh().face_scalar_array()
    sdf_values[np.isnan(sdf_values)] = 0
    if norm:
        # normalize and smooth shape diameter function values
        min = sdf_values.min()
        max = sdf_values.max()
        sdf_values = (sdf_values - min) / (max - min)
        sdf_values = np.log(sdf_values * alpha + 1) / np.log(alpha + 1)
    return sdf_values


def partition_faces(mesh: Trimesh, num_components: int, _lambda: float, smooth=True, smoothing_iterations=1, **kwargs) -> NumpyTensor['f']:
    """
    """
    sdf_values = shape_diameter_function(mesh, norm=True).reshape(-1, 1)

    # fit 1D GMM to shape diameter function values
    gmm = GaussianMixture(num_components)
    gmm.fit(sdf_values)
    probs = gmm.predict_proba(sdf_values)
    if not smooth:
        return np.argmax(probs, axis=1)

    # data and smoothness terms
    cost_data       = -np.log(probs + EPSILON)
    cost_smoothness = -np.log(mesh.face_adjacency_angles / np.pi + EPSILON)
    cost_smoothness = _lambda * cost_smoothness

    # generate initial partition and refine with alpha expansion graph cut
    partition = np.argmin(cost_data, axis=1)
    partition = repartition(mesh, partition, cost_data, cost_smoothness, smoothing_iterations=smoothing_iterations)
    return partition


def partition2label(mesh: Trimesh, partition: NumpyTensor['f']) -> NumpyTensor['f']:
    """
    """
    edges = trimesh.graph.face_adjacency(mesh=mesh)
    graph = defaultdict(set)
    for face1, face2 in edges:
        graph[face1].add(face2)
        graph[face2].add(face1)
    labels = set(list(np.unique(partition)))
    
    components = []
    visited = set()

    def dfs(source: int):
        stack = [source]
        components.append({source})
        visited.add(source)
        
        while stack:
            node = stack.pop()
            for adj in graph[node]:
                if adj not in visited and partition[adj] == partition[node]:
                    stack.append(adj)
                    components[-1].add(adj)
                    visited.add(adj)

    for face in range(len(mesh.faces)):
        if face not in visited:
            dfs(face)

    partition_output = np.zeros_like(partition)
    label_total = 0
    for component in components:
        for face in component:
            partition_output[face] = label_total
        label_total += 1
    return partition_output


def segment_mesh_sdf(filename: Path | str, config: OmegaConf, extension='glb') -> Trimesh:
    """
    """
    print('Segmenting mesh with Shape Diameter Funciont: ', filename)
    filename = Path(filename)
    config = copy.deepcopy(config)
    config.output = Path(config.output) / filename.stem

    mesh = read_mesh(filename, norm=True)
    mesh = prep_mesh_shape_diameter_function(mesh)
    partition              = partition_faces(mesh, config.num_components, config.repartition_lambda, config.repartition_iterations)
    partition_disconnected = partition2label(mesh, partition)
    faces2label = {int(i): int(partition_disconnected[i]) for i in range(len(partition_disconnected))}

    os.makedirs(config.output, exist_ok=True)
    mesh_colored = colormap_partition(mesh, partition_disconnected)
    mesh_colored.export        (f'{config.output}/{filename.stem}_segmented.{extension}')
    json.dump(faces2label, open(f'{config.output}/{filename.stem}_face2label.json', 'w'))
    return mesh_colored
    

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
    #filenames = [Path('/home/gtangg12/data/samesh.backflip-benchmark-remeshed-processed/jacket.glb')]
    config = OmegaConf.load('/home/gtangg12/samesh/configs/mesh_segmentation_shape_diameter_function.yaml')
    for i, filename in enumerate(filenames):
        segment_mesh_sdf(filename, config)

    config_original = OmegaConf.load('/home/gtangg12/samesh/configs/mesh_segmentation_shape_diameter_function_coseg.yaml')
    categories = ['candelabra', 'chairs', 'fourleg', 'goblets', 'guitars', 'irons', 'lamps', 'vases']
    for cat in categories:
        filenames = read_filenames(f'/home/gtangg12/data/samesh/coseg/{cat}/*.off')
        for filename in filenames:
            config = copy.deepcopy(config_original)
            config.output = Path(config.output) / cat
            segment_mesh_sdf(filename, config)
    
    config_original = OmegaConf.load('/home/gtangg12/samesh/configs/mesh_segmentation_shape_diameter_function_princeton.yaml')
    filenames = read_filenames('/home/gtangg12/data/samesh/MeshsegBenchmark-1.0/data/off/*.off')
    for i, filename in enumerate(filenames):
        name, extension = filename.stem, filename.suffix[1:]
        category = (int(name) - 1) // 20 + 1
        if category in [14]: #[4, 8, 13, 14, 17]:
            continue
        config = copy.deepcopy(config_original)
        segment_mesh_sdf(filename, config)