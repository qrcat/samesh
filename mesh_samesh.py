from omegaconf import OmegaConf, DictConfig
from samesh.utils.timer import timer
from samesh.data.loaders import *
from samesh.models.sam_mesh import *
from functools import wraps
from pathlib import Path
import os
import time
import copy
import argparse


@timer
def init_model(config):
    return SamModelMesh(config)


@timer
def load_model(filename: Path | str, texture=True, process=True):
    filename = Path(filename)
    tmesh = read_mesh(filename, norm=True, process=process)
    if not texture:
        tmesh = remove_texture(tmesh, visual_kind="vertex")
    return tmesh


@timer
def label2seam_face_f(model, faces2label, tmesh):
    cutting_edges = []
    for face1, face2 in model.mesh_edges:
        if faces2label[face1] != faces2label[face2]:
            cutting_edges.append([face1, face2])
    cutting_edges = np.array(cutting_edges)
    return cutting_edges


@timer
def label2seam_edge_v(model, faces2label, tmesh):
    cutting_edges_vertices = []
    for i, (face1, face2) in enumerate(model.mesh_edges):
        if faces2label[face1] != faces2label[face2]:
            edge_vertices = tmesh.face_adjacency_edges[i]  # 例如 array([v1, v2])
            cutting_edges_vertices.append(edge_vertices)
    cutting_edges_vertices = np.array(cutting_edges_vertices)
    return cutting_edges_vertices


@timer
def workflow(
    config: DictConfig,
    model: SamModelMesh,
    tmesh: trimesh.Trimesh,
    filename: Path | str,
    mode: str,
    visualize=False,
    extension="obj",
):
    config = copy.deepcopy(config)

    filename = Path(filename)

    cache_path = Path(config.cache) / filename.stem if "cache" in config else None
    output_path = Path(config.output) / filename.stem
    # run sam grounded mesh and optionally visualize renders
    visualize_path = output_path / f"{filename.stem}_visualized" if visualize else None

    if mode == "face":
        faces2label, _ = model.forward(
            tmesh,
            cache_path=cache_path,
            visualize_path=visualize_path,
            target_labels=None,
        )
    elif mode == "component":
        faces2label, _ = model.forward_component(
            tmesh,
            cache_path=cache_path,
            visualize_path=visualize_path,
            target_labels=None,
        )
    else:
        raise ValueError(f"Invalid mode {mode}.")

    # colormap and save mesh
    if visualize:
        output_path.mkdir(parents=True, exist_ok=True)
        tmesh = read_mesh(filename, norm=False, process=False)
        tmesh_colored = colormap_faces_mesh(tmesh, faces2label)
        tmesh_segment = split_mesh_by_label_optimized(tmesh, faces2label)
        tmesh_colored.export(output_path / f"{filename.stem}_recolored.{extension}")
        tmesh_segment.export(output_path / f"{filename.stem}_segmented.{extension}")

    return label2seam_edge_v(model, faces2label, tmesh)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/mesh_segmentation.yaml")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument(
        "--mode", type=str, choices=["face", "component"], default="component"
    )

    parser.add_argument("--cache", type=str)
    parser.add_argument("--cache_overwrite", action="store_true")
    parser.add_argument("--output", type=str)

    sam_group = parser.add_argument_group("sam")
    sam_group.add_argument("--sam.sam.checkpoint", type=str)
    sam_group.add_argument("--sam.sam.model_config", type=str)
    sam_group.add_argument("--sam.sam.engine_config.points_per_side", type=int)
    sam_group.add_argument("--sam.sam.engine_config.crop_n_layers", type=int)
    sam_group.add_argument("--sam.sam.engine_config.pred_iou_thresh", type=float)
    sam_group.add_argument("--sam.sam.engine_config.stability_score_thresh", type=float)
    sam_group.add_argument("--sam.sam.engine_config.stability_score_offset", type=float)

    sam_mesh_group = parser.add_argument_group("sam_mesh")
    sam_mesh_group.add_argument(
        "--sam_mesh.use_modes", nargs="+", type=str, choices=["matte", "norms"]
    )
    sam_mesh_group.add_argument("--sam_mesh.color_res", type=float)
    sam_mesh_group.add_argument("--sam_mesh.min_area", type=int)
    sam_mesh_group.add_argument("--sam_mesh.face2label_threshold", type=int)
    sam_mesh_group.add_argument("--sam_mesh.connections_threshold", type=int)
    sam_mesh_group.add_argument("--sam_mesh.counter_lens_threshold_min", type=int)
    sam_mesh_group.add_argument("--sam_mesh.connections_bin_resolution", type=int)
    sam_mesh_group.add_argument(
        "--sam_mesh.connections_bin_threshold_percentage", type=float
    )
    sam_mesh_group.add_argument(
        "--sam_mesh.smoothing_threshold_percentage_size", type=float
    )
    sam_mesh_group.add_argument(
        "--sam_mesh.smoothing_threshold_percentage_area", type=float
    )
    sam_mesh_group.add_argument("--sam_mesh.smoothing_iterations", type=int)
    sam_mesh_group.add_argument("--sam_mesh.repartition_cost", type=int)
    sam_mesh_group.add_argument("--sam_mesh.repartition_lambda", type=int)
    sam_mesh_group.add_argument("--sam_mesh.repartition_iterations", type=int)

    renderer_group = parser.add_argument_group("renderer")
    renderer_group.add_argument("--renderer.target_dim", nargs="+", type=int)
    renderer_group.add_argument(
        "--renderer.camera_params.type", type=str, choices=["orth", "pers"]
    )
    renderer_group.add_argument("--renderer.camera_params.xmag", type=float)
    renderer_group.add_argument("--renderer.camera_params.ymag", type=float)
    renderer_group.add_argument("--renderer.camera_params.fov", type=float)
    renderer_group.add_argument("--renderer.camera_params.yfov", type=float)
    renderer_group.add_argument("--renderer.camera_params.aspectRatio", type=float)
    renderer_group.add_argument("--renderer.camera_params.znear", type=float)
    renderer_group.add_argument("--renderer.camera_params.zfar", type=float)
    renderer_group.add_argument(
        "--renderer.camera_generation_method",
        type=str,
        choices=[
            "tetrahedron",
            "octohedron",
            "cube",
            "icosahedron",
            "dodecahedron",
            "standard",
            "swirl",
            "sphere",
        ],
        help="Method used to sample camera viewpoints.",
    )
    renderer_group.add_argument(
        "--renderer.sampling_args.radius",
        type=float,
        help="All methods require `radius`.",
    )
    renderer_group.add_argument(
        "--renderer.sampling_args.n",
        type=int,
        help="`sphere`, `standard` and `swirl`: requires `n` (number of samples).",
    )
    renderer_group.add_argument(
        "--renderer.sampling_args.elevation",
        type=int,
        help="`standard`: requires `elevation`.",
    )
    renderer_group.add_argument(
        "--renderer.sampling_args.cycles", type=int, help="`swirl`: requires `cycles`."
    )
    renderer_group.add_argument(
        "--renderer.sampling_args.elevation_range",
        type=int,
        help="`swirl`: requires `elevation_range`.",
    )

    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    for k, v in vars(args).items():
        k = k.split(".")
        if v is not None:
            node = config
            for i in k[:-1]:
                node = node[i]
            node[k[-1]] = v

    model = init_model(config)
    tmesh = load_model(args.filename)

    boundary_edges = workflow(
        config, model, tmesh, args.filename, args.mode, args.visualize
    )
