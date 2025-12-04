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
    sam_group.add_argument("--sam.sam.checkpoint", type=str,
        help="Path to SAM model checkpoint file.")
    sam_group.add_argument("--sam.sam.model_config", type=str,
        help="Path to SAM model configuration file.")
    sam_group.add_argument("--sam.sam.engine_config.points_per_side", type=int,
        help="Controls the density of the automatic sampling grid. A value of 8 places 8 sample points along each side of the image, producing a total of 8 × 8 = 64 points. Speed priority: reduce to 4-6; accuracy priority: increase to 16-32.")
    sam_group.add_argument("--sam.sam.engine_config.crop_n_layers", type=int,
        help="Number of layers used for multi-scale segmentation. A value of 0 disables multi-scale processing for highest speed. Increasing to 2-3 improves small object detection but significantly increases computation cost.")
    sam_group.add_argument("--sam.sam.engine_config.pred_iou_thresh", type=float,
        help="Threshold for predicted mask quality based on IoU (mask-object overlap). High precision: increase to 0.8-0.9; high recall: decrease to 0.4-0.6.")
    sam_group.add_argument("--sam.sam.engine_config.stability_score_thresh", type=float,
        help="Threshold for mask stability under small perturbations. Smooth edges: increase to 0.95-1.0; preserve details: decrease to 0.7-0.8.")
    sam_group.add_argument("--sam.sam.engine_config.stability_score_offset", type=float,
        help="Offset used in stability score computation. Standard use cases typically do not require adjustment.")

    sam_mesh_group = parser.add_argument_group("sam_mesh")
    sam_mesh_group.add_argument(
        "--sam_mesh.use_modes", nargs="+", type=str, choices=["matte", "norms"],
        help="Rendering modes to use. Choose 'matte' for masks, add 'norms' for normal information."
    )
    sam_mesh_group.add_argument("--sam_mesh.color_res", type=float,
        help="is use for DFS.")
    sam_mesh_group.add_argument("--sam_mesh.min_area", type=int,
        help="Connected component size threshold (in pixels) for removing small artifacts from binary masks using OpenCV's connectedComponentsWithStats. High noise: increase to 64-128; preserve details: decrease to 8-16.")
    sam_mesh_group.add_argument("--sam_mesh.face2label_threshold", type=int,
        help="Minimum occurrences of a face ID within a label region before the face-label association is considered valid. High precision: increase to 8-12; high recall: decrease to 2-3.")
    sam_mesh_group.add_argument("--sam_mesh.connections_threshold", type=int,
        help="Minimum shared-face observations required for a connection between two labels (from different views) to be kept in the match graph. High confidence: increase to 8-16; more connections: decrease to 1-2.")
    sam_mesh_group.add_argument("--sam_mesh.counter_lens_threshold_min", type=int,
        help="Lower bound on connections before treating a label as 'overly connected'. Used with percentile-based threshold to enforce graph sparsity. Stricter: increase to 20-30; more lenient: decrease to 0.")
    sam_mesh_group.add_argument("--sam_mesh.connections_bin_resolution", type=int,
        help="Number of histogram bins for modeling connection-strength ratio distribution. Fine-grained: increase to 150-200; coarse-grained: decrease to 50-80.")
    sam_mesh_group.add_argument(
        "--sam_mesh.connections_bin_threshold_percentage", type=float,
        help="Fraction of histogram's total area for adaptive cutoff bin; connections below corresponding ratio are discarded. Stricter: increase to 0.05-0.1; more lenient: keep 0.0."
    )
    sam_mesh_group.add_argument(
        "--sam_mesh.smoothing_threshold_percentage_size", type=float,
        help="Fractional size threshold for removing small connected components (by face count) relative to largest component. More aggressive: increase to 0.15-0.2; conservative: decrease to 0.05-0.08."
    )
    sam_mesh_group.add_argument(
        "--sam_mesh.smoothing_threshold_percentage_area", type=float,
        help="Fractional area threshold for removing small connected components (by surface area) relative to largest component. More aggressive: increase to 0.15-0.2; conservative: decrease to 0.05-0.08."
    )
    sam_mesh_group.add_argument("--sam_mesh.smoothing_iterations", type=int,
        help="Number of smoothing passes where unlabeled faces adopt the most common label among labeled neighbors. More filling: increase to 64-128; avoid over-smoothing: decrease to 8-16.")
    sam_mesh_group.add_argument("--sam_mesh.repartition_cost", type=int,
        help="Cost parameter for repartitioning. Adjust as needed.")
    sam_mesh_group.add_argument("--sam_mesh.repartition_lambda", type=float,
        help="Weight (λ) balancing data cost vs. smoothness cost in graph-cut energy: TotalCost = DataCost + λ × SmoothnessCost. Smoother: increase to 2-4; more details: decrease to 0.5-1.0.")
    sam_mesh_group.add_argument("--sam_mesh.repartition_iterations", type=int,
        help="Number of alpha-expansion cycles by graph-cut optimizer. Better convergence: increase to 3-8; fast processing: keep 1.")

    renderer_group = parser.add_argument_group("renderer")
    renderer_group.add_argument("--renderer.target_dim", nargs="+", type=int,
        help="Output resolution of rendered images. High quality: increase to 1024x1024; fast processing: decrease to 256x256.")
    renderer_group.add_argument(
        "--renderer.camera_params.type", type=str, choices=["orth", "pers"],
        help="Camera projection type: 'orth' for orthographic projection, 'pers' for perspective projection."
    )
    renderer_group.add_argument("--renderer.camera_params.xmag", type=float,
        help="Orthographic camera magnification in X direction.")
    renderer_group.add_argument("--renderer.camera_params.ymag", type=float,
        help="Orthographic camera magnification in Y direction.")
    renderer_group.add_argument("--renderer.camera_params.fov", type=float,
        help="Field of view angle for perspective camera.")
    renderer_group.add_argument("--renderer.camera_params.yfov", type=float,
        help="Vertical field of view angle for perspective camera.")
    renderer_group.add_argument("--renderer.camera_params.aspectRatio", type=float,
        help="Aspect ratio (width/height) for perspective camera.")
    renderer_group.add_argument("--renderer.camera_params.znear", type=float,
        help="Near clipping plane distance.")
    renderer_group.add_argument("--renderer.camera_params.zfar", type=float,
        help="Far clipping plane distance.")
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
            "edge",
        ],
        help="Method for sampling camera viewpoints. Choose based on object shape.",
    )
    renderer_group.add_argument(
        "--renderer.sampling_args.radius",
        type=float,
        help="Camera distance from object center. Required for all methods.",
    )
    renderer_group.add_argument(
        "--renderer.sampling_args.n",
        type=int,
        help="Number of camera samples. Required for 'sphere', 'standard', and 'swirl' methods.",
    )
    renderer_group.add_argument(
        "--renderer.sampling_args.elevation",
        type=int,
        help="Camera elevation angle. Required for 'standard' method.",
    )
    renderer_group.add_argument(
        "--renderer.sampling_args.cycles", type=int, 
        help="Number of swirl cycles. Required for 'swirl' method."
    )
    renderer_group.add_argument(
        "--renderer.sampling_args.elevation_range",
        type=int,
        help="Elevation angle range for swirl pattern. Required for 'swirl' method.",
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
