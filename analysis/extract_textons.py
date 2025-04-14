import datetime
import logging
import typing

import numpy as np
import tqdm

import segmentation as segmentation_methods
import vector_node


def get_vector_node_detail(
        nodes: typing.List[vector_node.VectorNode], image: np.ndarray,
        detail_segmentation: segmentation_methods.BaseSegmentation
):
    for node in tqdm.tqdm(nodes):
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        node.binary_rasterize(mask)

        segments = detail_segmentation.segment(image, mask)

        for segment in segments:
            mask_node = vector_node.MaskNode(segment, color=np.median(image[segment], axis=0))
            node.children.append(mask_node.to_vector_node())
            mask = np.logical_and(mask, np.logical_not(segment))

        if np.any(mask):
            node.color = np.median(image[mask], axis=0)


def get_detail(
        parent: vector_node.MaskNode, image: np.ndarray, detail_segmentation: segmentation_methods.BaseSegmentation
):
    for child in tqdm.tqdm(parent.children):
        remaining_mask = child.mask.copy()

        children = detail_segmentation.segment(image, child.mask)

        for c in children:
            child.children.append(vector_node.MaskNode(c, color=np.median(image[c], axis=0)))
            remaining_mask = np.logical_and(remaining_mask, np.logical_not(c))

        if np.any(remaining_mask):
            child.color = np.median(image[remaining_mask], axis=0)


def extract_textons(
        image: np.ndarray, primary_segmentation: segmentation_methods.BaseSegmentation,
        detail_segmentation: typing.Union[segmentation_methods.BaseSegmentation, None] = None,
        mask: np.ndarray = None
) -> typing.Tuple[vector_node.VectorNode, np.ndarray]:

    start = datetime.datetime.now()

    if mask is None:
        remaining_mask = np.ones(image.shape[:2]).astype(bool)
    else:
        remaining_mask = mask.copy()

    background_color = np.median(image[remaining_mask], axis=0)

    logging.info("Segmenting textons with {}...".format(primary_segmentation.get_algorithm_name()))
    segments = primary_segmentation.segment(image, mask=remaining_mask)

    logging.info("Converting masks to tree structure...")
    parent = vector_node.MaskNode(np.ones(image.shape[:2]).astype(bool))

    for segment in tqdm.tqdm(segments):
        remaining_mask = np.logical_and(remaining_mask, np.logical_not(segment))
        mask_node = vector_node.MaskNode(segment, color=np.median(image[segment], axis=0))
        parent.add_child(mask_node)

    if detail_segmentation is not None:
        logging.info("Extracting detail for each identified segment...")
        detail_segmentation.silent = True
        get_detail(parent, image, detail_segmentation)

    logging.info("Converting binary masks to polygons...")
    polygons = parent.to_vector_node(silent=False)

    end = datetime.datetime.now()
    logging.info("Texton extraction took {}".format(end - start))

    if np.any(remaining_mask):
        background_color = np.median(image[remaining_mask], axis=0)

    polygons.color = background_color

    return polygons, remaining_mask


def remove_edge_textons(textons: vector_node.VectorNode, buffer: int = 3):
    min_x, min_y, max_x, max_y = textons.as_shapely().bounds

    for texton in textons.children[:]:
        texton_min_x, texton_min_y = texton.exterior.min(axis=0)
        texton_max_x, texton_max_y = texton.exterior.max(axis=0)

        if (
                texton_min_x < min_x + buffer or
                texton_min_y < min_y + buffer or
                texton_max_x > max_x - buffer or
                texton_max_y > max_y - buffer
        ):
            textons.remove_child(texton)
