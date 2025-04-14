import argparse
import logging
import math
import os.path
import typing

import PIL.Image
import cv2
import numpy as np
import scipy.spatial
import tqdm

import analysis
import common
import segmentation
import texton_categorization
import vector_node


class AnalyzeImage:
    def __init__(
            self,
            primary_segmentation: segmentation.BaseSegmentation = segmentation.FloodFillSegmentation(0.2, 10, 300),
            secondary_segmentation: typing.Union[segmentation.BaseSegmentation, None] = segmentation.FloodFillSegmentation(0.1, 3, 300),
            detail_segmentation: typing.Union[segmentation.BaseSegmentation, None] = segmentation.FloodFillSegmentation(0.05),
            primary_texton_categorization: texton_categorization.BaseCategorization = texton_categorization.ColorAreaCompactnessCategorization(),
            average_included: float = 2.25
    ):
        self.primary_segmentation = primary_segmentation
        self.secondary_segmentation = secondary_segmentation
        self.detail_segmentation = detail_segmentation
        self.primary_texton_categorization = primary_texton_categorization
        self.average_included = average_included

        if self.detail_segmentation is not None:
            self.detail_segmentation.silent = True

    def analyze_image(self, filename: str) -> typing.Tuple[
        analysis.PrimaryTextonResult, analysis.SecondaryTextonResult, analysis.GradientFieldResult
    ]:
        logging.info("Reading image '{}'...".format(os.path.abspath(filename)))
        image = common.loader.load_image(filename)

        logging.info("Extracting primary textons...")
        primary_polygons, primary_remainder = analysis.extract_textons(
            image, self.primary_segmentation, self.detail_segmentation
        )

        if self.secondary_segmentation is None or np.count_nonzero(primary_remainder) < 5:
            secondary_polygons = vector_node.VectorNode(primary_polygons.exterior.copy())
            distances = []
            gradient_field_colors = []
            gradient_field_points = []
            density = math.inf

        else:
            logging.info("Extracting secondary textons...")
            secondary_polygons, secondary_remainder = analysis.extract_textons(
                image, self.secondary_segmentation, detail_segmentation=None, mask=primary_remainder
            )

            promoted = analysis.promote_textons(primary_polygons, secondary_polygons)

            if self.detail_segmentation is not None:
                logging.info("Extracting detail of promoted textons...")
                analysis.get_vector_node_detail(promoted, image, self.detail_segmentation)

            logging.info("Triangulating secondary texton centroids...")
            lines = analysis.get_secondary_texton_triangulation(primary_polygons, secondary_polygons)

            logging.info("Computing line distances...")
            distances = analysis.get_triangulation_distances(lines)
            logging.info("Found {} lines".format(len(distances)))

            try:
                logging.info("Extracting data points for gradient field...")
                gradient_field_colors, gradient_field_points, density = analysis.get_background_gradient_field(
                    image, secondary_remainder
                )

            except scipy.spatial.QhullError:
                logging.error(
                    "Not enough points could be extracted to create a background gradient field! "
                    "Using solid colored background instead!"
                )

                gradient_field_colors = []
                gradient_field_points = []
                density = math.inf

        if len(gradient_field_colors) > 0:
            logging.info("Computing color deltas...")
            for polygon in tqdm.tqdm(secondary_polygons.children):
                background_color = common.gradient_field.interpolate(
                    gradient_field_points, gradient_field_colors, density, polygon.get_centroid()
                )
                polygon.color_delta = polygon.color - background_color

        else:
            logging.warning(
                "Not enough pixels are left over to build a background gradient field! "
                "Use less aggressive segmentation parameters to fix this problem"
            )

        logging.info("Removing polygons that are too close to the edge")
        analysis.remove_edge_textons(primary_polygons)
        analysis.remove_edge_textons(secondary_polygons)

        logging.info("Categorizing polygons using {}".format(self.primary_texton_categorization.get_algorithm_name()))
        self.primary_texton_categorization.categorize(primary_polygons.children)

        logging.info("Computing primary texton descriptors...")
        descriptor_size = analysis.get_descriptors(primary_polygons, image.shape[:2][::-1], average_included=self.average_included)

        logging.info("{} selectable primary textons remain".format(len(primary_polygons.children)))

        return (
            analysis.PrimaryTextonResult(primary_polygons, descriptor_size),
            analysis.SecondaryTextonResult(secondary_polygons, distances),
            analysis.GradientFieldResult(gradient_field_points, gradient_field_colors, density)
        )

    def save_extraction(
            self, output_dir: str, filename: str, primary_textons: analysis.PrimaryTextonResult,
            secondary_textons: analysis.SecondaryTextonResult, gradient_field: analysis.GradientFieldResult
    ):
        logging.info("Saving extraction files to {}...".format(os.path.abspath(output_dir)))
        os.makedirs(output_dir, exist_ok=True)

        logging.info("Saving data files...")
        primary_textons.save(os.path.join(output_dir, "primary_textons.dat"))
        secondary_textons.save(os.path.join(output_dir, "secondary_textons.dat"))
        gradient_field.save(os.path.join(output_dir, "gradient_field.dat"))

        logging.info("Converting and copying exemplar...")
        image = common.loader.load_image(filename)
        bgr_image = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(output_dir, "exemplar.png"), bgr_image)

        logging.info("Saving raster version of extractions...")
        primary_textons.primary_textons.to_raster(
            os.path.join(output_dir, "primary_textons.png"), background_color=np.array([0, 0, 0])
        )

        secondary_textons.secondary_textons.to_raster(
            os.path.join(output_dir, "secondary_textons.png"), background_color=np.array([0, 0, 0])
        )

        if len(gradient_field.points) > 0:
            background_gradient = common.gradient_field.rasterize(
                gradient_field.points, gradient_field.colors, gradient_field.density, image.shape[:2][::-1]
            )
            bgr_image = cv2.cvtColor((background_gradient * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(output_dir, "gradient_field.png"), bgr_image)

        else:
            background_gradient = None

        logging.info("Merging vector layers into raster result and saving to disk...")

        if secondary_textons.secondary_textons.color is None:
            background_color = (0, 0, 0)
        else:
            background_color = tuple(secondary_textons.secondary_textons.color_to_int())

        raster = PIL.Image.new("RGB", tuple(image.shape[:2][::-1]), background_color)

        if background_gradient is not None:
            raster.paste(PIL.Image.open(os.path.join(output_dir, "gradient_field.png")))

        secondary_textons = PIL.Image.open(os.path.join(output_dir, "secondary_textons.png"))
        primary_textons = PIL.Image.open(os.path.join(output_dir, "primary_textons.png"))

        raster.paste(secondary_textons, (0, 0), secondary_textons)
        raster.paste(primary_textons, (0, 0), primary_textons)

        raster.save(os.path.join(output_dir, "vector_representation.png"))

    def analyze_and_save(self, filename: str, output_dir: str):
        image_components = self.analyze_image(filename)
        self.save_extraction(output_dir, filename, *image_components)

    def from_argv(self):
        parser = argparse.ArgumentParser(
            description="Performs all of the necessary preprocessing before a novel image can be synthesized",
            epilog='If there are any issues or questions, feel free to visit our GitHub repository at '
                   'https://github.com/ChrisPGraphics/BreakingArt',
        )

        parser.add_argument('input_image', help="The path to the picture that should be processed")
        parser.add_argument(
            '--log_level', default='INFO', choices=['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'NOTSET'],
            help="The logging level to run the script as"
        )
        parser.add_argument(
            '--average_included', type=float, default=2.25,
            help="The average number of textons to be included in each direction of all descriptors"
        )

        args = parser.parse_args()

        self.average_included = args.average_included
        self.analyze_and_save(
            args.input_image,
            os.path.join("intermediate", os.path.splitext(os.path.basename(args.input_image))[0])
        )


if __name__ == '__main__':
    common.configure_logger()

    analyzer = AnalyzeImage()
    analyzer.from_argv()
