import argparse
import logging
import math
import os
import typing

import cv2
import numpy as np

import analysis
import common
import synthesis
import vector_node


class GuidanceImage:
    def __init__(self, image_path: str, intensity_category_map: typing.Dict[int, int]):
        self.guidance_raster = common.loader.load_image(image_path, grayscale=True, normalize=False)
        self.intensity_category_map = intensity_category_map

    def get_descriptor_map(self) -> np.ndarray:
        raster = np.zeros_like(self.guidance_raster, dtype=int)

        for intensity, category in self.intensity_category_map.items():
            raster[self.guidance_raster == intensity] = category

        return raster

    def get_defined_categories(self) -> typing.List[int]:
        return list(self.intensity_category_map.values())


class SynthesizeTexture:
    def __init__(self):
        self.placement_tries: int = 10
        self.improvement_steps: int = 5
        self.max_fails: int = 30
        self.selection_probability_decay: float = 2
        self.log_steps_path: typing.Union[str, None] = None
        self.radius_percentile: int = 50
        self.save_only_result: bool = False
        self.raster_only: bool = False
        self.skip_density_correction: bool = False
        self.guidance_image: typing.Union[GuidanceImage, None] = None

    def synthesize_texture(
            self, input_dir: str, size: typing.Tuple[int, int] = None,
    ) -> typing.Tuple[vector_node.VectorNode, vector_node.VectorNode, common.gradient_field.GradientField]:

        if size is None and self.guidance_image is None:
            raise ValueError("You must specify either the 'size' or 'guidance_image' parameter!")

        if self.guidance_image is None:
            descriptor_map = None
            skip_categories = []

        else:
            descriptor_map = self.guidance_image.get_descriptor_map()
            skip_categories = self.guidance_image.get_defined_categories()
            size = descriptor_map.shape[::-1]

        input_primary_textons = analysis.PrimaryTextonResult.load(os.path.join(input_dir, "primary_textons.dat"))
        input_secondary_textons = analysis.SecondaryTextonResult.load(os.path.join(input_dir, "secondary_textons.dat"))
        input_gradient_field = analysis.GradientFieldResult.load(os.path.join(input_dir, "gradient_field.dat"))

        weights = synthesis.Weights()

        logging.info("Synthesizing primary texton distribution")
        primary_textons = synthesis.generate_primary_texton_distro(
            input_primary_textons.primary_textons, size, placed_descriptors=descriptor_map,
            placement_tries=self.placement_tries, improvement_steps=self.improvement_steps, max_fails=self.max_fails,
            selection_probability_decay=self.selection_probability_decay, log_steps_directory=self.log_steps_path,
            weights=weights
        )

        if self.skip_density_correction:
            logging.info("Skipping primary texton density")

        else:
            logging.info("Correcting primary texton density")
            synthesis.primary_density_cleanup(
                input_primary_textons.primary_textons, primary_textons,
                skip_categories=skip_categories
            )

        logging.info("Synthesizing secondary texton distribution")
        secondary_textons = synthesis.generate_secondary_texton_distro(
            size, input_secondary_textons.element_spacing, input_secondary_textons.secondary_textons,
            background_color=input_secondary_textons.secondary_textons.color, percentile=self.radius_percentile
        )

        if input_gradient_field.density == math.inf or len(input_gradient_field.colors) == 0:
            logging.warning("Gradient field was not successfully extracted! Using solid colored background instead!")
            gradient_field = common.gradient_field.GradientField(
                np.array([]), np.array([]), math.inf, np.full((*size[::-1], 3), secondary_textons.color), size
            )

        else:
            logging.info("Synthesizing background gradient field")
            gradient_field_points, gradient_field_colors = synthesis.generate_gradient_field(
                input_gradient_field.colors, input_gradient_field.density, size
            )
            gradient_field = common.gradient_field.rasterize(
                gradient_field_points, gradient_field_colors, input_gradient_field.density, size
            )

            logging.info("Recomputing secondary texton color...")
            for texton in secondary_textons.children:
                if texton.color_delta is None:
                    continue

                centroid = np.clip(texton.get_centroid().astype(int), 0, size[::-1])
                field_color = gradient_field[*centroid]

                color_change = (field_color + texton.color_delta) - texton.color

                for node in texton.level_order_traversal(include_self=True):
                    node.color = np.clip(node.color + color_change, 0, 1)

            gradient_field = common.gradient_field.GradientField(
                gradient_field_points, gradient_field_colors, input_gradient_field.density, gradient_field, size
            )

        return primary_textons, secondary_textons, gradient_field

    def save_synthesis(
            self, output_dir: str, primary_textons: vector_node.VectorNode, secondary_textons: vector_node.VectorNode,
            gradient_field: common.gradient_field.GradientField
    ) -> str:

        logging.info("Saving synthetic files to {}...".format(os.path.abspath(output_dir)))
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)

        if self.save_only_result:
            logging.warning(
                "'save_only_result' has been set to True, so only rasterized and polygon results will be saved!"
            )

        else:
            logging.info("Saving data files...")
            if not self.raster_only:
                primary_textons.save(os.path.join(output_dir, "primary_textons.dat"))
                secondary_textons.save(os.path.join(output_dir, "secondary_textons.dat"))
                gradient_field.save(os.path.join(output_dir, "gradient_field.dat"))

            logging.info("Saving raster files...")
            primary_textons.to_raster(os.path.join(output_dir, "primary_textons.png"), background_color=np.zeros(3))
            secondary_textons.to_raster(os.path.join(output_dir, "secondary_textons.png"), background_color=np.zeros(3))

            if gradient_field.raster is None or gradient_field.raster.flatten()[0] is None:
                gradient_field.raster = np.zeros(
                    (int(primary_textons.get_bounding_height()), int(primary_textons.get_bounding_width()), 3)
                )

            bgr_image = cv2.cvtColor((gradient_field.raster * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(output_dir, "gradient_field.png"), bgr_image)

        logging.info("Merging layers into final result and saving to disk...")
        merged = secondary_textons
        merged.add_children(primary_textons.children)

        if not self.raster_only:
            merged.save(os.path.join(output_dir, "result.dat"))

        merged.to_raster(
            os.path.join(output_dir, "result.png"), background_image=gradient_field.raster,
            transparent_background=False
        )

        logging.info("Exported resulting image as '{}'".format(os.path.abspath(os.path.join(output_dir, "result.png"))))

        return os.path.join(output_dir, "result.png")

    def synthesize_and_save(self, filename: str, output_dir: str, size: typing.Tuple[int, int] = None):
        image_components = self.synthesize_texture(filename, size)
        self.save_synthesis(output_dir, *image_components)

    def from_argv(self):
        parser = argparse.ArgumentParser(
            description="Synthesizes a novel picture after 'analyze_picture.py' has completed the preprocessing",
            epilog='If there are any issues or questions, feel free to visit our GitHub repository at '
                   'https://github.com/ChrisPGraphics/BreakingArt'
        )

        parser.add_argument(
            'image_name',
            help="The name of the processed file. "
                 "Note that this is not the path to the image nor the directory in the intermediate folder. "
                 "Just the base name of the image without file extension. "
                 "For example, if the image path is /foo/bar.png and you analyzed the image with "
                 "'python analyze_picture.py /foo/bar.png', "
                 "you would call this script with 'python synthesize_picture.py bar'"
        )
        parser.add_argument(
            '--intermediate_directory',
            help="The path to the intermediate directory to store converted files"
        )
        parser.add_argument(
            '--output_directory',
            help="The path to the output directory to store synthetic files including the result"
        )
        parser.add_argument(
            '--log_level', default='INFO', choices=['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'NOTSET'],
            help="The logging level to run the script as"
        )
        parser.add_argument(
            '--width', default=500, type=int, help="The width of the resulting image, default is 500"
        )
        parser.add_argument(
            '--height', default=500, type=int, help="The height of the resulting image, default is 500"
        )
        parser.add_argument('--placement_tries', default=10, type=int)
        parser.add_argument('--improvement_steps', default=5, type=int)
        parser.add_argument('--max_fails', default=30, type=int)
        parser.add_argument('--selection_probability_decay', default=2, type=float)
        parser.add_argument('--log_steps_path', default=None, type=str)
        parser.add_argument('--radius_percentile', default=50, type=int)
        parser.add_argument(
            '--result_only', action='store_true',
            help="Only the merged result will be saved instead of also saving all layers separately too"
        )
        parser.add_argument(
            '--raster_only', action='store_true',
            help="Save only raster (image) files, and not vector files for further processing"
        )
        parser.add_argument(
            '--skip_density_correction', action='store_true',
            help="If enabled, there will be no attempt to correct the per-category density of the primary texton layer"
        )

        args = parser.parse_args()

        if args.intermediate_directory is None:
            intermediate_directory = os.path.join("intermediate", args.image_name)
        else:
            intermediate_directory = args.intermediate_directory

        if args.output_directory is None:
            output_directory = os.path.join("output", args.image_name)
        else:
            output_directory = args.output_directory

        self.placement_tries = args.placement_tries
        self.improvement_steps = args.improvement_steps
        self.max_fails = args.max_fails
        self.selection_probability_decay = args.selection_probability_decay
        self.log_steps_path = args.log_steps_path
        self.radius_percentile = args.radius_percentile
        self.save_only_result = args.result_only
        self.raster_only = args.raster_only
        self.skip_density_correction = args.skip_density_correction

        self.synthesize_and_save(intermediate_directory, output_directory, (args.width, args.height))


if __name__ == '__main__':
    common.configure_logger()

    synth = SynthesizeTexture()
    synth.from_argv()
