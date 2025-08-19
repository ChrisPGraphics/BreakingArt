# BreakingArt
The official code repository for the paper "Breaking Art: Synthesizing Abstract Expressionism Through Image Rearrangement" by Christopher Palazzolo, Oliver van Kaick, and David Mould

[[Paper](https://doi.org/10.1016/j.cag.2025.104224)] [[Project Page](https://chrispgraphics.github.io/2025_Breaking_art_Synthesizing_abstract_expressionism_through_image_rearrangement.html)] [[Art Gallery Tour](https://www.youtube.com/watch?v=JAzxg2WRWDw)]

## Installation
First, clone the repository to your local machine.
```shell
git clone https://github.com/ChrisPGraphics/BreakingArt.git
cd BreakingArt
```

From here, you can either install the repository to your [Local Machine](#local-machine-install-instructions) or to a [Docker Container](#docker-install-instructions).

### Local Machine Install Instructions
Assuming you have Python installed (we use Python 3.11.5), follow these instructions to prepare your local machine to use our code. 

```shell
pip install -r requirements.txt
```

If you are using a Linux-based OS, you may need to also run the following commands to install dependencies for Open-CV2.

```shell
apt-get update
apt-get install ffmpeg libsm6 libxext6 -y
```

### Docker Install Instructions
Simply run the following commands.
```shell
docker build -t breaking_art .
docker run -it breaking_art /bin/bash
```

Now that your Docker container is built and running, you can proceed to the [Usage Instructions](#usage).

## Usage
Our process contains two steps. First is [Analysis](#analysis) which converts a natural image into our vector hierarchy format. Once that is complete, [Synthesis](#synthesis) can be run to create a novel image.

### Analysis
To analyze an image, run the command `python analyze_picture.py`. The following is the usage of the script.
```
usage: analyze_picture.py [-h] [--log_level {CRITICAL,ERROR,WARNING,INFO,DEBUG,NOTSET}] [--average_included AVERAGE_INCLUDED] input_image

Performs all of the necessary preprocessing before a novel image can be synthesized

positional arguments:
  input_image           The path to the texture that should be processed

options:
  -h, --help            show this help message and exit
  --log_level {CRITICAL,ERROR,WARNING,INFO,DEBUG,NOTSET}
                        The logging level to run the script as
  --average_included AVERAGE_INCLUDED
                        The average number of textons to be included in each direction of all descriptors

If there are any issues or questions, feel free to visit our GitHub repository at https://github.com/ChrisPGraphics/BreakingArt
```

### Synthesis
To synthesize an analyzed image, run the command `python synthesize_picture.py` file.

**NOTE**: For this script, you do not provide the path to the image nor the directory in the intermediate folder. 
Simply provide the base name of the image without file extension.
For example, if the image path is `/foo/bar.png` and you analyzed the image with `python analyze_picture.py /foo/bar.png`, you would call this script with `python synthesize_picture.py bar`

The following is the usage of the script.
```
usage: synthesize_picture.py [-h] [--intermediate_directory INTERMEDIATE_DIRECTORY] [--output_directory OUTPUT_DIRECTORY] [--log_level {CRITICAL,ERROR,WARNING,INFO,DEBUG,NOTSET}] [--width WIDTH]
                             [--height HEIGHT] [--placement_tries PLACEMENT_TRIES] [--improvement_steps IMPROVEMENT_STEPS] [--max_fails MAX_FAILS] [--selection_probability_decay SELECTION_PROBABILITY_DECAY]
                             [--log_steps_path LOG_STEPS_PATH] [--radius_percentile RADIUS_PERCENTILE] [--result_only] [--raster_only] [--skip_density_correction]
                             image_name

Synthesizes a novel texture after 'analyze_texture.py' has completed the preprocessing

positional arguments:
  image_name            The name of the processed file. Note that this is not the path to the image nor the directory in the intermediate folder. Just the base name of the image without file extension. For
                        example, if the image path is /foo/bar.png and you analyzed the image with 'python analyze_texture.py /foo/bar.png', you would call this script with 'python synthesize_texture.py bar'

options:
  -h, --help            show this help message and exit
  --intermediate_directory INTERMEDIATE_DIRECTORY
                        The path to the intermediate directory to store converted files
  --output_directory OUTPUT_DIRECTORY
                        The path to the output directory to store synthetic files including the result
  --log_level {CRITICAL,ERROR,WARNING,INFO,DEBUG,NOTSET}
                        The logging level to run the script as
  --width WIDTH         The width of the resulting image, default is 500
  --height HEIGHT       The height of the resulting image, default is 500
  --placement_tries PLACEMENT_TRIES
  --improvement_steps IMPROVEMENT_STEPS
  --max_fails MAX_FAILS
  --selection_probability_decay SELECTION_PROBABILITY_DECAY
  --log_steps_path LOG_STEPS_PATH
  --radius_percentile RADIUS_PERCENTILE
  --result_only         Only the merged result will be saved instead of also saving all layers separately too
  --raster_only         Save only raster (image) files, and not vector files for further processing
  --skip_density_correction
                        If enabled, there will be no attempt to correct the per-category density of the primary texton layer

If there are any issues or questions, feel free to visit our GitHub repository at https://github.com/ChrisPGraphics/BreakingArt
```

### Guidance Map
You can import the objects directly in a Python script to use advanced features like the guidance map like so.

```python
import common
import synthesize_picture

common.configure_logger()

synth = synthesize_picture.SynthesizeTexture()
synth.guidance_image = synthesize_picture.GuidanceImage(image_path, intensity_category_map)

synth.synthesize_and_save(intermediate_directory, output_directory, (width, height))
```

In the ``GuidanceImage`` constructor, ``image_path`` is the path to the greyscale image that is the guidance map. 
``intensity_category_map`` is a dictionary that has intensity values of the ``image_path`` as keys and the category ID of polygons that can be placed there as values.
Any unmapped value is assumed to allow any category.

For example, if category 1 should appear where the intensity of the grayscale image = 255, and the remainder of the area could contain any polygons, then the `intensity_category_map` would be `{255: 1}`.

## Evaluation Code
* [Learning Photography Aesthetics with Deep CNNs](https://github.com/rawmarshmellows/deep-photo-aesthetics) 
* [Recognizing Art Style Automatically in painting with deep learning](https://github.com/bnegreve/rasta)
* [Perceptual Similarity Metrics](https://github.com/chaofengc/IQA-PyTorch)

Thank you for checking out our paper and repository. If you have any issues, feel free to visit the [Issues](https://github.com/ChrisPGraphics/BreakingArt/issues) page of the repository.
