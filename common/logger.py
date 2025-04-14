import logging
import sys


def configure_logger(level=logging.INFO):
    for argv in sys.argv:
        if argv.startswith("--log_level="):
            level_name = argv[12:].upper()
            mapping = logging.getLevelNamesMapping()

            if level_name in mapping:
                level = mapping[level_name]

    logging.root.handlers = []
    logging.basicConfig(
        format='(%(asctime)s) [%(levelname)-8.8s] %(message)s',
        level=level,
        datefmt='%Y-%m-%d %H:%M:%S'
    )
