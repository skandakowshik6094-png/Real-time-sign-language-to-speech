import json
import os

# Always resolve config.json relative to this file's directory (src/)
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_CONFIG_PATH = os.path.join(_SRC_DIR, '..', 'config.json')
# Normalise — src/utils/../config.json  → src/config.json
_CONFIG_PATH = os.path.normpath(os.path.join(_SRC_DIR, 'config.json'))


def get_classes():
    try:
        with open(_CONFIG_PATH) as f:
            config = json.load(f)
        classes = config['classes']
        assert len(classes) > 0, (
            'You need to specify classes inside of config.json '
            'e.g. {"classes":["hello", "iloveyou", "hola"]}'
        )
        return classes
    except Exception as e:
        return f'Something went wrong loading your config file: {e}'


def get_colors():
    try:
        with open(_CONFIG_PATH) as f:
            config = json.load(f)
        classes = config['classes']
        colors = config['colors']
        assert len(colors) > 0, (
            'You need to specify colors in RGB inside of config.json '
            'e.g. {"colors":[[131, 193, 103], [240, 172, 95]]}'
        )
        assert len(classes) == len(colors), (
            f'Please specify one color per class. '
            f'You have {len(colors)} colours and {len(classes)} classes'
        )
        return colors
    except Exception as e:
        return f'Something went wrong loading your config file: {e}'


if __name__ == '__main__':
    classes = get_classes()
    print(classes)