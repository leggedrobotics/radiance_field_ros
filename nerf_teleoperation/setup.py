## ! DO NOT MANUALLY INVOKE THIS setup.py, USE CATKIN INSTEAD

from distutils.core import setup
if __name__ == "__main__":
    setup()
    exit()

from catkin_pkg.python_setup import generate_distutils_setup

d = generate_distutils_setup(
    package_dir={'': 'src'}
)

setup(**d)