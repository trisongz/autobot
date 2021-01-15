import os
import sys
import os.path

from setuptools.command.install import install as _install
from setuptools import find_packages, setup

root = os.path.abspath(os.path.dirname(__file__))
package_name = "auto"
packages = find_packages(
    include=[package_name, "{}.*".format(package_name)]
)

with open("README.md") as f:
    long_description = f.read()

if __name__ == "__main__":
    setup(
        name="autobot",
        version="0.0.1",
        description="autobot - trainer for transformers bc why not",
        long_description=long_description,
        long_description_content_type="text/markdown",
        author="Tri Songz",
        author_email="ts@contentenginex.com",
        url="https://github.com/trisongz/autobot",
        license="Apache License",
        packages=packages,
        include_package_data=True,
        install_requires=[
            "rich",
            "typer",
            "pysimdjson"
            ],
        platforms=["linux", "unix"],
        python_requires=">3.6",
        entry_points={
            "console_scripts": [
                "autobot = auto.cli:cli"
            ]
        }
    )
