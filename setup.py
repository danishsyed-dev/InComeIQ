from setuptools import find_packages, setup
from typing import List


def get_requirements(filepath: str) -> List[str]:
    """Read requirements from file, filtering out editable install flags."""
    requirements = []

    with open(filepath) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.strip() for req in requirements if req.strip()]

        # Remove editable install flag if present
        if "-e ." in requirements:
            requirements.remove("-e .")

    return requirements


setup(
    name="ml_income_predictor",
    version="1.0.0",
    description="ML-powered income bracket prediction using the Adult Census dataset",
    author="Danish",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt"),
)
