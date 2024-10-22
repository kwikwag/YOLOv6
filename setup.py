# -*- coding: utf-8 -*-

from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as f:
    readme = f.read()

setup(
    name="yolov6",
    version="0.4.1",
    description="YOLOv6",
    long_description=readme,
    author="Meituan",
    author_email="info@meituan.com",
    url="https://gitlab.com/meituan/YOLOv6",
    license="GPLv3",
    packages=find_packages(),
)
