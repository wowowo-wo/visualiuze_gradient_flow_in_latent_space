from setuptools import setup, find_packages

setup(
    name="vgf",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "torch",
        "diffusers",
        "transformers",
        "Pillow",
        "matplotlib"
    ],
    extras_require={
        "ssim":["pytorch-msssim"],
        "lpips": ["lpips"],
        "sinkhorn": ["geomloss","pykeops"]
    },
    entry_points={
        "console_scripts": [
            "vgfls=vgf.main:main",
        ]
    },
)