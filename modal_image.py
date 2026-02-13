import modal

# Base image with heavy dependencies installed once
base_image = (
    modal.Image.debian_slim()
    .pip_install_from_requirements("requirements.txt")
)

