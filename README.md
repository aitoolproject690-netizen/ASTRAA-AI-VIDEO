# ASTRAA AI VIDEO

Free/open-source AI video workflow for the ASTRAA series.

## Goal
Build a reliable ASTRAA image-to-video workflow with code maintained in GitHub. Codespaces is used for code fixes and testing; a GPU runtime is used only for video generation.

## Current approach
- Google Colab + free GPU when available
- LTX-Video 0.9.8 distilled 2B model for lighter VRAM usage
- Image-to-video workflow
- Short shots can be extended/assembled in CapCut

## Development
Open the repository in GitHub Codespaces. See **CODESPACES.md**.

## GPU generation
**ASTRAA_LTX_Colab.ipynb** contains the LTX-Video GPU workflow. Free GPU availability and runtime duration are controlled by the provider and are not guaranteed. This repository does not bypass any service limits.

## Safety
Never put passwords, API keys, access tokens, or private credentials in this repository.
