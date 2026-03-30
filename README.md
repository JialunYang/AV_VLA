# OpenEMMA Baseline Reproduction

This repository contains a baseline reproduction of the OpenEMMA framework for AV_VLA project.

## Setup
- Git clone OpenEMMA repo
- Install dependencies from requirements.txt in OpenEMMA repo
- Configure OpenAI API key

## Execution
Run:
python scripts/test_main.py

## Outputs
The pipeline generates:
- Predicted trajectories
- Reasoning outputs (scene, object, intent)
- Annotated images and videos

## Notes
- Dataset: nuScenes v1.0 mini
- Backend: GPT-4o via OpenAI API