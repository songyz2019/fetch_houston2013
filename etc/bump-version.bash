#!/bin/bash

export VERSION=$(uvx dunamai from git --bump --no-metadata --style pep440)
uvx --from=toml-cli toml set --toml-path=pyproject.toml project.version $VERSION