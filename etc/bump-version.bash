#!/bin/bash

export VERSION=$(uvx dunamai from any --bump --style pep440)
uvx --from=toml-cli toml set --toml-path=pyproject.toml project.version $VERSION