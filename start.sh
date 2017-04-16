#!/usr/bin/env bash
/usr/bin/docker run -d -p 2036:9999 -v $(pwd)/books:/jupyter tschm/presentation:v1.1
