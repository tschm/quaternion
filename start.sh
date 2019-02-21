#!/usr/bin/env bash
echo "http://localhost:9015"
docker-compose run -p "9015:8888" web
