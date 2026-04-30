#!/bin/bash
# Double-click this file in Finder to start the Transcriber web server.
cd "$(dirname "$0")"
pipenv run python app.py
