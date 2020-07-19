#!/bin/bash

rm -f vid.mp4
ffmpeg -r 15 -s 1280x720  -i image_0_0_0.%05d.tga -vcodec libx264 -crf 15 vid.mp4

