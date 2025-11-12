#!/bin/bash

wget  https://srtm.csi.cgiar.org/wp-content/uploads/files/srtm_30x30/TIFF/N00E060.zip
wget https://srtm.csi.cgiar.org/wp-content/uploads/files/srtm_30x30/TIFF/N00E090.zip
wget https://srtm.csi.cgiar.org/wp-content/uploads/files/srtm_30x30/TIFF/N30E060.zip
wget https://srtm.csi.cgiar.org/wp-content/uploads/files/srtm_30x30/TIFF/N30E090.zip
wget https://srtm.csi.cgiar.org/wp-content/uploads/files/srtm_30x30/TIFF/N30E120.zip
wget https://srtm.csi.cgiar.org/wp-content/uploads/files/srtm_30x30/TIFF/N00E120.zip

mkdir -p tif

unzip -d tif N00E060.zip
unzip -d tif N00E090.zip
unzip -d tif N30E060.zip
unzip -d tif N30E090.zip
unzip -d tif N30E120.zip
unzip -d tif N00E120.zip
