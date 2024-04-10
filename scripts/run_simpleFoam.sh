#!/bin/sh
cd ${0%/*} || exit 1    # Run from this directory
echo ${0%/*} 
source $HOME/OpenFOAM/OpenFOAM-4.x/etc/bashrc WM_LABEL_SIZE=64 FOAMY_HEX_MESH=yes

simpleFoam
