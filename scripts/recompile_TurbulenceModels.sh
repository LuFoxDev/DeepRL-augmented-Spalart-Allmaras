#!/bin/sh
source $HOME/OpenFOAM/OpenFOAM-4.x/etc/bashrc WM_LABEL_SIZE=64 FOAMY_HEX_MESH=yes

/home/lukas/OpenFOAM/OpenFOAM-4.x/src/TurbulenceModels/Allwmake
