#!/bin/bash

if [ -z "$1" ]
  then
    echo "Please input name model file"
    exit 1
fi
cd ..
cp ./Speech-Recognition-model/models/$1 ./Smart-Home-Control-Unit/cpro/model/model-final.h5 
cd ./Smart-Home-Control-Unit
git add .
git commit -a -m "UPDATE model $1"
git push
