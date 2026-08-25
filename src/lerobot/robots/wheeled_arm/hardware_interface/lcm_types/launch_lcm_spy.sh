#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd ${DIR}/java
export CLASSPATH="${DIR}/java/my_types.jar:${DIR}/java/lcm.jar"
pwd

if [ $# == 0 ];then
  com=8880
else
  com=$1
fi

lcm-spy --lcm-url=udpm://239.255.76.67:$com?ttl=255
