#!/usr/bin/env bash

inicio=$(date +%s)

MUSICDIR="/home/fernando/machines/MUSIC-3.1"
iS3DDIR="/home/fernando/machines/iS3D-master"
AFTERBURNERDIR="/home/fernando/machines/urqmd-afterburner-master"
ICDIR="/home/fernando/machines/11-OO_k10_90a100/FS"
ORIGINALICDIR="/home/fernando/machines/11-OO_k10_90a100/IC_90a100"

BASEDIR="/home/fernando/machines/11-OO_k10_90a100"
cd "$BASEDIR"

for eventNo in {000..001}; do
 echo "$eventNo" >> eventNo.txt

 mkdir "$BASEDIR/event_$eventNo"
 cd "$BASEDIR/event_$eventNo"
 DIRBASE=`pwd`

 # MUSIC
 echo "Now running MUSIC"
 cp ../parameters/music_input_mode_2 ./music_input
 echo "Initial_Distribution_input_filename $ICDIR/${eventNo}_FS_v1p0.dat" >> music_input
 echo "EndOfData" >> music_input
 ln -s $MUSICDIR/EOS EOS
 ln -s $MUSICDIR/tables tables
 $MUSICDIR/MUSIChydro music_input >> music.log
 mkdir input
 mv surface*.dat ./input/surface.dat
 rm tables

 #iS3D
 echo "Now running iS3D" 
 mkdir results
 ln -s $iS3DDIR/PDG PDG
 ln -s $iS3DDIR/deltaf_coefficients deltaf_coefficients
 ln -s $iS3DDIR/tables tables
 ln -s ../parameters/iS3D_parameters.dat iS3D_parameters.dat
 $iS3DDIR/iS3D >> iS3D.log

 # Afterburner
 echo "Now running afterburner"
 $AFTERBURNERDIR/afterburner results/particle_list_osc.dat particle_list.f19 >> afterburner.log

 cd "$DIRBASE"
 #find "$BASEDIR/event_$eventNo" -type f ! -name "particle_list.f19" -delete
done

fim=$(date +%s)
tempo_total=$((fim - inicio))

horas=$((tempo_total / 3600))
minutos=$(( (tempo_total % 3600) / 60 ))
segundos=$((tempo_total % 60))

echo "Tempo total gasto: $horas horas, $minutos minutos, $segundos segundos."