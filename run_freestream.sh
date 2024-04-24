#!/usr/bin/env bash
inicio=$(date +%s)

FSEXEDIR="/home/fernando/machines/freestream_velocity"
ICDIR="/home/fernando/machines/11-OO_k10_90a100/IC_90a100"

# Loop para percorrer todos os arquivos do diretório IC
for i in {000..199}; do
    file="$ICDIR/$i.dat"
    if [ -f "$file" ]; then
        filename=$(basename "$file" .dat)

        python3 "$FSEXEDIR/EMTensor_IO_music.py" "$ICDIR/${i}.dat" "${i}_FS_v1p0.dat" "${i}_FS_v1p0_EMtensor.dat" --time 0.37 --velocity 1.0 --grid-max 5.0 --renorm 1.0

        mv "${i}_FS_v1p0.dat" ./FS/.
        mv "${i}_FS_v1p0_EMtensor.dat" ./EMtensors/.
    fi
done

fim=$(date +%s)
tempo_execucao=$((fim-inicio))
echo "Tempo de execução: $tempo_execucao segundos"
