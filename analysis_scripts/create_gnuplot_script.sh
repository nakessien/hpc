
# This scripts must be executed in a directory containing the loss function files
# see 'collect_data.sh'
echo 'plot 0 not'
echo 'set yrange [0.15:0.32]'
echo 'set key left bottom'
echo 'set xlabel "epochs"'
echo 'set ylabel "loss"'
wc -l *.tsv | \
grep out.tsv | \
awk '($1 > 1){print $2}'| \
sed 's/.out.tsv//' | \
xargs -I {} echo 'replot "<(tail -n +2 '{}'.out.tsv)" u 0:2 t "'{}'"  '
echo "pause -1"
echo "set term png"
echo 'set out "loss_trajectories.png"'
echo 'replot'
