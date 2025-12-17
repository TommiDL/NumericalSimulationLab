sed -i -e 's/NMIGR.*/NMIGR\t\t\t'"$1"'/g' INPUT/input.dat

cat INPUT/input.dat
mkdir OUTPUT/varN/N$1
mpiexec -np 6 ./ga.exe
mv OUTPUT/*.dat OUTPUT/varN/N$1/