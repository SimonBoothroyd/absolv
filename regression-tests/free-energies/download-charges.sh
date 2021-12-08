systems=(
"methane"
"methanol"
"ethane"
"toluene"
"neopentane"
"2-methylfuran"
"2-methylindole"
"2-cyclopentanylindole"
"7-cyclopentanylindole"
)

rm -rf reference-charges && mkdir reference-charges
cd reference-charges

for system in ${systems[@]}
do

  echo $system
  curl https://raw.githubusercontent.com/halx/relative-solvation-inputs/ec698ac2a80b79448f2b0f1d0c9aa811c6f1fe78/FESetup/setup/$system/gaff.mol2 --output "$system.mol2"

  antechamber -i "$system.mol2" -fi mol2 -o "$system.mol2" -fo mol2 -at sybyl
  rm ANTECHAMBER_* ATOMTYPE.INF

done

cd -