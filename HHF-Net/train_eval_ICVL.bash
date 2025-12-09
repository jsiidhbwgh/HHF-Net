folder="icvl_experiment"

ex0="export ICVL_PATH=/home/stc/stc/TriHorn-Net/data/ICVL"


ex1="python main.py --check $folder --config_file configs/icvl.yaml"
ex2="python eval.py --path $folder/checkpoints"


$ex0
$ex1
$ex2
