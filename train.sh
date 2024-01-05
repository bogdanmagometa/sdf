OBJ_NUM=0

mkdir -p xyz_dir
mkdir -p siren_weights

python3 train_siren.py --epochs 50 --obj_path ./test_task_meshes/${OBJ_NUM}.obj --xyz_path ./xyz_dir/${OBJ_NUM}.xyz --weights_path ./siren_weights/${OBJ_NUM}.pt
