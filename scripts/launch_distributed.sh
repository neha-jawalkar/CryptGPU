python ./distributed_launcher.py \
    --prepare_cmd="export PATH=/home/ezpcuser/anaconda3/envs/cryptgpu/bin:$PATH" \
    --aux_files=benchmark.py,network.py \
    --cuda_visible_devices="0"\
    launcher.py \
