#!/usr/bin/env sh

# path to benchmark
benchmark="./benchmark/mnist.py"
#benchmark="./benchmark/CIFAR.py"

############### Neural network ############################
epochs=20
batch_size=100
test_batch_size=100
scaler_dw=1
lr=0.01

crxb_size=128
vdd=1
gwire=0.375
gload=0.25
gmax=3.85e-8
gmin=5e-9
resistance_variance_gamma=0.1
retention_time=1e2
drift_coefficient=0.1
freq=1.2e9
temp=300

# CIFAR & ImageNet
wd=1e-4

{
python3 $benchmark  --epochs $epochs\
                    --batch_size $batch_size\
                    --test_batch_size $test_batch_size\
                    --lr $lr\
                    --crxb_size $crxb_size\
                    --vdd $vdd\
                    --freq $freq\
                    --gwire $gwire\
                    --gload $gload\
                    --gmax $gmax\
                    --gmin $gmin\
                    --test \
                    --freq $freq\
                    --temp $temp
} &

# --enable_ec_SAF \
# --ir_drop \
# --test \
# --enable_noise \
# --no_cuda \
# --enable_SAF \
# --enable_resistance_variance \
# --enable_retention \