#!/usr/bin/env sh

# path to benchmark
#benchmark="./benchmark/mnist.py"
benchmark="./benchmark/CIFAR.py"

############### Neural network ############################
epochs=1
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
drift_coefficient=0.31
freq=1.2e9
temp=300

# CIFAR & ImageNet
wd=1e-4

{
python3 $benchmark  --epochs $epochs\
                    --batch_size $batch_size\
                    --test_batch_size $test_batch_size\
                    --scaler_dw $scaler_dw\
                    --lr $lr\
                    --crxb_size $crxb_size\
                    --vdd $vdd\
                    --gwire $gwire\
                    --gload $gload\
                    --gmax $gmax\
                    --gmin $gmin\
                    --test \
                    --enable_resistance_variance \
                    --enable_retention \
                    --resistance_variance_gamma $resistance_variance_gamma\
                    --retention_time $retention_time\
                    --drift_coefficient $drift_coefficient\
                    --freq $freq\
                    --temp $temp
} 

# for mac add "&" at the end
#&

# --enable_ec_SAF \
# --ir_drop \
# --test \
# --enable_noise \
# --no_cuda \
# --enable_SAF \
# --enable_resistance_variance \
# --enable_retention \