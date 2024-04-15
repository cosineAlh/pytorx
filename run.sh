#!/usr/bin/env sh

# path to benchmark
benchmark="./benchmark/mnist.py"

############### Neural network ############################
epochs=20
batch_size=1000
test_batch_size=100
crxb_size=64
vdd=3.3
gwire=0.375
gload=0.25
gmax=0.000333
gmin=0.0000000333
ir_drop=false
scaler_dw=1
test=false
enable_noise=false
enable_SAF=false
enable_ec_SAF=false
freq=10e6
temp=300

{
python3 $benchmark  --epochs $epochs\
                    --batch_size $batch_size\
                    --test_batch_size $test_batch_size\
                    --crxb_size $crxb_size\
                    --vdd $vdd\
                    --freq $freq\
                    --gwire $gwire\
                    --gload $gload\
                    --gmax $gmax\
                    --gmin $gmin\
                    --ir_drop $ir_drop\
                    --scaler_dw $scaler_dw\
                    --test $test\
                    --enable_noise $enable_noise\
                    --enable_SAF $enable_SAF\
                    --enable_ec_SAF $enable_ec_SAF\
                    --freq $freq\
                    --temp $temp
} &

