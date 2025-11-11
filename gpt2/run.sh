echo "runing task $1"

seeds=(1234 421)

methods=("adapter0" "adapter")

lrs=("1e-4" "5e-4"  "8e-4" "1e-3" "2e-3" "3e-3")

for lr in ${lrs[@]};
do
 for seed in ${seeds[@]};
   do
      for method in ${methods[@]};
           do 
              echo  "==========start to run on task:$1  method:${method}  seed:${seed} lr:${lr}==========="
              echo "python train.py configs/$1/${method}.json ${seed} ${lr}"
              python train.py configs/$1/${method}.json ${seed} ${lr}
              echo  "==========finished run on task:$1  method:${method}  seed:${seed}  lr:${lr} ==========="
              echo "-------------------------------------------------------------------------------"
              echo "           "
           done
   done
 done



