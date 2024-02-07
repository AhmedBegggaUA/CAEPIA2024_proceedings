for dataset in actor 
do
    for hidden_channels in 16 64 128
    do
        for hops in 3 4 5 10
        do
            for lr in 0.01 0.03 0.001 0.003
            do
                for dropout in 0.2 0.3 0.4 0.5
                do
                    for wd in 0.0005 0.001
                    do
                        echo "Running $dataset with $hidden_channels hidden channels and $hops hops and $lr learning rate and $dropout dropout and $wd weight decay"
                        python main.py --dataset $dataset --hidden_channels $hidden_channels --hops $hops --lr $lr --dropout $dropout  --epochs 1000  --wd $wd --cuda cuda:0
                        echo $(tail -n 1 results.csv)
                        # Now we print the best row with the highest Accuracy (penuultimate column) 
                        echo "Best row:"
                        echo $(sort -t, -k 6 -n results.csv | tail -n 1)
                    done
                done
            done
        done
    done
done