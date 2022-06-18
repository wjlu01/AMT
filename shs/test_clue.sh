dataset=clue
task_type=multilabel
lr=4e-5
lb=multi_label
device=1


echo " ========================================= Task: $lb ================================================ "
python amt/train.py --device ${device} --batch_sz 8 --gradient_accumulation_steps 40 --model AMT \
 --savedir save/AMT/${dataset}/${lb}/ --name model \
 --data_path amt/data/ --dataset ${dataset} \
 --task_type ${task_type} \
 --label ${lb} --output_mode avg \
 --num_image_embeds 6 \
 --n_workers 12 \
 --patience 5 --dropout 0.1 --lr $lr --warmup 0.1 --max_epochs 0 --seed 1
