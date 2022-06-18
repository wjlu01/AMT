dataset=bloomberg
task_type=classification
device=1
lr=4e-5

for lb in img_label txt_label it_label
do
echo " ========================================= Task: $lb ================================================ "
python amt/train.py --device ${device} --batch_sz 8 --gradient_accumulation_steps 40 --model AMT \
 --savedir save/AMT/${dataset}/${lb}/ --name model \
 --data_path amt/data/ --dataset ${dataset} \
 --task_type ${task_type} \
 --label ${lb} --output_mode avg \
 --num_image_embeds 3 \
 --n_workers 12 \
 --patience 5 --dropout 0.1 --lr $lr --warmup 0.1 --max_epochs 100 --seed 0
done