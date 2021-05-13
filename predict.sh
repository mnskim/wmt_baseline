#model_name=Helsinki-NLP/opus-mt-en-fr
model_name=../use_terms/checkpoint-2099

# Use terms
data_path='data_2_terminology'
output_dir='use_terms'


# Don't use terms
#data_path='data_2'
#output_dir='no_terms'

batch_size=32
eval_batch_size=64
max_length=128

n_epochs=10

CUDA_VISIBLE_DEVICES=2 python baseline_hf.py --model_name_or_path $model_name \
                                             --output_dir $output_dir \
                                             --data_path $data_path \
                                             --do_predict \
                                             --load_best_model_at_end \
                                             --metric_for_best_model 'bleu' \
                                             --greater_is_better true \
                                             --predict_with_generate \
                                             --fp16 \
                                             --evaluation_strategy "epoch" \
                                             --learning_rate 2e-5 \
                                             --per_device_train_batch_size $batch_size \
                                             --per_device_eval_batch_size $eval_batch_size \
                                             --max_source_length $max_length \
                                             --max_target_length $max_length \
                                             --weight_decay=0.01 \
                                             --save_total_limit 3 \
                                             --num_train_epochs $n_epochs \
                                             #--max_train_samples 1000
