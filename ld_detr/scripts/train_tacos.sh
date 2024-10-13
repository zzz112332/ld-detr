dset_name=tacos
ctx_mode=video_tef
v_feat_types=slowfast_clip
t_feat_type=clip
results_root=results_tacos
exp_id=exp

######## data paths
train_path=data/tacos/train.jsonl
eval_path=data/tacos/test.jsonl
eval_split_name=val

######## setup video+text features
feat_root=~/PycharmProjects/zpc_projects/features/tacos

# video features
v_feat_dim=0
v_feat_dirs=()
if [[ ${v_feat_types} == *"slowfast"* ]]; then
  v_feat_dirs+=(${feat_root}/slowfast_features)
  (( v_feat_dim += 2304 ))  # double brackets for arithmetic op, no need to use ${v_feat_dim}
fi
if [[ ${v_feat_types} == *"clip"* ]]; then
  v_feat_dirs+=(${feat_root}/clip_features)
  (( v_feat_dim += 512 ))
fi

# text features
if [[ ${t_feat_type} == "clip" ]]; then
  t_feat_dir=${feat_root}/clip_text_features/
  t_feat_dim=512
else
  echo "Wrong arg for t_feat_type."
  exit 1
fi

#### training
n_epoch=250
distillation_coefficient=0.7
max_v_l=-1
lw_saliency=4
seed=$((RANDOM << 15 | RANDOM))

PYTHONPATH=$PYTHONPATH:. python ld_detr/train.py \
--dset_name ${dset_name} \
--ctx_mode ${ctx_mode} \
--train_path ${train_path} \
--eval_path ${eval_path} \
--eval_split_name ${eval_split_name} \
--v_feat_dirs ${v_feat_dirs[@]} \
--v_feat_dim ${v_feat_dim} \
--t_feat_dir ${t_feat_dir} \
--t_feat_dim ${t_feat_dim} \
--results_root ${results_root} \
--exp_id ${exp_id} \
--max_v_l ${max_v_l} \
--n_epoch ${n_epoch} \
--lw_saliency ${lw_saliency} \
--distillation_coefficient ${distillation_coefficient} \
--seed ${seed} \
${@:1}
