# Atom-Attention-MPNN
## Pretraining:

To pretrain the model using ZINC15 database:
python pretraining.py --data_path zinc15_250K_2D.csv --atom_attention --normalize_matrices --no_features_scaling --save_dir save_dir_pretrain/ --features_generator morgan_count

## Training:

To train the model using blood-brain barrier dataset using the pretrained Atom-Attention MPNN:
python train.py --data_path data/B3DB_class_data.csv --atom_attention --normalize_matrices --no_features_scaling --features_generator morgan_count --pretrained_checkpoint save_dir_pretrain/pretrained.pt --save_dir bbb_pretrained/ --extra_metrics accuracy precision recall specificity

If we don't want to use the pretrained model:
python train.py --data_path data/B3DB_class_data.csv --atom_attention --normalize_matrices --no_features_scaling --features_generator morgan_count --save_dir bbb_nopretrained/ --extra_metrics accuracy precision recall specificity
