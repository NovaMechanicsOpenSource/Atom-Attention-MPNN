# Atom-Attention-MPNN
## Pretraining:

To pretrain the model using ZINC15 database:
python pretraining.py --data_path zinc15_250K_2D.csv --atom_attention --normalize_matrices --no_features_scaling --save_dir save_dir_pretrain/ --features_generator morgan_count

To train the model using blood-brain barrier dataset:
