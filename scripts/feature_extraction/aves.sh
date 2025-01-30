python src/extract_aves.py -m experiment=extract_aves-bio/abzaliev.yaml,extract_aves-bio/watkins.yaml,extract_aves-bio/imv.yaml,extract_aves-bio/alex.yaml,extract_aves-bio/kaja.yaml +slurm=gpu

python src/compile_feats.py -m experiment=compile_features/aves-bio_abzaliev.yaml,compile_features/aves-bio_watkins.yaml,compile_features/aves-bio_alex.yaml,compile_features/aves-bio_imv.yaml,compile_features/aves-bio_kaja.yaml +slurm=gpu
