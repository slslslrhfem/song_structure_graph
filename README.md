# Combinatorial Music generation model with song structure graph analysis


For training from scratch..

1. download lpd_17 dataset in https://salu133445.github.io/lakh-pianoroll-dataset/dataset, make root as pattern_representation/lpd_17...

2. download lmd_full dataset in https://colinraffel.com/projects/lmd/, make root as pattern_representation/dataset/lmd_full...

3. run following commands..
   
```
    python main.py dataset_conformity
    python main.py process_CBIR
    python main.py train_CBIR
    python main.py process_pattern
    python main.py train_AE
    python main.py preprocess_npz
    python main.py train_graph2vec
    python main.py preprocess_unet
    python main.py train_unet
    python main.py generate_music
```

simple explanation for each part is in main.py. note that you should change checkpoint loading in some codes..!
(especially, preprocess_npz, preprocess_unet, generate_music...)

Note that folder "datasets" from various site, to get genre label for musics.

This github now on updating..
