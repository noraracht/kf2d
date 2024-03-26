
# kf2d
<!-- k-mer frequency to distance-->

Combination function to perform backbone preprocessing and training classifier and distance models 
------------
It's a wraper function that consequtively runs computation of k-mer frequences for backbone sequences, splits backbone tree into subtrees and produce corresponding true distance matrices as well as trains classifier and distance models. 
```
 python main.py build_library -input_dir $INPUT_DIR -output_dir $OUTPUT_DIR -size 800 -tree $INPUT_PHYLOGENY -mode subtrees_only -cl_epochs 1 -di_epochs 1
```
###### Input: 
**$INPUT_DIR** is an input directory that should contain genome sequences in .fastq/.fq/.fa/.fna/.fasta format. Optional parameter is **-k** which is a k-mer length, set to 7 by default. This command requires [Jellyfish](https://github.com/gmarcais/Jellyfish) to be installed as a dependancy. Optional parameter is **-p** corresponds to number of processors that Jellyfish can utilize to preprocess input sequences.

**$INPUT_PHYLOGENY** is an input backbone phylogenetic tree in .newick/.nwk format that should be split into multiple smaller subtrees. **-size** parameteter is the user spacified subtree size. We set -size default to 850 but in practice we recommend user to define it. **-mode** parameter can take values full_only, hybrid (default), subtrees_only and specifies whether distance matrices should be computed only for a full backbone tree, subtrees or both. This command requires [TreeCluster](https://github.com/niemasd/TreeCluster) to be installed as a dependancy.

Next set of optional parameters are dealing with conditions for training classifier model. These parametrs are equivalent to parameters used by `train_classifier` function. Thus **-cl_epochs** specifies maximum number of training epochs (default is 2000), **-cl_batch_sz** identifies batch size (default values is 16), **-cl_lr**, **-cl_lr_min** and **-cl_lr_decay** refer to starting learning rate, minimum allowed learning rate and learning rate decay values. We suggest to keep learning rate paramaters at their default values unless user has a specific need to modify them.

Final set of optional parameters are related to conditions for training distance model. These parametrs are equivalent to parameters used by `train_model_set` function. Thus **-di_epochs** specifies maximum number of training epochs (default is 8000), **-di_hidden_size** is a dimension of hidden layer in the model, **-di_batch_sz** identifies batch size (default values is 16), **-di_lr**, **-di_lr_min** and **-di_lr_decay** refer to starting learning rate, minimum allowed learning rate and learning rate decay values. We suggest to keep learning rate paramaters at their default values unless user has a specific need to modify them.
###### Output: 
All output files from this command are stored in $OUTPUT_DIR. 

This command generates normalized k-mer frequencies for every entry in the $INPUT_DIR. For every entry it outputs corresponding single file (comma delimited) with extention `.kf`. Next this command will compute subtrees (file with extension `.subtrees` that lists every leaf of a phylogeny and its corresponding subtree number) and corresponding true distance matrices (files named `$PREFIX_subtree_$SUBTREE_NUM.di_mtrx`). Output includes a classifier model called `classifier_model.ckpt` and distance models for every subtree.


Extracting k-mer frequencies
------------
To obtain k-mer frequencies for backbone species and a query set the user can execute the get_frequencies command:
```
 python main.py get_frequencies -input_dir $INPUT_DIR
```
###### Input: 
$INPUT_DIR is an input directory that should contain genome sequences in .fastq/.fq/.fa/.fna/.fasta format. Optional parameter is -k which is a k-mer length, set to 7 by default.
This command requires [Jellyfish](https://github.com/gmarcais/Jellyfish) to be installed as a dependancy.
###### Output: 
This command generates normalized k-mer frequencies for every entry in the $INPUT_DIR. For every entry it outputs corresponding single file (comma delimited) with extention `.kf`.

Split phylogeny into subtrees 
------------
To generate subtrees for a phylogeny with number of leaves > 4000 the user can execure divide_tree command:
```
 python main.py divide_tree -size $SUBTREE_SIZE -tree $INPUT_PHYLOGENY
```
###### Input: 
$INPUT_PHYLOGENY is an input phylogenetic tree in .newick/.nwk format that should be split into multiple smaller subtrees. -size parameteter is the user spacified subtree size. We set -size default to 850 but in practice we recommend user to define it. 
This command requires [TreeCluster](https://github.com/niemasd/TreeCluster) to be installed as a dependancy.
###### Output: 
Output is a text file (extension `.subtrees` that lists every leaf of a phylogeny and its corresponding subtree number.

Training a subtree classifier model
------------
To train a classifier model user can use the following command:
```
 python main.py train_classifier -input_dir $INPUT_DIR -subtrees $FILE.subtrees -e 2000 -o $OUTPUT_DIR
```
###### Input: 
$INPUT_DIR is an input directory that should contain k-mer frequency count file for backbone species in .kf format (output of get_frequencies command). $FILE.subtrees is the file where each input genome has an assigned subtree number. -e number of epochs (default is 4000). $OUTPUT_DIR is the directory where classifier model will be saved once training is complete.
###### Output: 
Output is a classifier model called `classifier_model.ckpt` stored in a user definied repository.

Classification of queries into subtrees
------------
Command to classify query sequences into subtrees:
```
 python main.py classify -input_dir $INPUT_DIR -model $MODEL_DIR  -o $OUTPUT_DIR
```
###### Input: 
$INPUT_DIR is an input directory that should contain k-mer frequency count file for query species in .kf format (output of get_frequencies command). $MODEL_DIR is the folder where model named `classifier_model.ckpt` is located. $OUTPUT_DIR is the directory where `classes.out` will be stored. 
###### Output: 
Output is `classes.out` tab delimited file stored in a user definied repository. File contains information about each query sequence, assigned subtree number and probability values for top as well as all other classes.

Ground truth distance matrix computation 
------------
To compute distance matrix for backbone phylogeny:
```
python main.py get_distances -tree $INPUT_PHYLOGENY  -subtrees $FILE.subtrees -mode [hybrid or subtrees_only]
```
###### Input: 
$INPUT_PHYLOGENY is an input phylogenetic tree in .newick/.nwk format. $FILE.subtrees is the file where each input genome has an assigned subtree number. 
###### Output: 
Output is will be saved in a directory where phylogeny is located.

Train models for subtrees
------------
To train:
```
python main.py train_model_set -input_dir $INPUT_DIR  -true_dist $TRUE_DIST_MATRIX_DIR  -subtrees $FILE.subtrees -e 4000 -o $OUTPUT_DIR
```
###### Input: 
$INPUT_DIR is an input directory that should contain k-mer frequency count file for backbone species in .kf format (output of get_frequencies command). $FILE.subtrees is the file where each input genome has an assigned subtree number. -e number of epochs (default is 4000). $OUTPUT_DIR is the directory where `models_subtree_INDEX.ckpt` will be stored. 
###### Output: 
Output is a set of trained models for each input subtree.

Query subtree models
------------
To query models:
```
python main.py query -input_dir $INPUT_DIR  -model $MODEL_DIR  -classes $CLASSES_DIR -o $OUTPUT_DIR
```
###### Input: 
$INPUT_DIR is an input directory that should contain k-mer frequency count file for query species in .kf format (output of get_frequencies command). $MODEL_DIR is the folder where model named `classifier_model.ckpt` is located. $CLASSES_DIR is the directory where `classes.out` is located. $OUTPUT_DIR is the directory where `apples_input_di_mtrx_query_INDEX.csv` will be stored. 
###### Output: 
Output is a query per backbone sequences distance matrix for subtrees.

