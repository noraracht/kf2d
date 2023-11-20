# kf2d
<!-- k-mer frequency to distance-->

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
Output is a text file (extension `.subtrees` that lists every leaf of a phylpgeny and its corresponding subtree number.

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
