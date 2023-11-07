# kf2d
<!-- k-mer frequency to distance-->

Extracting k-mer frequencies
------------
To obtain k-mer frequencies for backbone species and a query set the user can execute the get_frequencies command:
```
 python main.py get_frequencies -input_dir $INPUT_DIR
```
###### Input: 
$INPUT_DIR is aan input directory that should contain genome sequences in .fastq/.fq/.fa/.fna/.fasta format. Optional parameter is -k which is a k-mer length, set to 7 by default.
This command requires [Jellyfish](https://github.com/gmarcais/Jellyfish) to be installed as a dependancy.
###### Output: 
This command generates normalized k-mer frequencies for every entry in the $INPUT_DIR. For every entry it outputs corresponding single file (comma delimited) with extention `.kf`.

Split phylogeny into subtrees 
------------
To generate subtrees for a phylogeny with number of leaves > 4000 the user can execure divide_tree command:
```
 python main.py divide_tree -size $SUBTREE_SIZE -tree tree $INPUT_PHYLOGENY
```
###### Input: 
$INPUT_PHYLOGENY is an input phylogenetic tree in .newick/.nwk format that should be split into multiple smaller subtrees. -size parameteter is the user spacified subtree size. We set -size default to 850 but in practice we recommend user to define it. 
This command requires [TreeCluster](https://github.com/niemasd/TreeCluster) to be installed as a dependancy.
###### Output: 
Output is a text file (extension `.subtrees` that lists every leaf of a phylpgeny and its corresponding subtree number.
