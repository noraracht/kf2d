# kf2d
<!-- k-mer frequency to distance-->

Extracting k-mer frequencies
------------
To obtain k-mer frequencies for backbone species and a query set the user can execute the get_frequencies command:
```
 python main.py get_frequencies -input_dir $INPUT_FOLDER
```
###### Input: 
$INPUT_FOLDER should contain genomes sequences in .fastq/.fq/.fa/.fna/.fasta format. Optional parameter is -k which is a k-mer length, set to 7 by default.
This command requires [Jellyfish](https://github.com/gmarcais/Jellyfish) to be installed as a dependancy.
