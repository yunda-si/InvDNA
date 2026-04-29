# InvDNA: An End-to-End Deep Learning Method for ssDNA Sequence Design

InvDNA is a deep learning-based framework that designs single-stranded DNA (ssDNA) sequences. It takes as input the backbone atom coordinates (including 'OP1', 'OP2', 'C2', 'C4', 'C5', 'O3', 'O4', 'O5', 'P', 'C1', 'C3') along with masked sequences, and outputs the corresponding ssDNA sequences along with their all-atom structures.

## Requirements

The following Python packages are required:

* `Pytorch`
* `biopython`
* `numpy`
* `ml_collections`

You can download the pre-trained model weights from Google Drive: [Download Weights](https://drive.google.com/drive/folders/1RnKpKtaqu0QcphJeTctckxpsVmujR20A?usp=drive_link), and place them in the `weights` folder.

## Input Parameters

* **seq_file**: Path to a file containing the input ssDNA sequence, using only the characters 'A', 'T', 'C', 'G', and 'X'.

* **pdb_file**: Path to a file containing the backbone atom coordinates for InvDNA.

* **save_path**: Directory where the output (designed sequences and all-atom structures) will be saved.

* **weight_file**: Path to the pre-trained model file. Choose from:

  * `base_struc.pt`: Used to reconstruct all-atom structure from the backbone and sequence.
  * `full_seq.pt`: Used to design ssDNA sequences from the backbone.

* **noise_sd**: Standard deviation of the noise to be added to the input backbone. This introduces variability to generate diverse sequences.

* **sample_atoms**: A flag (True/False) to determine whether atoms in the backbone should be randomly sampled (i.e., deleted) to create more diverse sequences.

## Example Usage

```bash
python -u prediction.py -weight ./weights/full_seq.pt -seq_file ./example/test.fasta -pdb_file ./example/test.pdb -save_path ./example
```

## Contact & Support

If you encounter any issues, please feel free to open an issue or contact us at [yunda_si@ucas.edu.cn](mailto:yunda_si@ucas.edu.cn).

## Citation

Si, Y., Xu, Y., & Chen, L. (2025). End-to-end single-stranded DNA sequence design with all-atom structure reconstruction. *bioRxiv*, 2025.12.05.692525. [DOI: 10.64898/2025.12.05.692525](https://doi.org/10.64898/2025.12.05.692525)
