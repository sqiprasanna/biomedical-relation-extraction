The following scripts are tested on Linux system.

# Download and process bLURB datasets
1. Install requirements 
- system requirements
```
python >= 3.6
pip
git
unrar
```

- python requirements
```
pip install -r requirements.txt
```

2. Download data
```
sh download_BLURB_data.sh
```
Before proceeding to the next step:
you will need to register and manually download [ChemProt](https://biocreative.bioinformatics.udel.edu/news/corpora/) and [BioASQ](http://participants-area.bioasq.org/datasets/) from the official site. For BioASQ, we use task b BioASQ7 for both train and test data. 

3. Once you have downloaded chemprot and BioASQ, put them under raw_data/.e.g. 
raw_data/ChemProt_Corpus.zip
raw_data/BioASQ-training7b.zip
raw_data/Task7BGoldenEnriched.zip

4. Process data
```
sh preprocess_BLURB_data.sh
```

## License

See the [LICENSE](LICENSE.md) file for license rights and limitations (MIT).