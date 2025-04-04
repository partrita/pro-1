pip install kaggle

## insert your kaggle api key here

kaggle competitions download -c cafa-5-protein-function-prediction

apt-get update && apt-get install unzip
unzip cafa-5-protein-function-prediction.zip
rm cafa-5-protein-function-prediction.zip
mv "Test (Targets)" Test
mv Test Train IA.txt sample_submission.tsv function/cafa/