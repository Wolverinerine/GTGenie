# GTGenie
Prediction of biomarker–disease associations based on graph attention network and text representation
# Data description
We use HMDAD as example and other datasets files structure are similar.  
HMDAD/microbes: ID and names for microbes.  
HMDAD/diseases: ID and names for diseases.  
HMDAD/adj: interaction pairs between microbes and diseases.  
HMDAD/interaction: known microbe-disease interaction matrix.  
HMDAD/D_SSM: disease semantic similarity.  
HMDAD/M_SSM: microbe semantic similarity.  
HMDAD/all_text: text description about disease-microbe associations.  
HMDAD/microbe_to_taxon: the taxon of microbes which is used to create microbe semantic similarity.  
# Text_encoding description
get_text_embedding.py: get text features accroding to 5-fold cross-validation.
# Run steps
1. Download biobert_v1.1 from (https://huggingface.co/dmis-lab/biobert-v1.1/tree/main) --- files "pytorch_model.bin", "config.json", and "vocab.txt" are required.
2. Generate text embedding using all_text.csv for each dataset.
3. Run main.py to train the model using 5-fold cross-validation.
# Requirements
* Pytorch 1.8.1
* tensorflow 1.15
* transformers
* sentencepiece
* numpy
* scipy
* sklearn
* xlrd 1.2.1
* tensorflow-determinism-0.3.0
