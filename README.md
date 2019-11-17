# ParClassifier

Simple implementation of experiments for evaluating Paraphrase Detectors/Semantic Similarity Estimators based
on linear classifires applied to vector sentence representations, as published in
[Souza and Sanches, 2018](https://doi.org/10.21814/lm.10.2.286) (an earlier version in english language can be obtained at:
http://www.inf.pucrs.br/linatural/wordpress/wp-content/uploads/2018/09/123450040.pdf)).


## Usage

This code is written for python 3.7. To download, clone this repository:
```bash
git clone https://github.com/marlovss/ParClassifier.git
```
The experiment was divide into two main parts: computing sentence representations (computeRep.py) and evaluate these representations for paraphrase classification and semantic similarity prediction (evaluate.py)

### Computing Sentence Representations

The computation of sentence representation is performed by the script computeRep.py. This script optionally takes as input a configuration file (Example: rep.cfg) which defines experiment parameters.

Usage:
```python3 computeRep.py [rep.cfg]]
```

the configuration file defines paths to the input, as well as the pre-trained models for different word representations (word embeddings, skip-thought model, Elmo model etc) and path for outputs, as well as which representations should be computed in the experiment.
 

#### Obtaining the input data

The input data should be compatible with ASSIN corpus, e.g. [http://nilc.icmc.usp.br/assin/] and [https://sites.google.com/view/assin2/].

#### Word Representations and language models

To compute the representations, the script uses:

   - a serialized pre-trained word embedding model compatible with gensim and memory mapped for fast loading, i.e. a gensim KeyedVectors model (c.f. [https://radimrehurek.com/gensim/] and [https://radimrehurek.com/gensim/models/keyedvectors.html]);
   - a serialized IDF dictionary (dict: vocabulary -> value)
   - (possibly) a serialized Skip-Thoughts models, trained using Daniel Watson's implementation (c.f. [https://github.com/danielwatson6/skip-thoughts]) using the same word embeddings model used for processing the input. (A ST model used in the experiments, trained using Hartmann et al's Word2vwc skipgram model with 300 dim is available at <a href="https://drive.google.com/drive/folders/1HrQqevtT9SaXZGbx0RB6fVFKazR3OWCz?usp=sharing">Here</a>)
   - (possibly) an Elmo model (c.f. https://allennlp.org/elmo)

### Evaluating Sentence Representations

The evaluation of sentence representations for paraphrase classificatio nand semantic similarity estimation is performed by the script evaluate.py and evaluation metric are printed on the terminal. 

Usage:
```python3 evaluate.py [--input path_to_vector] [--train root_filename] [--test root_filename] [--oversample method] [--total True/False] [--sim True/False]
```
The parameters define:
   - input (default: "data/vectors") : Path to the directory containing the text files
   - train (default: "train"): Path to the directory containing the train data files
   - test (default: "test"):Path to the directory containing the test data files
   - oversample (default: "none"): oversampling method to be used (possible "none","random", "smote" and "adasyn")
   - total (default: False): Whether the classifier should test a dataset with combined representations (memory expensive and low performance).
   - sim (default: False):If True evaluates Sentence Similarity Estimation, otherwise evaluates Paraphrase Classification

## Dependencies

All the dependencies are listed in the `requirements.txt` file. They can be installed with `pip` as follows:
```bash
pip3 install -r requirements.txt
```
