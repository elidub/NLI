# Analysis of NLI model confidence & sentence embedding enhancements

This is a reprodicibilty report of [_Supervised Learning of Universal Sentence Representations from Natural Language Inference Data_ by Conneau _et al._ (2017)](https://arxiv.org/abs/1705.02364), where we reproduce and analyze the main findings. Various models are trained on a Natural Language Inference (NLI) task. These trained models function as encoders for sentence embeddings for [_SentEval_ by Conneau and Kiela (2018)](https://arxiv.org/abs/1803.05449), a benchmark transfer task suite. Furthermore, we analyze why certain models perform better than others by looking at the distribution and confidence of the predictions. Finally, we explore how sentence embeddings can be enhanced by multiplying them with trainable parameter.

<p float="left" align="middle">
  <img align="middle" src="figs/models.png" height="250" /> 
  <img align="middle" src="figs/confs.png" height="250" /> 
  <br>
  <img align="middle" src="figs/features.png" height="200" />
</p>

**Top left**. The four models that are implemented for the NLI task. **Top right.** Example of plots that are used to analyze the confidence and why certain models fail. **Bottom.** Each sentence embedding is enhanced by multiplying it with a trainable parameters. The multiplier behavior during training is shown.

## Installation instructions
Install and activate the conda environment.
```
conda env create -f env_nli.yml
conda activate nli2
```

Install SentEval inside the NLI repository
```
# Clone repo from FAIR GitHub
git clone https://github.com/facebookresearch/SentEval.git
cd SentEval/
cp ../nli/senteval_utils.py senteval/utils.py # see comment below

# Install sentevall and download transfer datasets
python setup.py install 
cd data/downstream/
./get_transfer_data.bash
cd ../../.. # Go back to main dir
```
Comment about `cp ... senteval/utils.py`: Because we pass some extra arguments to the `batcher` function in SentEval, we have to comment out a check that doesn't allow custom arguments being passed. That is, line `89 - 93` from `/SentEval/senteval/utils.py` are commented out.

Downloading and preprocessing SNLI, downloading GloVe and creating the vocabulary can all be done at once. The vocabulary can also be downloaded [here](https://drive.google.com/file/d/1syMGFLZimX5SBFVh3bxpRiGdVV9Bc8q6/view?usp=sharing), in that case flag `--create_vocab` can be omitted. Default directory is `store/vocab.pkl`. 
```
python nli/preprocess.py --download_snli --download_glove --create_vocab
```

## Code structure
- `data/` default directory where GloVe and SNLI are saved
	- `examples_snli.json` example sentences that are discussed in `analysis.ipynb`
- `figs/` default directory where figures and images can be saved
- `jobs/` scripts to send jobs to LISA
	- `slurm_output/` SLURM output files of the jobs
- `logs/` To be downloaded [here](https://drive.google.com/file/d/1sttjLJdJ6hFLF_si3Fbz6wDyVDccpEMv/view?usp=sharing) (zip-file). Contains pretrained models and related data such as checkpoint files, Tensorboard and calculated accuracies. 
- `nli/` source code that trains on SNLI, evaluates on SentEval and calculates these results. See the table [below](#nli-structure) for detailed overview.
- `SentEval/` to be cloned repository from FAIR.
- `store/` directory to store intermediate files, i.e. the vocabulary.
- `analysis.ipynb` Notebook that explains problem, shows and discusses results, error and confidence analysis, sentence embedding enhancements. 

### `nli/` structure
| File                | Description                                                                                                                                                  |
| ------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `data.py`           | Vocabulary, custom PyTorch dataset and datamodule.                                                                                                           |
| `eval.py`           | Script to run transfers tasks of SentEval.                                                                                                                   |
| `learner.py`        | PyTorch Lightning module that handles training.                                                                                                              |
| `models.py`         | All four models (AvgWordEmv, UniLSTM, BiLSTM-last, BiLSTM-Max), MLP classifier, sentence embedding concatenation and a network that consolidate all of that. |
| `plot.py`           | Plot classes used in `analysis.ipynb` that read the results calculated by `results.py`.                                                                      |
| `preprocess.py`     | Script that downloads and prepossesses SNLI, downloads GloVe and creates vocabulary.                                                                         |
| `results.py`        | Script that reads data made by `train.py` and `eval.py`, calculates accuracies and stores them.                                                              |
| `senteval_utils.py` | File to replace `SentEval/senteval/utils.py`, see installation instructions.                                                                                 |
| `setup.py`          | Utilities to setup and load the vocabulary, model and similar objects.                                                                                       |
| `train.py`          | Script that trains a given model                                                                                                                             |


## Run instructions
### Quick Start
To directly train, evaluate and store the results, where `avg_word_emb` is one of the four models in `[avg_word_emb, uni_lstm, bi_lstm, max_pool_lstm]`, run the commands below. However, with  `logs/` downloaded from the link [above](#code-structure), one can directly analyze the results in `analysis.ipynb`. One can inspect the training on TensorBoard with `tensorboard --logdir logs`.
```
python nli/train.py   --model_type avg_word_emb
python nli/eval.py    --model_type avg_word_emb
python nli/results.py --model_type avg_word_emb
```

### Scripts
Here, the most important flags and their use cases for the scripts are being discussed

- `nli/preprocess.py`
	- `--download_snli` Download and pre-process the SNLI dataset.
	- `--download_glove` Download the GloVe dataset.
	- `--create_vocab` Create the vocabulary.
- `nli/train.py`
	- `--model_type` Chose one of the model types `[avg_word_emb, uni_lstm, bi_lstm, max_pool_lstm]`.
	- `--feature_type` Chose the feature type `[baseline, multiplication]` for sentence embedding enhancement. Default is `baseline`. 
	- `--ckpt_path` Give a path where the model will saved inside `logs/`. Default is set to be the same as `model_type`.
	- `--version` Give a subdirectory path where the model will be saved inside `logs/ckpt_path/`. Default is `version_0`
- `nli/eval.py`
	- `--model_type, --ckpt_path, --version` See `nli/train.py`.
- `nli/results.py`
	- `--model_type, --feature_type, --ckpt_path, --version` See `nli/train.py`.
	- `--transfers_results` When this flag is given, the SentEval transfer results wil _not_ be calculated.
	- `--nli_results` When this flag is given, the NLI task results wil _not_ be calculated.

### Pretrained models
Pretrained models can be downloaded, see the link [above](#code-structure). This folder should be directory `logs/`. To work with them, know that:
- All models have `--version version_0`
- All `--feature_type baseline` models have default `ckpt_path`, i.e. the same as its `model_type`.
- All `--feature_type multiplication` have `--ckpt_path <MODEL_TYPE>_mult`.

To clarify the above, one can retrain, evaluate and calculate the results  of the pretrained models with a given model by:
```
# Unenhanced 'baseline' features
python nli/train.py   --model_type <MODEL_TYPE>
python nli/eval.py    --model_type <MODEL_TYPE>
python nli/results.py --model_type <MODEL_TYPE>

# Enhanced 'multiplication' features
python nli/train.py   --model_type <MODEL_TYPE> --feature_type multiplication
python nli/results.py --model_type <MODEL_TYPE> --feature_type multiplication \
	--ckpt_path <MODEL_TYPE>_mult --transfer_results
```


## Contact
If you have questions or found a bug, send a email to [eliasdubbeldam@gmail.com](mailto:eliasdubbeldam@gmail.com)
