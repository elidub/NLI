## Folder structure
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


## Scripts
Here, the most important flags and their use cases for the scripts are being discussed

- `preprocess.py`
	- `--download_snli` Download and pre-process the SNLI dataset.
	- `--download_glove` Download the GloVe dataset.
	- `--create_vocab` Create the vocabulary.
- `train.py`
	- `--model_type` Chose one of the model types `[avg_word_emb, uni_lstm, bi_lstm, max_pool_lstm]`.
	- `--feature_type` Chose the feature type `[baseline, multiplication]` for sentence embedding enhancement. Default is `baseline`. 
	- `--ckpt_path` Give a path where the model will saved inside `logs/`. Default is set to be the same as `model_type`.
	- `--version` Give a subdirectory path where the model will be saved inside `logs/ckpt_path/`. Default is `version_0`
- `eval.py`
	- `--model_type, --ckpt_path, --version` See `train.py`.
- `results.py`
	- `--model_type, --feature_type, --ckpt_path, --version` See `train.py`.
	- `--transfers_results` When this flag is given, the SentEval transfer results wil _not_ be calculated.
	- `--nli_results` When this flag is given, the NLI task results wil _not_ be calculated.

## Pretrained models
Pretrained models can be downloaded, see the main README for a link. This folder should be directory `logs/`. To work with them, know that:
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
