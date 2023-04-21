# Analysis of NLI model confidence & sentence embedding enhancements

This is a reprodicibilty report of [_Supervised Learning of Universal Sentence Representations from Natural Language Inference Data_ by Conneau _et al._ (2017)](https://arxiv.org/abs/1705.02364), where we reproduce and analyze the main findings. Furthermore, we analyze why certain models perform better than others by looking at the distribution and confidence of the predictions. Finally, we explore how sentence embeddings can be enhanced by multiplying them with trainable weights.



<p float="left" align="middle">
 <!---<img align="middle" src="/figures/vocabulary_size.png" width="200" />--->  
  <img align="middle" src="/Extras/Files/Pasted image 20230421112437.png" width="300" /> 
  <img align="middle" src="Pasted image 20230421112416.png.png" width="500" />
</p>


![[Pasted image 20230421112437.png]]
![[Pasted image 20230421112422.png]]
![[Pasted image 20230421112416.png]]
## Installation instructions
Install and activate the conda environment.
```
conda env create -f env_nli.yml
conda activate nli
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
Comment about `cp ... senteval/utils.py`: Because we pass some extra arguments to the `batcher` function in SentEval, we have to comment out some check. That is, line `89 - 93` from `/SentEval/senteval/utils.py` are commented out.

Downloading and preprocessing SNLI, downloading GloVe and creating the vocabulary can all be done at once 
```
python nli/preprocess.py --download_snli --download_glove --create_vocab
```

## Minimal Working Example
To directly train, evaluate and store the results,where `avg_word_emb` is one of the four models  `[avg_word_emb, uni_lstm, bi_lstm, max_pool_lstm]`:
```
python nli/train.py   --model_type avg_word_emb
python nli/eval.py    --model_type avg_word_emb
python nli/results.py --model_type avg_word_emb
```

One can inspect the training on TensorBoard
```
tensorboard --logdir logs
```




## Code structure
- `data/` default directory where GloVe and SNLI are saved
	- `examples_snli.json` example sentences that are discussed in `analysis.ipynb`
- `figs/` default directory where figures and images can be saved
- `jobs/` scripts to send jobs to LISA
	- `slurm_output/` SLURM output files of the jobs
- `logs/` trained models and related data such as checkpoint files, Tensorboard, calculated accuracies. Can be downloaded here
- `nli/` source code that trains on SNLI, evaluates on SentEval and calculates these results.
- `SentEval/` cloned repository from FAIR
- `store/` directory to store intermediate files, i.e. the vocabulary


python nli/train.py   --model_type avg_word_emb --epochs 2