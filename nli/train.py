import torch
import pytorch_lightning as pl
import argparse
import pickle

from data import NLIDataModule
from setup import setup_vocab, setup_model, find_checkpoint


def parse_option():
    parser = argparse.ArgumentParser(description="Training NLI models")

    parser.add_argument('--model_type', type=str, default='uni_lstm', help='Model type', choices=['avg_word_emb', 'uni_lstm', 'bi_lstm', 'max_pool_lstm'])
    parser.add_argument('--feature_type', type=str, default = 'baseline', help='Type of features to use', choices=['baseline', 'multiplication'])
    
    parser.add_argument('--ckpt_path', type=str, default = None, help='Path to save checkpoint, if None is given, it will be the same as model_type')
    parser.add_argument('--version', default='version_0', help='Version of the model to load')

    parser.add_argument('--path_to_vocab', type=str, default='store/vocab.pkl', help='Path to vocab')

    parser.add_argument('--epochs', type=int, default=20, help='Max number of training epochs')
    parser.add_argument('--num_workers', type=int, default=3, help='Number of workers for dataloader')

    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    return args

def main(args):
    pl.seed_everything(args.seed, workers=True)


    args.ckpt_path = args.ckpt_path if args.ckpt_path is not None else args.model_type
    ckpt_path, _ = find_checkpoint(args.ckpt_path, args.version)

    vocab    = setup_vocab(args.path_to_vocab)
    model, _ = setup_model(args.model_type, vocab, ckpt_path, args.feature_type)
    datamodule = NLIDataModule(vocab=vocab, batch_size=64, num_workers=args.num_workers)

    trainer = pl.Trainer(
        logger = pl.loggers.TensorBoardLogger('logs', name=args.ckpt_path, version=args.version),
        max_epochs = args.epochs, 
        log_every_n_steps = 1, 
        accelerator = 'gpu' if torch.cuda.is_available() else 'cpu',
        callbacks = [
            # pl.callbacks.ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc"),
            pl.callbacks.TQDMProgressBar(refresh_rate = 1000)
            ],
        deterministic = True,
    )
    trainer.fit(model,  datamodule=datamodule)
    trainer.test(model, datamodule=datamodule)

if __name__ == "__main__":
    args = parse_option()
    print(args)
    main(args)