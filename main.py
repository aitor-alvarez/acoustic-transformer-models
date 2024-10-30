from argparse import ArgumentParser
import torch
from lightning import Trainer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
from model import AcousticTransformer
from transformers import AutoConfig
from datasets import load_dataset
from data import AudioDataset


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--num_epochs', type=int)
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--n_gpus', type=int)
    parser.add_argument('--n_nodes', type=int)
    parser.add_argument('--strategy', type=str)
    args = parser.parse_args()

    if args.data_dir and args.model_name:
        dataset = load_dataset("audiofolder", data_dir=args.data_dir, drop_labels=False)
        labels = dataset["train"].features["label"].names
        label2id, id2label = dict(), dict()
        for i, label in enumerate(labels):
            label2id[label] = str(i)
            id2label[str(i)] = label
        num_labels = len(id2label)
        config = AutoConfig.from_pretrained(args.model_name,
                                num_labels=num_labels, label2id=label2id, id2label=id2label)


        model = AcousticTransformer(config)

        model = torch.compile(model)

        early_stopping = EarlyStopping(monitor="Validation Accuracy", min_delta=0.00, patience=3, verbose=False,
                                            mode="max")

        checkpoint_callback = ModelCheckpoint(dirpath="checkpoints/", save_top_k=2, monitor="Validation Accuracy",
                                              mode="max")

        data = AudioDataset(model_name=args.model_name, batch_size=args.batch_size,
                                dataset=dataset)

        logger = WandbLogger(
            project="acoustic_transformer",
            log_model=False,
            offline = True,
            save_dir="./wandb",
        )
        if args.n_gpus and args.n_nodes:
            if args.n_gpus > 1:
                trainer = Trainer(max_epochs=args.num_epochs, logger=logger, accelerator='cuda', accumulate_grad_batches=2,
                              strategy=args.strategy, devices=args.n_gpus, num_nodes=args.n_nodes, log_every_n_steps=10, precision=16,
                                  callbacks=[early_stopping, checkpoint_callback])
            elif args.n_gpus < 2:
                trainer = Trainer(max_epochs=args.num_epochs, logger=logger, accelerator='cuda', accumulate_grad_batches=2,
                              devices=args.n_gpus, num_nodes=args.n_nodes, log_every_n_steps=10, precision=16,
                                  callbacks=[early_stopping, checkpoint_callback])
        else:
            #For testing on Mac prior to SLURM set accelerator="mps". If mps is not available change accelerator="cpu"
            trainer = Trainer(max_epochs=args.num_epochs, logger=logger, accumulate_grad_batches=2,
                              accelerator="cpu", devices="auto", log_every_n_steps=10, precision=16,
                              callbacks=[early_stopping, checkpoint_callback])


        data.setup()
        trainer.fit(model, datamodule=data)
        trainer.test(model, datamodule=data)
        print("training process completed")

    else:
        print("Please, provide model name and data directory.")