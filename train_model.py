#!/usr/bin/env python
#  coding=utf-8

"""
Script to train a model
"""

import argparse
import logging

import numpy as np
import rdkit
import rdkit.Chem as rc
import torch
import tqdm

import models.reinvent.dataset as md
import models.reinvent.model as mm
import models.reinvent.utils as mu


rdkit.rdBase.DisableLog("rdApp.error")


class TrainModelRunner:
    """Trains a given model."""
    EPSILON = 0.01

    def __init__(self, input_model_path, output_model_prefix_path, input_smiles_path, log_file_path="train.log",
                 save_every_n_epochs=0, batch_size=128, learning_rate=0.01, num_epochs=10, starting_epoch=0, patience=1000000, temperature=1.0, 
                 no_shuffle_each_epoch=False, learning_rate_change_gamma=0.1, learning_rate_change_step=1):
        """
        Creates a TrainModelRunner.
        :param input_model_path: The input model path.
        :param output_model_prefix_path: Prefix path to the trained models.
        :param input_smiles_path: Smiles file with the training set.
        :param log_file_path: Path to store the log CSV file.
        :param save_every_n_epochs: Save the trained model every n epochs appending the epoch in the end (do not save until the end = 0).
        :param batch_size: Batch size (beware GPU memory usage).
        :param learning_rate: Learning rate.
        :param num_epochs: Number of epochs to train.
        :param starting_epoch: Starting epoch (resume training)
        :param patience: Number of steps where the training get stopped if no loss improvement is noticed.
        :param temperature: Temperature for the sequence sampling. Has to be larger than 0.
                            Values below 1 make the RNN more confident in it's generation, but also more conservative.
                            Values larger than 1 result in more random sequences.
        :param no_shuffle_each_epoch: Don't shuffle the training set after each epoch.
        :param learning_rate_change_gamma: Ratio which the learning change is changed.
        :param learning_rate_change_step: Number of epochs until the lr is changed.
        :return:
        """
        self._model = mm.Model.load_from_file(input_model_path)

        self._output_model_prefix_path = output_model_prefix_path
        self._save_every_n_epochs = save_every_n_epochs

        self._training_set = mu.read_smi_file(input_smiles_path)
        self._learning_rate = learning_rate
        self._epochs = num_epochs
        self._starting_epoch = starting_epoch
        self._batch_size = batch_size
        self._patience = patience
        self._temperature = temperature
        self._shuffle_each_epoch = not no_shuffle_each_epoch

        self._lr_change_gamma = learning_rate_change_gamma
        self._lr_change_step = learning_rate_change_step

        self._log_file = open(log_file_path, 'w+')

        self._already_run = False

        self._data_loader = None
        self._optimizer = None
        self._lr_scheduler = None

    def run(self):
        """
        Trains the model.
        :return:
        """
        if self._already_run:
            return False

        self._initialize_dataloader()
        self._initialize_optimizer()

        for epoch in range(self._starting_epoch, self._epochs + self._starting_epoch):
            logging.info("Starting EPOCH #%d", epoch)
            if not self._train_epoch(epoch):
                logging.warning("Early leave at EPOCH #%d", epoch)
                break

        self._already_run = True
        return True

    def __del__(self):
        self._log_file.close()

    def _initialize_dataloader(self):
        dataset = md.Dataset.for_model(self._training_set, self._model)
        self._data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=self._batch_size, shuffle=self._shuffle_each_epoch, collate_fn=md.Dataset.collate_fn)

    def _initialize_optimizer(self):
        self._optimizer = torch.optim.Adam(
            self._model.rnn.parameters(), lr=self._learning_rate)
        self._lr_scheduler = torch.optim.lr_scheduler.StepLR(
            self._optimizer, step_size=self._lr_change_step, gamma=self._lr_change_gamma)

    def _train_epoch(self, epoch):
        current_patience = self._patience
        lowest_loss = np.float("inf")
        self._lr_scheduler.step()
        step = 0
        for batch in tqdm.tqdm(self._data_loader, total=len(self._data_loader)):
            input_vectors = batch.long()

            loss_var = self._calculate_loss(input_vectors)
            loss = loss_var.item()

            if loss + self.EPSILON < lowest_loss:
                current_patience = self._patience
                lowest_loss = loss
            else:
                current_patience -= 1

            if current_patience == 0:
                logging.warning("Reached patience = 0")
                self._save_model(epoch)
                return False

            self._optimizer.zero_grad()
            loss_var.backward()
            self._optimizer.step()
            step += 1

        self._log_epoch(epoch, step, loss, current_patience)

        if self._save_every_n_epochs > 0 and (epoch % self._save_every_n_epochs == 0):
            self._save_model(epoch)

        return True

    def _calculate_loss(self, input_vectors):
        log_p = self._model.likelihood(
            input_vectors, temperature=self._temperature)
        return -log_p.mean()

    def _save_model(self, epoch):
        self._model.checkpoint()
        path = "{}.{}".format(self._output_model_prefix_path, epoch)

        self._model.save(path)

    def _log_epoch(self, epoch, step, loss, current_patience):
        smis, _ = self._model.sample_smiles(
            self._batch_size, temperature=self._temperature)
        valid_smis = []
        invalid_smis = []
        for smi in smis:
            if rc.MolFromSmiles(smi):
                valid_smis.append(smi)
            else:
                invalid_smis.append(smi)
        ratio_valid_smiles = len(valid_smis) / len(smis)

        self._log_file.write("{};{};{:.6f};{};{:.6f};{:.2f};Valid|{}|;Invalid|{}|\n".format(
            epoch, step, loss, current_patience,
            self._optimizer.param_groups[0]["lr"],
            ratio_valid_smiles,
            ",".join(valid_smis[:10]),
            ",".join(invalid_smis[:10])
        ))
        self._log_file.flush()


def parse_args():
    """Parses input arguments."""

    parser = argparse.ArgumentParser(
        description="Train a model on a SMILES file.")

    parser.add_argument("--input-model-path", "-i",
                        help="Input model file", type=str, required=True)
    parser.add_argument("--output-model-prefix-path", "-o",
                        help="Prefix to the output model (may have the epoch appended)", type=str, required=True)
    parser.add_argument("--input-smiles-path", "-s",
                        help="Path to the SMILES file", type=str, required=True)
    parser.add_argument("--log-file-path", "-l",
                        help="Path to store the log CSV file [DEFAULT: train.log]", type=str)
    parser.add_argument("--save-every-n-epochs",
                        help="Save the model after n epochs [DEFAULT: 0 (disabled)]", type=int)
    parser.add_argument(
        "--num-epochs", "-e", help="Number of epochs to train [DEFAULT: 10]", type=int)
    parser.add_argument("--starting-epoch",
                        help="Starting epoch [DEFAULT: 0]", type=int)
    parser.add_argument("--no-shuffle-each-epoch", help="Don't shuffle the training set after each epoch.",
                        action="store_true")
    parser.add_argument(
        "--batch-size", help="Number of molecules processed per batch [DEFAULT: 128]", type=int)
    parser.add_argument("--learning-rate", "--lr",
                        help="Learning rate for training [DEFAULT: 0.01]", type=float)
    parser.add_argument(
        "--patience",
        help="Number of steps where the training get stopped if no loss improvement is noticed [DEFAULT: 30000]",
        type=int)
    parser.add_argument("--learning-rate-change-gamma", "--lrcg",
                        help="Ratio which the learning change is changed [DEFAULT: 0.1]", type=float)
    parser.add_argument("--learning-rate-change-step", "--lrcs",
                        help="Number of epochs until the learning rate changes [DEFAULT: 1]", type=int)
    parser.add_argument("--temperature", "-t",
                        help=("Temperature for the sequence sampling. Has to be larger than 0. "
                              "Values below 1 make the RNN more confident in it's generation, "
                              "but also more conservative. Values larger than 1 result in more random sequences. "
                              "[DEFAULT: 1.0]"), type=float)

    return {k: v for k, v in vars(parser.parse_args()).items() if v is not None}


def run_main():
    """Main function."""

    args = parse_args()

    runner = TrainModelRunner(**args)
    runner.run()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format='%(asctime)s: %(module)s.%(funcName)s +%(lineno)s: %(levelname)-8s %(message)s', datefmt='%H:%M:%S')

    run_main()
