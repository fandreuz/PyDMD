import logging
from copy import deepcopy

import torch
import torch.optim as optim
from torch.nn.functional import mse_loss
from torch.utils.data import DataLoader

logging.basicConfig(
    format="[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
)


class DLDMD(torch.nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
        reconstruction_weight,
        prediction_weight,
        phase_space_weight,
        dmd,
        optimizer=optim.Adam,
        optimizer_kwargs={"lr": 1.0e-3, "weight_decay": 1.0e-9},
        epochs=1000,
        print_every=100,
        batch_size=256,
    ):
        super().__init__()

        self._encoder = encoder
        self._decoder = decoder

        self._reconstruction_weight = reconstruction_weight
        self._prediction_weight = prediction_weight
        self._phase_space_weight = phase_space_weight

        if isinstance(epochs, float):
            self._epochs = 100000000
            self._acceptable_loss = epochs
        else:
            self._epochs = epochs
            self._acceptable_loss = 0.0

        self._optimizer = optimizer(self.parameters(), **optimizer_kwargs)
        logging.info(f"Optimizer: {self._optimizer}")

        self._print_every = print_every
        self._batch_size = batch_size

        logging.info(f"DMD instance: {type(dmd)}")
        # TODO we should support batching internally with PyTorch
        logging.info(f"Replicating DMD instance {self._batch_size} times")
        self._dmd = tuple(deepcopy(dmd) for _ in range(self._batch_size))

        logging.info("----- DLDMD children -----")
        logging.info(tuple(self.children()))

    def forward(self, input):
        if input.ndim == 2:
            input = input[None]

        encoded_input = self._encoder(input)
        encoded_output = torch.stack(
            [
                dmd.fit(data).reconstructed_data
                for data, dmd in zip(encoded_input, self._dmd)
            ]
        )

        if not torch.is_complex(input):
            old_dtype = encoded_output.dtype
            encoded_output = encoded_output.real
            logging.info(
                f"Removing complex part from output_immersion: {old_dtype} to {encoded_output.dtype}"
            )
        if encoded_output.dtype != input.dtype:
            logging.info(
                f"Casting output_immersion dtype from {encoded_output.dtype} to {input.dtype}"
            )
            encoded_output = encoded_output.to(dtype=input.dtype)

        return self._decoder(encoded_output)

    def _compute_loss(self, output, input):
        decoder_loss = mse_loss(self._decoder(self._encoder(input)), input)

        # TODO we should support batching internally with PyTorch
        batched_psp = torch.stack(
            tuple(
                dmd.operator.phase_space_prediction
                for dmd in self._dmd[:len(input)]
            )
        )
        psp_loss = torch.linalg.matrix_norm(batched_psp).sum()

        reconstruction_loss = mse_loss(output, input)

        return (
            self._reconstruction_weight * decoder_loss
            + self._prediction_weight * psp_loss
            + self._phase_space_weight * reconstruction_loss
        )

    def fit(self, X):
        """
        Compute the Deep-Learning enhanced DMD on the input data.

        :param X: the input dataset as a dict with keys `'training_data'` and `test_data`.
        :type X: dict
        """
        train_dataloader = DataLoader(
            X["training_data"], batch_size=self._batch_size, shuffle=True
        )
        test_dataloader = DataLoader(
            X["test_data"], batch_size=self._batch_size, shuffle=True
        )

        for epoch in range(1, self._epochs + 1):
            self.train()
            for minibatch in train_dataloader:
                self._optimizer.zero_grad()
                output = self(minibatch)
                loss = self._compute_loss(output, minibatch)
                loss.backward()
                self._optimizer.step()

            self.eval()
            running_loss_sum = 0.0
            for i, minibatch in enumerate(test_dataloader):
                output = self(minibatch)
                running_loss_sum += self._compute_loss(output, minibatch).item()
            running_loss_mean = running_loss_sum / (i + 1)

            if epoch % self._print_every == 0:
                print(f"[{epoch}] loss: {running_loss_mean:.3f}")

        return self
