from torch.utils.data import DataLoader
import torch.nn as nn
from torch.autograd import Variable
import torch
import logging
import wandb
from utils.utils import login_wandb

class Trainer:
    def __init__(
        self,
        train_data_loader,
        model,
        fold,
        device,
        config,
        criterion,
        softmax,
        reg_criterion,
        optimizer_gaze,
    ):
        self._train_data_loader = train_data_loader
        self._model = model
        self._fold = fold
        self._device = device
        self._config = config
        self._criterion = criterion
        self._softmax = softmax
        self._reg_criterion = reg_criterion
        self._optimizer_gaze = optimizer_gaze

        ids = [idx for idx in range(28)]
        self._idx_tensor = Variable(torch.FloatTensor(ids)).cuda(self._device)

    def train(self):
        login_wandb()
        wandb_config = dict(
            epochs = self._config.train.num_epochs,
            batch_size = self._config.train.batch_size,
            learning_rate = self._config.train.learning_rate,
            optimizer = self._config.train.optimizer,
            use_gpu = self._config.train.use_gpu,
            seed = self._config.train.seed,
            is_pipeline_test = self._config.train.is_pipeline_test,
            backbone = self._config.model.backbone,
            dataset_name = self._config.data.dataset_name
        )
        
        with wandb.init(project="gaze_estimation", config=wandb_config):
            wandb.watch(self._model, self._reg_criterion, log="all", log_freq=10)

            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s [%(levelname)s] -------- %(message)s --------",
                datefmt="%d/%m/%Y %H:%M:%S",
                handlers=[
                    logging.FileHandler(
                        self._config.experiment_path / "output" / "train.log"
                    ),
                    logging.StreamHandler(),
                ],
            )

            logging.info(
                f"Starting training with fold: {self._fold}, num_epochs: {self._config.train.num_epochs}, num_batches: {len(self._train_data_loader)}, batch_size: {self._config.train.batch_size}"
            )

            batches_seen = 0
            for epoch in range(self._config.train.num_epochs):
                logging.info(f"Starting epoch: {epoch+1}/{self._config.train.num_epochs}")
                sum_loss_pitch_gaze = sum_loss_yaw_gaze = iter_gaze = 0

                for i, (images_gaze, labels_gaze, cont_labels_gaze) in enumerate(
                    self._train_data_loader
                ):
                    batches_seen += 1
                    images_gaze = Variable(images_gaze).to("cuda")

                    # Binned labels
                    label_pitch_gaze = Variable(labels_gaze[:, 0]).cuda(self._device)
                    label_yaw_gaze = Variable(labels_gaze[:, 1]).cuda(self._device)

                    # Continuous labels
                    label_pitch_cont_gaze = Variable(cont_labels_gaze[:, 0]).cuda(
                        self._device
                    )
                    label_yaw_cont_gaze = Variable(cont_labels_gaze[:, 1]).cuda(
                        self._device
                    )

                    pitch, yaw = self._model(images_gaze)

                    # Cross entropy loss
                    loss_pitch_gaze = self._criterion(pitch, label_pitch_gaze)
                    loss_yaw_gaze = self._criterion(yaw, label_yaw_gaze)

                    # MSE loss
                    pitch_predicted = self._softmax(pitch)
                    yaw_predicted = self._softmax(yaw)

                    pitch_predicted = (
                        torch.sum(pitch_predicted * self._idx_tensor, 1) * 3 - 42
                    )
                    yaw_predicted = torch.sum(yaw_predicted * self._idx_tensor, 1) * 3 - 42

                    loss_reg_pitch = self._reg_criterion(
                        pitch_predicted, label_pitch_cont_gaze
                    )
                    loss_reg_yaw = self._reg_criterion(yaw_predicted, label_yaw_cont_gaze)

                    # Total loss
                    loss_pitch_gaze += loss_reg_pitch
                    loss_yaw_gaze += loss_reg_yaw

                    sum_loss_pitch_gaze += loss_pitch_gaze
                    sum_loss_yaw_gaze += loss_yaw_gaze

                    loss_seq = [loss_pitch_gaze, loss_yaw_gaze]
                    grad_seq = [
                        torch.tensor(1.0).cuda(self._device) for _ in range(len(loss_seq))
                    ]

                    self._optimizer_gaze.zero_grad(set_to_none=True)
                    torch.autograd.backward(loss_seq, grad_seq)
                    self._optimizer_gaze.step()

                    iter_gaze += 1

                    if i % 100 == 0:
                        wandb.log({"epoch": epoch, "batch": i, "avg_pitch_loss": sum_loss_pitch_gaze / iter_gaze, "avg_yaw_loss": sum_loss_yaw_gaze / iter_gaze}, step=batches_seen)

                        logging.info(
                            "Batch: [%d/%d] Losses: Gaze Yaw %.4f, Gaze Pitch %.4f"
                            % (
                                i,
                                len(self._train_data_loader),
                                sum_loss_pitch_gaze / iter_gaze,
                                sum_loss_yaw_gaze / iter_gaze,
                            )
                        )

                        torch.save(
                            self._model.state_dict(),
                            self._config.experiment_path
                            / "output"
                            / "models"
                            / f"fold_{self._fold}"
                            / f"epoch_{epoch+1}.pkl",
                        )

                wandb.log({"epoch": epoch, "batch": i, "avg_pitch_loss": sum_loss_pitch_gaze / iter_gaze, "avg_yaw_loss": sum_loss_yaw_gaze / iter_gaze}, step=batches_seen)

                logging.info(
                    "Batch: [%d/%d] Losses: Gaze Yaw %.4f, Gaze Pitch %.4f"
                    % (
                        len(self._train_data_loader),
                        len(self._train_data_loader),
                        sum_loss_pitch_gaze / iter_gaze,
                        sum_loss_yaw_gaze / iter_gaze,
                    )
                )

                torch.save(
                    self._model.state_dict(),
                    self._config.experiment_path
                    / "output"
                    / "models"
                    / f"fold_{self._fold}"
                    / f"epoch_{epoch+1}.pkl",
                )

                torch.onnx.export(self._model, images_gaze, self._config.experiment_path / "output" / "models" / "model.onnx")
                wandb.save(self._config.experiment_path / "output" / "models" / "model.onnx", base_path = self._config.experiment_path / "output" / "models")

            logging.info(f"Training complete for model with fold: {self._fold}")
            


class TrainerBuilder:
    def new_session(self):
        self._train_dataloader = None
        self._fold = None
        self._model = None
        self._device = None
        self._config = None
        self._criterion = None
        self._softmax = None
        self._reg_criterion = None
        self._optimizer_gaze = None
        return self

    def add_model(self, model):
        self._model = model
        return self

    def add_train_dataloader(self, train_dataloader):
        self._train_dataloader = train_dataloader
        return self

    def add_fold(self, fold):
        self._fold = fold
        return self

    def add_device(self, device):
        self._device = device
        return self

    def add_config(self, config):
        self._config = config
        return self

    def add_criterion(self, criterion):
        self._criterion = criterion
        return self

    def add_softmax(self, softmax):
        self._softmax = softmax
        return self

    def add_reg_criterion(self, reg_criterion):
        self._reg_criterion = reg_criterion
        return self

    def add_optimizer_gaze(self, optimizer_gaze):
        self._optimizer_gaze = optimizer_gaze
        return self

    def build(self):
        return Trainer(
            self._train_dataloader,
            self._model,
            self._fold,
            self._device,
            self._config,
            self._criterion,
            self._softmax,
            self._reg_criterion,
            self._optimizer_gaze,
        )
