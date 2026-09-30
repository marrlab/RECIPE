import torch
import torch.utils.data as data_utils
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix

import time
import copy
import numpy as np
from tqdm import tqdm
import os
import pandas as pd
from typing import Dict, Tuple, List

from rollout import generate_rollout
from patient_data import DataMatrix, PatientRecord


class ModelTrainer:
    '''class containing all the info about the training process and handling the actual
    training function'''

    def __init__(
            self,
            model,
            dataloaders,
            epochs,
            optimizer,
            sched_builder,
            class_count,
            device,
            save_path,
            early_stop=20,
            grad_accum=20):
        self.model = model
        self.dataloaders = dataloaders
        self.epochs = epochs
        self.optimizer = optimizer
        self.scheduler = sched_builder.scheduler
        self.scheduler_name = sched_builder.name
        self.class_count = class_count
        self.early_stop = early_stop
        self.device = device
        self.save_path = save_path
        self.grad_accum = grad_accum
        self.data_obj = DataMatrix(class_count)
        self.metric_mode = (
            "max" if self.scheduler_name == "ReduceLROnPlateau" and self.scheduler.mode == "max" else "min"
        )

    def launch_training(self):
        '''initializes training process.'''
        #print("Initialized")
        best_metric = -np.inf if self.metric_mode == "max" else np.inf
        best_state = copy.deepcopy(self.model.state_dict())
        _no_improve_epochs = 0

        for ep in range(self.epochs):
            # perform train/val iteration
            val_metrics = self._run_epoch(ep, backprop_every=self.grad_accum)
            torch.cuda.empty_cache()
            target_metric = val_metrics["weighted_f1"] if self.metric_mode == "max" else val_metrics["val_loss"]
            is_better = (target_metric > best_metric) if self.metric_mode == "max" else (target_metric < best_metric)

            # if improvement, reset counter
            if is_better:
                best_metric = target_metric
                best_state = copy.deepcopy(self.model.state_dict())
                _no_improve_epochs = 0
                os.makedirs(self.save_path, exist_ok=True)
                torch.save(best_state, self.save_path / "cAItomorph_best_weights.pth")
                print(f"🔖  Saved new best model (metric={best_metric:.4f})")
            else:
                _no_improve_epochs += 1
                if _no_improve_epochs >= self.early_stop:
                    print(f"⏹️  Early-stopping triggered after {ep} epochs")
                    break

            if self.scheduler_name == "ReduceLROnPlateau":
                self.scheduler.step(target_metric)
            elif self.scheduler is not None:
                self.scheduler.step()

        # load best performing model, and launch on test set
        self.model.load_state_dict(best_state)
        #Save logits for val

        _ = self._infer_split('train')
        _ = self._infer_split('val')
        test_metrics = self._infer_split('test')
        print("Inference completed for splits: train, val, test")

        self.data_obj.save_hdf5(self.save_path / "patient_data.h5")

        return self.model, test_metrics['conf_matrix']

    def launch_eval_only(self):
        '''Skip training entirely and evaluate the already-loaded model weights on
        all splits. Used to recover the test result from a saved best-val
        checkpoint (e.g. a run that timed out after training but before the test
        step wrote test_conf_matrix.npy) without retraining. The caller is
        responsible for having loaded the weights into ``self.model``.'''
        self.model.eval()
        _ = self._infer_split('train')
        _ = self._infer_split('val')
        test_metrics = self._infer_split('test')
        print("Eval-only inference completed for splits: train, val, test")
        self.data_obj.save_hdf5(self.save_path / "patient_data.h5")
        return self.model, test_metrics['conf_matrix']

    def _run_epoch(self, epoch, backprop_every=20):
        '''runs one epoch of training and validation. Returns the loss, accuracy and f1 score for the epoch.'''
        # Training
        train_loss = 0
        all_predictions = []
        all_labels = []
        time_pre_epoch = time.time()
        self.optimizer.zero_grad()
        backprop_counter = 0

        self.model.train()
        for (bag, label, img_paths, patient_id) in tqdm(self.dataloaders['train']):
            patient_id = patient_id[0]
            # send to gpu
            label = label.to(self.device)
            bag = bag.to(self.device)
            prediction = self.model(bag)

            loss_func = nn.CrossEntropyLoss()

            #loss_out = loss_func(prediction, label[0])
            loss_out = loss_func(prediction, label)
            train_loss += loss_out.item()

            loss_out.backward()
            backprop_counter += 1

            if(backprop_counter % backprop_every == 0) or (backprop_counter == len(self.dataloaders['train'])):
                self.optimizer.step()
                self.optimizer.zero_grad()

            # transforms prediction tensor into index of position with highest
            # value
            label_prediction = torch.argmax(prediction, dim=1).item()
            label_groundtruth = label.item()

            all_predictions.append(label_prediction)
            all_labels.append(label_groundtruth)

        train_loss /= len(self.dataloaders['train'])
        accuracy = accuracy_score(all_labels, all_predictions)
        balanced_acc = balanced_accuracy_score(all_labels, all_predictions)
        w_f1 = f1_score(all_labels, all_predictions, average='weighted')

        print('- ep: {}/{}, loss: {:.3f}, acc: {:.3f}, balanced acc: {:.3f},weighted_f1: {:.3f}, {}s, {}'.format(
            epoch + 1, self.epochs, train_loss,
            accuracy, balanced_acc, w_f1, int(time.time() - time_pre_epoch), 'train'))

        # Validation
        val_loss = 0.
        time_pre_epoch = time.time()
        all_predictions = []
        all_labels = []
        self.model.eval()
        with torch.no_grad():
            for (bag, label, img_paths, patient_id) in tqdm(self.dataloaders['val']):
                patient_id = patient_id[0]
                # send to gpu
                label = label.to(self.device)
                bag = bag.to(self.device)
                prediction = self.model(bag)

                loss_func = nn.CrossEntropyLoss()
                loss_out = loss_func(prediction, label)
                val_loss += loss_out.item()

                label_prediction = torch.argmax(prediction, dim=1).item()
                label_groundtruth = label.item()

                all_predictions.append(label_prediction)
                all_labels.append(label_groundtruth)

        val_loss /= len(self.dataloaders['val'])

        accuracy = accuracy_score(all_labels, all_predictions)
        balanced_ac = balanced_accuracy_score(all_labels, all_predictions)
        wf1 = f1_score(all_labels, all_predictions, average='weighted')

        print('- ep: {}/{}, loss: {:.3f}, acc: {:.3f}, balanced acc: {:.3f},weighted_f1: {:.3f}, {}s, {}'.format(
            epoch + 1, self.epochs, val_loss,
            accuracy, balanced_ac, wf1, int(time.time() - time_pre_epoch), 'val'))

        return { 'train_loss': train_loss, 'val_loss': val_loss, 'accuracy': accuracy, 'weighted_f1': wf1 }


    def _infer_split(
        self,
        split: str,
        *,
        save_logits: bool = True,
    ) -> Dict[str, float]:
        self.model.eval()
        dataloader = self.dataloaders[split]

        preds, label_preds, labels, patients = [], [], [], []
        running_loss = 0.0
        with torch.no_grad():
            for bag, label, img_paths, patient_id in tqdm(dataloader, desc=f"Inference {split}"):
                patient_id = patient_id[0] 
                bag   = bag.to(self.device)
                label = label.to(self.device)

                latent, logits = self.model(bag, return_latent=True)
                loss = nn.CrossEntropyLoss()(logits, label)
                running_loss += loss.item() * bag.size(0)

                #cls_attention_scores = generate_rollout(self.model,bag,start_layer=0)
                #cls_attention_scores = cls_attention_scores.squeeze(0)
                cls_attention_scores = []
                label_prediction = torch.argmax(logits, dim=1).item()
                label_groundtruth = label.item()

                self.data_obj.add_patient(
                        patient_id,
                        PatientRecord.from_tensors(
                        true_label=label_groundtruth,
                        pred_label=label_prediction,
                        latent=latent,
                        attention=cls_attention_scores,
                        image_paths=img_paths,
                        prediction_vector=logits,
                        loss=loss,
                    ),
                )

                label_preds.append(label_prediction)
                labels.append(label_groundtruth)
                preds.append(logits)
                patients.append(patient_id)


        preds = torch.cat(preds, dim=0).detach().cpu().numpy()

        epoch_loss = running_loss / len(dataloader.dataset)
        acc = accuracy_score(labels, label_preds)
        bal_acc = balanced_accuracy_score(labels, label_preds)
        f1_weighted = f1_score(labels, label_preds, average='weighted')
        conf_matrix = confusion_matrix(labels, label_preds, labels=list(range(self.class_count)))

        print(f"{split}: loss={epoch_loss:.4f} acc={acc:.3f} balAcc={bal_acc:.3f} f1w={f1_weighted:.3f}")

        if save_logits:
            pd.DataFrame({
                "patient": patients,
                "label": labels,
                "prediction": [list(p) for p in preds],
            }).to_csv(self.save_path / f"metadata_results_{split}.csv", index=False)

        return {
            "loss": epoch_loss,
            "acc": acc,
            "balanced_acc": bal_acc,
            "f1": f1_weighted,
            "conf_matrix": conf_matrix,
        }