import logging
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from models.base import BaseLearner
from utils.inc_net import IncrementalNet
from utils.inc_net import CosineIncrementalNet
from utils.toolkit import target2onehot, tensor2numpy
from sklearn.metrics import confusion_matrix,precision_score,recall_score,f1_score
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from collections import Counter
EPSILON = 1e-8

init_epoch = 100
init_lr = 0.01
init_milestones = [40,80]
init_lr_decay = 0.1
init_weight_decay = 0.0005


epochs = 100
lrate = 0.01
milestones = [40,80]
lrate_decay = 0.01
batch_size = 128
weight_decay = 2e-4
num_workers = 8
T = 2


class iCaRL(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = IncrementalNet(args, False)

    def after_task(self):
        self._old_network = self._network.copy().freeze()
        self._known_classes = self._total_classes
        logging.info("Exemplar size: {}".format(self.exemplar_size))

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(
            self._cur_task
        )
        self._network.update_fc(self._total_classes)
        logging.info(
            "Learning on {}-{}".format(self._known_classes, self._total_classes)
        )

        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="train",
            appendent=self._get_memory(),
        )
        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test"
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
        class_counts = Counter(train_dataset.labels)
        self.cls_num_list = [count for _, count in sorted(class_counts.items())]
        max_val = max(self.cls_num_list)
        min_val = min(self.cls_num_list)
        self.imbalancefactor = max_val / min_val 
        if self._cur_task == 0:
            logging.info("the init train dataset is {} ,the imbalancefactor is:{:.3f} ".format(sum(self.cls_num_list),(self.imbalancefactor)))
                   
        else:
            logging.info("the task {} train dataset is: {},the imbalancefactor is:{:.3f} ".format((self._cur_task),sum(self.cls_num_list),(self.imbalancefactor)))
            self.incremental_dataset = self.cls_num_list[self._known_classes:]
            logging.info("the incremental dataset is: {}".format(sum(self.incremental_dataset)))
            for i in range(len(self.incremental_dataset)):
                logging.info(f"the incremental class {i + self._known_classes} has {self.incremental_dataset[i]} number samples.")

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader)
        self.build_rehearsal_memory(data_manager, self.samples_per_class)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)
        if self._old_network is not None:
            self._old_network.to(self._device)

        if self._cur_task == 0:
            optimizer = optim.SGD(
                self._network.parameters(),
                momentum=0.9,
                lr=init_lr,
                weight_decay=init_weight_decay,
            )
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer, milestones=init_milestones, gamma=init_lr_decay
            )
            self._init_train(train_loader, test_loader, optimizer, scheduler)
        else:
            optimizer = optim.SGD(
                self._network.parameters(),
                lr=lrate,
                momentum=0.9,
                weight_decay=weight_decay,
            )  # 1e-5
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer, milestones=milestones, gamma=lrate_decay
            )
            self._update_representation(train_loader, test_loader, optimizer, scheduler)

    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(init_epoch))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                # print(inputs.shape)
                logits = self._network(inputs)["logits"]
                # print(logits.shape)
                targets = targets.to(torch.int64)

                loss = F.cross_entropy(logits, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            if epoch % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    init_epoch,
                    losses / len(train_loader),
                    train_acc,
                    test_acc,
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    init_epoch,
                    losses / len(train_loader),
                    train_acc,
                )

            prog_bar.set_description(info)
            logging.info(info)

        self._network.eval()
        y_pred, y_true = [], []
        for _, (_, inputs, targets) in enumerate(test_loader):
            inputs = inputs.to(self._device, non_blocking=True)
            with torch.no_grad():
                outputs = self._network(inputs)["logits"]
            predicts = torch.topk(
                outputs, k=self.topk, dim=1, largest=True, sorted=True
            )[1]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())
        y_pred = np.concatenate(y_pred)
        y_true = np.concatenate(y_true)
        cnn_accy = self._evaluate(y_pred, y_true)
        y_true = np.array(y_true)
        y_pred = np.array(y_pred.T[0])
        macro_precision = precision_score(y_true, y_pred, average='macro', zero_division=0)  
        print('Macro precision', round(macro_precision * 100, 2), '%')  
        
        macro_recall = recall_score(y_true, y_pred, average='macro')  
        print('Macro recall', round(macro_recall * 100, 2), '%')  
        
        macro_f1 = f1_score(y_true, y_pred, average='macro')  
        print('Macro f1-score', round(macro_f1 * 100, 2), '%') 
    

    def _update_representation(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(epochs))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits = self._network(inputs)["logits"]
                targets = targets.to(torch.int64)

                loss_clf = F.cross_entropy(logits, targets)
                loss_kd = _KD_loss(
                    logits[:, : self._known_classes],
                    self._old_network(inputs)["logits"],
                    T,
                )

                loss = loss_clf + loss_kd

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            if epoch % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    epochs,
                    losses / len(train_loader),
                    train_acc,
                    test_acc,
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    epochs,
                    losses / len(train_loader),
                    train_acc,
                )
            prog_bar.set_description(info)
        logging.info(info)

        self._network.eval()
        y_pred, y_true = [], []
        for _, (_, inputs, targets) in enumerate(test_loader):
            inputs = inputs.to(self._device, non_blocking=True)
            with torch.no_grad():
                outputs = self._network(inputs)["logits"]
            predicts = torch.topk(
                outputs, k=self.topk, dim=1, largest=True, sorted=True
            )[1]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())
        y_pred = np.concatenate(y_pred)
        y_true = np.concatenate(y_true)



        
        cnn_accy = self._evaluate(y_pred, y_true)
        y_true = np.array(y_true)
        y_pred = np.array(y_pred.T[0])
        cm = confusion_matrix(y_true, y_pred, labels=np.arange(self._total_classes))
       
        cmn = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

        fig, ax = plt.subplots(figsize=(12, 10))  
        sns.heatmap(cmn, annot=False, cmap='rainbow', ax=ax)  # 使用自定义的colormap

        ax.set_xticks(np.arange(0, self._total_classes+1, 5))
        ax.set_yticks(np.arange(0, self._total_classes+1, 5))

        ax.set_xticklabels(np.arange(0, self._total_classes+1, 5))
        ax.set_yticklabels(np.arange(0, self._total_classes+1, 5))


        ax.set_xlabel("Predicted labels")
        ax.set_ylabel("True labels")
        ax.set_title("Confusion Matrix")
        plt.savefig("icarliAMDDataset25-35without_MYmethod_confusion_matrix.png")
        plt.close()

        print('------Macro------')
        macro_precision = precision_score(y_true, y_pred, average='macro', zero_division=0)  
        print('Macro precision', round(macro_precision * 100, 2), '%')  
        
        macro_recall = recall_score(y_true, y_pred, average='macro')  
        print('Macro recall', round(macro_recall * 100, 2), '%')  
        
        macro_f1 = f1_score(y_true, y_pred, average='macro')  
        print('Macro f1-score', round(macro_f1 * 100, 2), '%') 
        
        logging.info("CNN top1 curve: {}".format(cnn_accy["top1"]))
        logging.info("CNN top5 curve: {}".format(cnn_accy["top5"]))


def _KD_loss(pred, soft, T):
    pred = torch.log_softmax(pred / T, dim=1)
    soft = torch.softmax(soft / T, dim=1)
    return -1 * torch.mul(soft, pred).sum() / pred.shape[0]
