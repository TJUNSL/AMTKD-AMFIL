import logging
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from models.base import BaseLearner
from utils.inc_net import KINDROIDNet
from utils.toolkit import count_parameters, target2onehot, tensor2numpy
from collections import Counter
from utils.loss import IB_FocalLoss
import torch.nn.utils.prune as prune
from sklearn.metrics import confusion_matrix,precision_score,recall_score,f1_score
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns



EPSILON = 1e-8


class KINDROIDNet(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self._network = KINDROIDNet(args, False)
        self._basenet = None
        self.oofc = args["oofc"].lower()
        self.cls_num_list = None
        self._init_classes = args["init_cls"]
        self.lamda = None
        self.beta = 0.99
        self.imbalancefactor = None
        self.per_cls_weights = None



    def after_task(self):
        self._known_classes = self._total_classes
        logging.info("Exemplar size: {}".format(self.exemplar_size))

    def incremental_train(self, data_manager):
        self.data_manager = data_manager
        self._cur_task += 1
        
        

        self._total_classes = self._known_classes + data_manager.get_task_size(
            self._cur_task
        )
        self._network.update_fc(self._total_classes)
        
        logging.info(
            "Learning on {}-{}".format(self._known_classes, self._total_classes)
        )

        if self._cur_task > 0:
            for i in range(self._cur_task):
                for p in self._network.convnets[i].parameters():
                    p.requires_grad = False
            

        logging.info("All params: {}".format(count_parameters(self._network)))
        logging.info(
            "Trainable params: {}".format(count_parameters(self._network, True))
        )

        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="train",
            appendent=self._get_memory(),
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
            
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.args["batch_size"],
            shuffle=True,
            num_workers=self.args["num_workers"],
            pin_memory=True,
        )
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test"
        )
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.args["batch_size"],
            shuffle=False,
            num_workers=self.args["num_workers"],
        )

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader)
        self.build_rehearsal_memory(data_manager, self.samples_per_class)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def train(self):
        self._network.train()
        if len(self._multiple_gpus) > 1:
            self._network_module_ptr = self._network.module
        else:
            self._network_module_ptr = self._network
        self._network_module_ptr.convnets[-1].train()
        
        if self._cur_task >= 1:
            for i in range(self._cur_task):
                self._network_module_ptr.convnets[i].eval()

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)
        if hasattr(self._network, "module"):
            self._network_module_ptr = self._network.module
        if self._cur_task == 0:
            optimizer = optim.SGD(
                filter(lambda p: p.requires_grad, self._network.parameters()),
                momentum=0.9,
                lr=self.args["init_lr"],
                weight_decay=self.args["init_weight_decay"],
            )
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer, T_max=self.args["init_epochs"]
            )

            self._init_train(train_loader, test_loader, optimizer, scheduler)
            self._basenet = self._network.copy().freeze()
        else:
            optimizer = optim.SGD(
                filter(lambda p: p.requires_grad, self._network.parameters()),
                lr=self.args["lr"],
                momentum=0.9,
                weight_decay=self.args["weight_decay"],
            )
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer, T_max=self.args["boosting_epochs"]
            )
            if self.oofc == "az":
                for i, p in enumerate(self._network_module_ptr.fc.parameters()):
                    if i == 0:
                        p.data[
                            self._known_classes :, : self._network_module_ptr.out_dim
                        ] = torch.tensor(0.0)
            elif self.oofc != "ft":
                assert 0, "not implemented"
            self._basenet.to(self._device)
            self._update_representation(train_loader, test_loader, optimizer, scheduler)
            self._prune_network(train_loader, test_loader, optimizer, scheduler)
          
            
     

    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self.args["init_epochs"]))

        for _, epoch in enumerate(prog_bar):
            self.train()
            losses = 0.0
            correct, total = 0, 0

            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(
                    self._device, non_blocking=True
                ), targets.to(self._device, non_blocking=True)
                
                outputs = self._network(inputs)
                logits,features = outputs["logits"],outputs["IB_features"].detach()
                effective_num = 1.0 - np.power(self.beta, self.cls_num_list)
                self.per_cls_weights = (1.0 - self.beta) / np.array(effective_num)
                self.per_cls_weights = self.per_cls_weights / np.sum(self.per_cls_weights) * len(self.cls_num_list)
                self.per_cls_weights = torch.FloatTensor(self.per_cls_weights).to(self._device)
                criterion_ib = IB_FocalLoss(weight=self.per_cls_weights, alpha=1000, gamma=1).to(self._device)
                loss = criterion_ib(logits,targets,features,self._total_classes)
                # 如果你的显卡是RTX X0XX系列，可以使用下面的代码来自动混合精度训练
                # If your graphics card is RTX X0XX series, you can use the following code to automatically perform mixed precision training
                # logits = self._network(inputs)["logits"]
                # loss = F.cross_entropy(logits, targets)
                losses += loss.item()
                _, preds = torch.max(logits, dim=1)
                total += len(targets)

                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                optimizer.zero_grad()

                loss.backward()
                optimizer.step()
            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            
            if epoch % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args["init_epochs"],
                    losses / len(train_loader),
                    train_acc,
                    test_acc,
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args["init_epochs"],
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

        total_mem = 0
        for i in range(torch.cuda.device_count()):
            mem = torch.cuda.memory_reserved(i) / 1024 / 1024 / 1024
            total_mem += mem
        info = f"before prune net the GPU memory allocated: {total_mem:.2f} GB"
        logging.info(info)

    def _update_representation(self, train_loader, test_loader, optimizer, scheduler):
        
        prog_bar = tqdm(range(self.args["boosting_epochs"]))
        for _, epoch in enumerate(prog_bar):
            self.train()
            losses = 0.0
            losses_clf = 0.0
            losses_kd = 0.0
            correct, total = 0, 0

            

            self.lamda = self._init_classes /self._known_classes
            # self.lamda = 0.5
            for i, (_, inputs, targets) in enumerate(train_loader):
                
                inputs, targets = inputs.to(
                    self._device, non_blocking=True
                ), targets.to(self._device, non_blocking=True)

                outputs = self._network(inputs)                
                logits, old_logits, = (
                    outputs["logits"],
                    outputs["old_logits"].detach()
                )

                features = outputs["IB_features"].detach()
                base_logits = self._basenet(inputs)["logits"]
                effective_num = 1.0 - np.power(self.beta, self.cls_num_list)
                self.per_cls_weights = (1.0 - self.beta) / np.array(effective_num)      
                self.per_cls_weights = self.per_cls_weights / np.sum(self.per_cls_weights) * len(self.cls_num_list)
                self.per_cls_weights = torch.FloatTensor(self.per_cls_weights).to(self._device)
                criterion_ib = IB_FocalLoss(weight=self.per_cls_weights, alpha=1000, gamma=1).to(self._device)
                loss_clf = criterion_ib(logits,targets,features,self._total_classes)

                loss_kd = self.lamda * _KD_loss(logits[:,:self._init_classes], base_logits, self.args["T"]) + (1 - self.lamda) * _KD_loss(logits[:,self._init_classes:self._known_classes], old_logits[:,self._init_classes:], self.args["T"])   
                loss = (1 - self._known_classes / self._total_classes ) * loss_clf +  (self._known_classes / self._total_classes) * loss_kd 
                optimizer.zero_grad()
                loss.backward()
                if self.oofc == "az":
                    for i, p in enumerate(self._network_module_ptr.fc.parameters()):
                        if i == 0:
                            p.grad.data[
                                self._known_classes :,
                                : self._network_module_ptr.out_dim,
                            ] = torch.tensor(0.0)
                elif self.oofc != "ft":
                    assert 0, "not implemented"
                optimizer.step()
                losses += loss.item()

                losses_clf += loss_clf.item()

                losses_kd += loss_kd.item()
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)
            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            if epoch % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Loss_clf {:.3f}, Loss_kd {:.3f},Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args["boosting_epochs"],
                    losses / len(train_loader),
                    losses_clf / len(train_loader),
                    losses_kd / len(train_loader),
                    train_acc,
                    test_acc,
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f},Loss_clf {:.3f}, Loss_kd {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args["boosting_epochs"],
                    losses / len(train_loader),
                    losses_clf / len(train_loader),
                    losses_kd / len(train_loader),
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


        ax.set_xlabel("Predicted labels", fontsize=18)
        ax.set_ylabel("True labels", fontsize=18)
        ax.set_title("Confusion Matrix", fontsize=18)
        plt.savefig("XXXX_confusion_matrix.png")
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
    
    def _prune_network(self, train_loader, test_loader, optimizer, scheduler):
        """
        Prunes the network and fine-tunes it afterwards.
        """
        
        parameters_to_prune = []
        for name , module in self._network.convnets[-1].named_modules():
            if isinstance(module, nn.Conv2d):
                parameters_to_prune.append((module, 'weight'))
                
        
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=0.2,
        )
        
        # # 移除剪枝过程中添加的重参数化模块，使模型更加紧凑
        for module in parameters_to_prune:
            if isinstance(module, nn.Conv2d):
                prune.remove(module, 'weight')
            

        # 打印剪枝后的模型参数量
        # Print the number of model parameters after pruning

        # 统计非零参数  
        # Statistical non-zero parameters
        total_nonzero_params = 0  
        for module in self._network.modules():  
            if isinstance(module, nn.Conv2d):  
                weight = module.weight.data  
                total_nonzero_params += weight.nonzero().size(0)  
        
        print(f"Total non-zero parameters after pruning: {total_nonzero_params}")


        prog_bar = tqdm(range(self.args["prune_epochs"]))
        for _, epoch in enumerate(prog_bar):
            self.train()
            losses = 0.0
            correct, total = 0, 0

            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(
                    self._device, non_blocking=True
                ), targets.to(self._device, non_blocking=True)
                outputs = self._network(inputs)
                logits,features = outputs["logits"],outputs["IB_features"].detach()
                effective_num = 1.0 - np.power(self.beta, self.cls_num_list)
                self.per_cls_weights = (1.0 - self.beta) / np.array(effective_num)
                self.per_cls_weights = self.per_cls_weights / np.sum(self.per_cls_weights) * len(self.cls_num_list)
                self.per_cls_weights = torch.FloatTensor(self.per_cls_weights).to(self._device)
                criterion_ib = IB_FocalLoss(weight=self.per_cls_weights, alpha=1000, gamma=1).to(self._device)
                loss = criterion_ib(logits,targets,features,self._total_classes)
                losses += loss.item()
                _, preds = torch.max(logits, dim=1)
                total += len(targets)

                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                optimizer.zero_grad()

                loss.backward()
                optimizer.step()
            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            
            if epoch % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "prune Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args["prune_epochs"],
                    losses / len(train_loader),
                    train_acc,
                    test_acc,
                )
            else:
                info = "prune Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args["prune_epochs"],
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


        ax.set_xlabel("Predicted labels", fontsize=18)
        ax.set_ylabel("True labels", fontsize=18)
        ax.set_title("Confusion Matrix", fontsize=18)
        plt.savefig("XXXX_confusion_matrix.png")
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
        
     

    @property
    def samples_old_class(self):
        if self._fixed_memory:
            return self._memory_per_class
        else:
            assert self._total_classes != 0, "Total classes is 0"
            return self._memory_size // self._known_classes

    def count_sparsity(self,model: torch.nn.Module, p=True):
        sum_zeros_num = 0
        sum_weights_num = 0
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                zeros_elements = torch.sum(torch.eq(module.weight, 0)).item()
                weights_elements = module.weight.numel()

                sum_zeros_num += zeros_elements
                sum_weights_num += weights_elements
                if p is True:
                    print("Sparsity in {}.weights {:.2f}%".format(name, 100 * zeros_elements / weights_elements))
        print("Global sparsity: {:.2f}%".format(100 * sum_zeros_num / sum_weights_num))
        return 100 * sum_zeros_num / sum_weights_num

    def count_nonzero_parameters(self,model):
        total_params = 0
        for param in model.parameters():
            total_params += torch.count_nonzero(param)
        return total_params.item()

            

def _KD_loss(pred, soft, T):
    pred = torch.log_softmax(pred / T, dim=1)
    soft = torch.softmax(soft / T, dim=1)
    return -1 * torch.mul(soft, pred).sum() / pred.shape[0]


