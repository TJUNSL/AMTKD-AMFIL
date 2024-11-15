import copy
import logging
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from utils.toolkit import tensor2numpy, accuracy
from scipy.spatial.distance import cdist
import os

EPSILON = 1e-8
batch_size = 64


class BaseLearner(object):
    def __init__(self, args):
        self.args = args
        self._cur_task = -1
        self._known_classes = 0
        self._total_classes = 0
        self._network = None
        self._old_network = None
        self._data_memory, self._targets_memory = np.array([]), np.array([])
        self.topk = 5

        self._memory_size = args["memory_size"]
        self._memory_per_class = args.get("memory_per_class", None)
        self._fixed_memory = args.get("fixed_memory", False)
        self._device = args["device"][0]
        self._multiple_gpus = args["device"]

    @property
    def exemplar_size(self):
        # print("the len of data_memory is:", len(self._data_memory))
        # print("the len of targets_memory is:", len(self._targets_memory))
        assert len(self._data_memory) == len(
            self._targets_memory
        ), "Exemplar size error."
        return len(self._targets_memory)

    @property
    def samples_per_class(self):
        if self._fixed_memory:
            return self._memory_per_class
        else:
            assert self._total_classes != 0, "Total classes is 0"
            return self._memory_size // self._total_classes

    @property
    def feature_dim(self):
        if isinstance(self._network, nn.DataParallel):
            return self._network.module.feature_dim
        else:
            return self._network.feature_dim

    def build_rehearsal_memory(self, data_manager, per_class):
        if self._fixed_memory:
            self._construct_exemplar_unified(data_manager, per_class)
        else:
            self._reduce_exemplar(data_manager, per_class)
            self._construct_exemplar(data_manager, per_class)

    def save_checkpoint(self, filename):
        self._network.cpu()
        save_dict = {
            "tasks": self._cur_task,
            "model_state_dict": self._network.state_dict(),
        }
        torch.save(save_dict, "{}_{}.pkl".format(filename, self._cur_task))

    def after_task(self):
        pass

    def _evaluate(self, y_pred, y_true):
        ret = {}
        grouped = accuracy(y_pred.T[0], y_true, self._known_classes)
        ret["grouped"] = grouped
        ret["top1"] = grouped["total"]
        ret["top{}".format(self.topk)] = np.around(
            (y_pred.T == np.tile(y_true, (self.topk, 1))).sum() * 100 / len(y_true),
            decimals=2,
        )

        return ret

    def eval_task(self, save_conf=False):
        y_pred, y_true = self._eval_cnn(self.test_loader)
        cnn_accy = self._evaluate(y_pred, y_true)

        if hasattr(self, "_class_means"):
            y_pred, y_true = self._eval_nme(self.test_loader, self._class_means)
            nme_accy = self._evaluate(y_pred, y_true)
        else:
            nme_accy = None

        if save_conf:
            _pred = y_pred.T[0]
            _pred_path = os.path.join(self.args['logfilename'], "pred.npy")
            _target_path = os.path.join(self.args['logfilename'], "target.npy")
            np.save(_pred_path, _pred)
            np.save(_target_path, y_true)

            _save_dir = os.path.join(f"./results/conf_matrix/{self.args['prefix']}")
            os.makedirs(_save_dir, exist_ok=True)
            _save_path = os.path.join(_save_dir, f"{self.args['csv_name']}.csv")
            with open(_save_path, "a+") as f:
                f.write(f"{self.args['time_str']},{self.args['model_name']},{_pred_path},{_target_path} \n")

        return cnn_accy, nme_accy

    def incremental_train(self):
        pass

    def _train(self):
        pass

    def _get_memory(self):
        if len(self._data_memory) == 0:
            return None
        else:
            return (self._data_memory, self._targets_memory)

    def _compute_accuracy(self, model, loader):
        model.eval()
        correct, total = 0, 0
        for i, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                # 通过模型前向传播输入数据，并从输出字典中提取logits。这里假设模型的输出是一个字典，其中包含键为"logits"的项，代表未归一化的预测分数
                # Pass the input data forward through the model and extract the logits from the output dictionary. 
                # Here we assume that the output of the model is a dictionary containing an item with the key "logits" representing the unnormalized prediction score
                outputs = model(inputs)["logits"]
            # 使用torch.max(outputs, dim=1)[1]找到每个样本预测分数最高的类别的索引，即模型的预测结果。dim=1表示在类别维度上寻找最大值
            # Use torch.max(outputs, dim=1)[1] to find the index of the category with the highest prediction score for each sample, 
            # that is, the prediction result of the model. dim=1 means to find the maximum value in the category dimension
            predicts = torch.max(outputs, dim=1)[1]
            # 将模型的预测结果（predicts）与真实标签（targets）进行比较，并计算正确预测的数量。注意这里使用.cpu()将预测结果转移到CPU上，以确保与targets（可能位于CPU上）进行比较时没有设备不匹配的问题。
            # Compare the model's predictions (predicts) with the true labels (targets) and count the number of correct predictions. 
            # Note that .cpu() is used here to transfer the predictions to the CPU to ensure that there is no device mismatch when comparing with the targets (which may be on the CPU).
            correct += (predicts.cpu() == targets).sum()
            total += len(targets)

        return np.around(tensor2numpy(correct) * 100 / total, decimals=2)

    def _eval_cnn(self, loader):
        self._network.eval()
        y_pred, y_true = [], []
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = self._network(inputs)["logits"]
            predicts = torch.topk(
                outputs, k=self.topk, dim=1, largest=True, sorted=True
            )[
                1
            ]  # [bs, topk]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())

        return np.concatenate(y_pred), np.concatenate(y_true)  # [N, topk]

    def _eval_nme(self, loader, class_means):
        self._network.eval()
        vectors, y_true = self._extract_vectors(loader)
        vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T

        dists = cdist(class_means, vectors, "sqeuclidean")  # [nb_classes, N]
        scores = dists.T  # [N, nb_classes], choose the one with the smallest distance

        return np.argsort(scores, axis=1)[:, : self.topk], y_true  # [N, topk]

    def _extract_vectors(self, loader):
        self._network.eval()
        vectors, targets = [], []
        for _, _inputs, _targets in loader:
            _targets = _targets.numpy()
            if isinstance(self._network, nn.DataParallel):
                _vectors = tensor2numpy(
                    self._network.module.extract_vector(_inputs.to(self._device))
                )
            else:
                _vectors = tensor2numpy(
                    self._network.extract_vector(_inputs.to(self._device))
                )

            vectors.append(_vectors)
            targets.append(_targets)

        return np.concatenate(vectors), np.concatenate(targets)


    def _reduce_exemplar(self, data_manager, m):
        logging.info("Reducing exemplars...({} per classes)".format(m))

        # 使用copy.deepcopy对_data_memory和_targets_memory进行深拷贝
        # Use copy.deepcopy to make a deep copy of _data_memory and _targets_memory
        # 将它们分别复制到dummy_data和dummy_targets中。这是为了确保在处理过程中不会修改原始数据。
        # Copy them into dummy_data and dummy_targets respectively. This is to ensure that the original data is not modified during processing.
        dummy_data, dummy_targets = copy.deepcopy(self._data_memory), copy.deepcopy(
            self._targets_memory
        )
        # 初始化一个名为_class_means的零数组，其维度是（总类别数, 特征维度）。这个数组将用于存储每个类的平均特征向量。
        # Initialize an array of zeros named _class_means with dimensions (total number of classes, feature dimensions). 
        # This array will be used to store the mean feature vector for each class.
        self._class_means = np.zeros((self._total_classes, self.feature_dim))
        # 清空_data_memory和_targets_memory，为新的提炼数据做准备。
        # Clear _data_memory and _targets_memory to prepare for new refined data.
        self._data_memory, self._targets_memory = np.array([]), np.array([])

        for class_idx in range(self._known_classes):
            # 对于每个已知的类，首先找到该类在dummy_targets中的所有索引。初始阶段mask是一个长度为0的数组，因为_data_memory和_targets_memory为空。
            # For each known class, first find all the indices of that class in dummy_targets. 
            # Initially, mask is an array of length 0 because _data_memory and _targets_memory are empty.
            mask = np.where(dummy_targets == class_idx)[0]
            # 使用这些索引从dummy_data和dummy_targets中提取前m个样本。初始阶段dd和dt是长度为0的数组。
            # Use these indices to extract the first m samples from dummy_data and dummy_targets. 
            # Initially dd and dt are arrays of length 0.
            # dd, dt = dummy_data[mask][:m], dummy_targets[mask][:m]
            if len(mask) < m:
                dd, dt = dummy_data[mask], dummy_targets[mask]
                print(f"Class {class_idx} has less than {m} exemplars, only {len(mask)}")
                # print("the len of dd is:", len(dd))
                # print("the len of dt is:", len(dt))
            else:
                dd, dt = dummy_data[mask][:m], dummy_targets[mask][:m]
            # 将这些样本添加到_data_memory和_targets_memory中。
            # Add these samples to _data_memory and _targets_memory.

            # 这里的 else dd 和 else dt 部分的作用是当 _data_memory 和 _targets_memory 为空时，
            # The else dd and else dt parts here are used when _data_memory and _targets_memory are empty.

            # 直接将 dd 和 dt 赋值给它们。这是因为在 NumPy 中，你不能将一个非空的数组和一个空数组进行连接，所以需要进行这种条件检查。
            # Just assign dd and dt to them. This is because in NumPy, you cannot concatenate a non-empty array with an empty array, so this conditional check is needed.
            self._data_memory = (
                np.concatenate((self._data_memory, dd))
                if len(self._data_memory) != 0
                else dd
            )
            self._targets_memory = (
                np.concatenate((self._targets_memory, dt))
                if len(self._targets_memory) != 0
                else dt
            )

            # Exemplar mean
            idx_dataset = data_manager.get_dataset(
                [], source="train", mode="test", appendent=(dd, dt)
            )
            idx_loader = DataLoader(
                idx_dataset, batch_size=batch_size, shuffle=False, num_workers=4
            )
            vectors, _ = self._extract_vectors(idx_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            mean = np.mean(vectors, axis=0)
            mean = mean / np.linalg.norm(mean)

            self._class_means[class_idx, :] = mean


    # 这是我们提出的过采样范例集构造方法。
    # This is the oversampling exemplar set construction method we proposed.
    def _construct_exemplar(self, data_manager, m):
        logging.info("Constructing exemplars...({} per classes)".format(m))
        for class_idx in range(self._known_classes, self._total_classes):
            data, targets, idx_dataset = data_manager.get_dataset(
                np.arange(class_idx, class_idx + 1),
                source="train",
                mode="test",
                ret_data=True,
            )
            idx_loader = DataLoader(
                
                idx_dataset, batch_size=batch_size, shuffle=False, num_workers=4
            )
            # 提取某一类的特征向量
            # Extract the feature vector of a certain class
            vectors, _ = self._extract_vectors(idx_loader)

            # 对该类特征向量进行L2归一化
            # Perform L2 normalization on this type of feature vector
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            # 计算该类特征向量的均值
            # Calculate the mean of the eigenvectors of this class
            class_mean = np.mean(vectors, axis=0)

            if len(data) < m:
                logging.info(f'the class of {class_idx} has {len(data)} exemplars')
                last_data = data[-1]
                padding_len = m - len(data)

                padding_data = np.full(padding_len,last_data)
                selected_exemplars = np.concatenate((data,padding_data))
                exemplar_targets = np.full(m, class_idx)

                self._data_memory = (
                np.concatenate((self._data_memory, selected_exemplars))
                if len(self._data_memory) != 0
                else selected_exemplars)
                self._targets_memory = (
                np.concatenate((self._targets_memory, exemplar_targets))
                if len(self._targets_memory) != 0
                else exemplar_targets)

                self._class_means[class_idx, :] = class_mean

            else:
                selected_exemplars = []
                exemplar_vectors = []  # [n, feature_dim]
                logging.info(f'the class of {class_idx} has {len(data)} exemplars')
                for k in range(1, m + 1):
                    S = np.sum(
                    exemplar_vectors, axis=0
                    )  # [feature_dim] sum of selected exemplars vectors
                    mu_p = (vectors + S) / k  # [n, feature_dim] sum to all vectors
                    i = np.argmin(np.sqrt(np.sum((class_mean - mu_p) ** 2, axis=1)))
                    selected_exemplars.append(
                    np.array(data[i])
                    )  # New object to avoid passing by inference
                    exemplar_vectors.append(
                    np.array(vectors[i])
                    )  # New object to avoid passing by inference

                    vectors = np.delete(
                        vectors, i, axis=0
                    )  # Remove it to avoid duplicative selection
                    data = np.delete(
                        data, i, axis=0
                    )  # Remove it to avoid duplicative selection
                selected_exemplars = np.array(selected_exemplars)
                """
                这两行代码将选定的exemplars及其对应的目标值添加到类的内存数组中。
                如果内存数组之前为空，则直接将选定的exemplars赋值给内存数组；否则，
                使用np.concatenate将新旧数据连接起来。
                """
                # These two lines of code add the selected exemplars and their corresponding target values ​​to the class's memory array.
                # If the memory array is empty before, the selected exemplars are directly assigned to the memory array; 
                # otherwise,np.concatenate is used to concatenate the new and old data.
                exemplar_targets = np.full(m, class_idx)
                self._data_memory = (
                np.concatenate((self._data_memory, selected_exemplars))
                if len(self._data_memory) != 0
                else selected_exemplars
                )
                self._targets_memory = (
                np.concatenate((self._targets_memory, exemplar_targets))
                if len(self._targets_memory) != 0
                else exemplar_targets
                )

           
                idx_dataset = data_manager.get_dataset([],
                source="train",
                mode="test",
                appendent=(selected_exemplars, exemplar_targets),
                )
                idx_loader = DataLoader(
                idx_dataset, batch_size=batch_size, shuffle=False, num_workers=4)   
                vectors, _ = self._extract_vectors(idx_loader)
                vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
                mean = np.mean(vectors, axis=0)
                mean = mean / np.linalg.norm(mean)

                self._class_means[class_idx, :] = mean
    # 这是未采用我们提出的过采样范例集构造方法。
    # This is without using our proposed oversampling example set construction method.
    def _construct_exemplar(self, data_manager, m):
        logging.info("Constructing exemplars...({} per classes)".format(m))
        for class_idx in range(self._known_classes, self._total_classes):
            data, targets, idx_dataset = data_manager.get_dataset(
                np.arange(class_idx, class_idx + 1),
                source="train",
                mode="test",
                ret_data=True,
            )
            idx_loader = DataLoader(
                idx_dataset, batch_size=batch_size, shuffle=False, num_workers=4
            )
            
            vectors, _ = self._extract_vectors(idx_loader)
          
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
     
            class_mean = np.mean(vectors, axis=0)

            if len(data) < m:
                logging.info(f'the class of {class_idx} has {len(data)} exemplars')


                selected_exemplars = data
                exemplar_targets = targets
                self._data_memory = (
                np.concatenate((self._data_memory, selected_exemplars))
                if len(self._data_memory) != 0
                else selected_exemplars)
                self._targets_memory = (
                np.concatenate((self._targets_memory, exemplar_targets))
                if len(self._targets_memory) != 0
                else exemplar_targets)

                print(f'the class of {class_idx} has less than {m} exemplars, only {len(data)}')
                self._class_means[class_idx, :] = class_mean

            else:
                selected_exemplars = []
                exemplar_vectors = []  # [n, feature_dim]
                logging.info(f'the class of {class_idx} has {len(data)} exemplars')
                for k in range(1, m + 1):
                    S = np.sum(
                    exemplar_vectors, axis=0
                    )  # [feature_dim] sum of selected exemplars vectors
                    mu_p = (vectors + S) / k  # [n, feature_dim] sum to all vectors
                    i = np.argmin(np.sqrt(np.sum((class_mean - mu_p) ** 2, axis=1)))
                    selected_exemplars.append(
                    np.array(data[i])
                    )  # New object to avoid passing by inference
                    exemplar_vectors.append(
                    np.array(vectors[i])
                    )  # New object to avoid passing by inference

                    vectors = np.delete(
                        vectors, i, axis=0
                    )  # Remove it to avoid duplicative selection
                    data = np.delete(
                        data, i, axis=0
                    )  # Remove it to avoid duplicative selection
                selected_exemplars = np.array(selected_exemplars)
                exemplar_targets = np.full(m, class_idx)
                self._data_memory = (
                np.concatenate((self._data_memory, selected_exemplars))
                if len(self._data_memory) != 0
                else selected_exemplars
                )
                self._targets_memory = (
                np.concatenate((self._targets_memory, exemplar_targets))
                if len(self._targets_memory) != 0
                else exemplar_targets
                )

           
                idx_dataset = data_manager.get_dataset([],
                source="train",
                mode="test",
                appendent=(selected_exemplars, exemplar_targets),
                )
                idx_loader = DataLoader(
                idx_dataset, batch_size=batch_size, shuffle=False, num_workers=4)   
                vectors, _ = self._extract_vectors(idx_loader)
                vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
                mean = np.mean(vectors, axis=0)
                mean = mean / np.linalg.norm(mean)

                self._class_means[class_idx, :] = mean

    


    # def _construct_exemplar_unified(self, data_manager, m):
    #     logging.info(
    #         "Constructing exemplars for new classes...({} per classes)".format(m)
    #     )
    #     _class_means = np.zeros((self._total_classes, self.feature_dim))

    #     # Calculate the means of old classes with newly trained network
    #     """
    #     历已知的旧类别（从0到self._known_classes-1），并为每个类别计算特征均值。这是通过以下步骤完成的：
    #     从内存中找到属于当前类别的样本索引。
    #     使用这些索引从内存中提取对应的数据和目标。
    #     创建一个数据集和数据加载器，以便将样本批量传递给网络以提取特征。
    #     提取特征并进行L2归一化。
    #     计算归一化后的特征均值。
    #     将计算得到的均值存储在_class_means数组中
    #     """
    #     for class_idx in range(self._known_classes):
    #         mask = np.where(self._targets_memory == class_idx)[0]
    #         class_data, class_targets = (
    #             self._data_memory[mask],
    #             self._targets_memory[mask],
    #         )

    #         class_dset = data_manager.get_dataset(
    #             [], source="train", mode="test", appendent=(class_data, class_targets)
    #         )
    #         class_loader = DataLoader(
    #             class_dset, batch_size=batch_size, shuffle=False, num_workers=4
    #         )
    #         vectors, _ = self._extract_vectors(class_loader)
    #         vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
    #         mean = np.mean(vectors, axis=0)
    #         mean = mean / np.linalg.norm(mean)

    #         _class_means[class_idx, :] = mean

    #     # Construct exemplars for new classes and calculate the means
    #     """
    #     遍历从self._known_classes到self._total_classes-1的新类别，并为每个类别执行以下操作：
    #     从数据管理器中获取当前类别的数据、目标和数据集。
    #     创建一个数据加载器以批量处理数据。
    #     提取特征并进行L2归一化。
    #     使用贪心策略选择m个代表性样本：在每个迭代中，选择使当前类别均值与已选样本均值的差异最小的样本。
    #     将选择的代表性样本添加到内存中的数据和目标数组中。
    #     为选择的代表性样本创建一个新的数据集和数据加载器，并重新提取特征以计算均值。
    #     将计算得到的均值存储在_class_means数组中。
    #     """
    #     for class_idx in range(self._known_classes, self._total_classes):
    #         data, targets, class_dset = data_manager.get_dataset(
    #             np.arange(class_idx, class_idx + 1),
    #             source="train",
    #             mode="test",
    #             ret_data=True,
    #         )
    #         class_loader = DataLoader(
    #             class_dset, batch_size=batch_size, shuffle=False, num_workers=4
    #         )

    #         vectors, _ = self._extract_vectors(class_loader)
    #         vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
    #         class_mean = np.mean(vectors, axis=0)

    #         if len(data) < m:
    #             selected_exemplars = data
    #             exemplar_targets = targets
    #             self._data_memory = (
    #                 np.concatenate((self._data_memory, selected_exemplars))
    #                 if len(self._data_memory) != 0
    #                 else selected_exemplars
    #             )
    #             self._targets_memory = (
    #                 np.concatenate((self._targets_memory, exemplar_targets))
    #                 if len(self._targets_memory) != 0
    #                 else exemplar_targets
    #             )
    #             _class_means[class_idx, :] = mean
            
    #         else:
    #              # Select
    #             selected_exemplars = []
    #             exemplar_vectors = []
    #             for k in range(1, m + 1):
    #                 S = np.sum(
    #                 exemplar_vectors, axis=0
    #                 )  # [feature_dim] sum of selected exemplars vectors
    #                 mu_p = (vectors + S) / k  # [n, feature_dim] sum to all vectors
    #                 i = np.argmin(np.sqrt(np.sum((class_mean - mu_p) ** 2, axis=1)))

    #                 selected_exemplars.append(
    #                     np.array(data[i])
    #                 )  # New object to avoid passing by inference
    #                 exemplar_vectors.append(
    #                     np.array(vectors[i])
    #                 )  # New object to avoid passing by inference

    #                 vectors = np.delete(
    #                     vectors, i, axis=0
    #                 )  # Remove it to avoid duplicative selection
    #                 data = np.delete(
    #                     data, i, axis=0
    #                 )  # Remove it to avoid duplicative selection

    #             selected_exemplars = np.array(selected_exemplars)
    #             exemplar_targets = np.full(m, class_idx)
    #             self._data_memory = (
    #                 np.concatenate((self._data_memory, selected_exemplars))
    #                 if len(self._data_memory) != 0
    #                 else selected_exemplars
    #             )
    #             self._targets_memory = (
    #                 np.concatenate((self._targets_memory, exemplar_targets))
    #                 if len(self._targets_memory) != 0
    #                 else exemplar_targets
    #             )

    #             # Exemplar mean
    #             exemplar_dset = data_manager.get_dataset(
    #                 [],
    #                 source="train",
    #                 mode="test",
    #                 appendent=(selected_exemplars, exemplar_targets),
    #             )
    #             exemplar_loader = DataLoader(
    #                 exemplar_dset, batch_size=batch_size, shuffle=False, num_workers=4
    #             )
    #             vectors, _ = self._extract_vectors(exemplar_loader)
    #             vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
    #             mean = np.mean(vectors, axis=0)
    #             mean = mean / np.linalg.norm(mean)

    #             _class_means[class_idx, :] = mean


           

    #     self._class_means = _class_means
