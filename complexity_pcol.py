
# PCOL --> PEREIRA COMPLEXITY LIBRARY.. ;)

import numpy as np
from sklearn.neighbors import BallTree
import itertools
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.neighbors import KNeighborsClassifier


class PPcol:
    def __init__(self, classes=None, random_state=None):
        import os
        self.classes = classes
        self.n_classes = len(classes)
        self._id = str(os.getpid())
        if random_state is None:
            random_state = np.random.RandomState()
        self.random_state = random_state

    def info(self):
        pcol_info = {}
        pcol_info['classes'] = self.classes
        return pcol_info

    def __str__(self):
        return str(self.info())

    def xy_measures(self, x_data, y_targ, classes=None):
        if classes is None:
            classes = self.classes

        x_data = np.asarray(x_data)
        y_targ = np.asarray(y_targ)

        e_matrix = self._euclidean_table(x_data)
        diict = {'F1': [], 'N2': [], 'N4': []}
        divided_y = self._get_OVA(y_targ)
        for y_ova in divided_y:
            f1 = self.F1_2c(x_data, y_ova)
            n2 = self.N2_2c(x_data, y_ova, e_matrix)
            n4 = self.N4_2c(x_data, y_ova)
            diict['F1'].append(round(f1, 5))
            diict['N2'].append(round(n2, 5))
            diict['N4'].append(round(n4, 5))

        return diict

    def _get_OVA(self, y_targ: np.ndarray):
        if len(self.classes) == 2:
            return (y_targ,)

        y_ova = []
        for cls in self.classes:
            new_y = y_targ==cls
            new_y = new_y.astype(int)
            y_ova.append(new_y)
        return y_ova


    def F1_2c(self, x_data: np.ndarray, y_targ: np.ndarray):
        cls_mask = y_targ == 1
        if np.count_nonzero(cls_mask) == 0 or np.count_nonzero(cls_mask) == len(y_targ):
            return 0.0

        X1 = x_data[cls_mask]
        X0 = x_data[~cls_mask]
        me0 = np.nan_to_num(X0.mean(axis=0))
        va0 = np.nan_to_num(X0.var(axis=0))
        me1 = np.nan_to_num(X1.mean(axis=0))
        va1 = np.nan_to_num(X1.var(axis=0))

        num = (me0 - me1)**2
        # TODO: verificar se isso é mais ou menos.. soma ou diminui as variâncias???
        den = va0 + va1
        zero_mask = den == 0.0
        den[zero_mask] = 1
        res = num/den
        res[zero_mask] = 0.0
        #res[np.isinf(res)] = 0.0
        res = np.asscalar(np.nanmax(res))
        return res

    def _euclidean_table(self, data):
        # get euclidean matrix (table) from all to all
        table = euclidean_distances(data, data)
        for i in range(len(table)):
            # in the main diagonal, we put a big value
            table[i, i] = np.inf
        return table

    def N2_2c(self, x_data, y_targ, distance_table=None):
        if distance_table is None:
            distance_table = self._euclidean_table(x_data)

        intra = inter = 0.0
        for dists, cls in zip(distance_table, y_targ):
            minInter = minIntra = 0.0

            # verifica no buffer quem é igual a instância que que está sendo testada no momento
            cls_mask = y_targ == cls
            if np.count_nonzero(cls_mask):
                minInter = dists[cls_mask].min()
                if np.isinf(minInter):
                    minInter = 0.0

            if np.count_nonzero(~cls_mask):
                minIntra = dists[~cls_mask].min()
                if np.isinf(minIntra):
                    minIntra = 0.0

            inter += minInter
            intra += minIntra
        N2 = 0.0
        if inter != 0 and intra != 0:
            N2 = round(inter/intra, 6)
        return N2


    def _make_syntetic_data(self, x_data, size):
        n_features = x_data.shape[1]
        result = []
        for _ in range(size):
            # TODO: o erro é aqui na linha abaixo...
            # idx1, idx2 = np.random.choice(len(x_data), 2, replace=False)
            # rnd = np.random.rand(n_features)
            idx1, idx2 = self.random_state.choice(len(x_data), 2, replace=False)
            rnd = self.random_state.rand(n_features)
            new_instance = x_data[idx1] * rnd + x_data[idx2] * (1 - rnd)
            result.append(new_instance)
        return result


    def N4_2c(self, x_data: np.ndarray, y_targ: np.ndarray):
        classes = np.unique(y_targ)
        if len(classes) > 2:
            raise ValueError('Esta função funciona para duas classes apenas')
        if len(classes) < 2:
            return 0.0

        neighbors = KNeighborsClassifier(n_neighbors=1, algorithm='auto') # n_jobs=-1??
        neighbors.fit(x_data, y_targ)
        n_samples = len(x_data)
        x_syntetic = []
        y_syntetic = []

        cls_mask = y_targ == classes[1]
        not_mask = ~cls_mask

        if np.count_nonzero(cls_mask) > 1:
            x_syntetic.extend(self._make_syntetic_data(x_data[cls_mask], n_samples))
            y_syntetic.extend((classes[1] for _ in range(n_samples)))

        if np.count_nonzero(not_mask) > 1:
            x_syntetic.extend(self._make_syntetic_data(x_data[not_mask], n_samples))
            y_syntetic.extend((classes[0] for _ in range(n_samples)))

        score = neighbors.score(x_syntetic, y_syntetic)
        wrong = 1-score
        N4 = wrong
        return N4

# def main_test():
#     from pds_dataset import PDataset
#     from complexity_dcol import PDcol
#     from pds_experiment import _rescale
#     measures = ('F1', 'N2', 'N4')
#
#     run_cfg = {'Wine': [15, 17], 'Adult': [6, 7], 'Banana': [10, 15, 18], 'Ecoli': [1, 3, 5, 7, 9, 11, 13]}
#     #run_cfg = {'Ecoli': (15, )}
#     for ds_name, execs in run_cfg.items():
#         print('\n\nDATASET:', ds_name)
#         dataset = PDataset.load_data(ds_name)
#
#         for exec_no in execs:
#             ds_train, ds_dsel, ds_test = dataset.load_split(exec_no)
#             _rescale(ds_train, ds_dsel, ds_test)
#
#             pcol = PPcol(classes=dataset.classes_)
#             p_res = pcol.xy_measures(dataset.x_data, dataset.y_targ)
#             print('Pcol::', p_res)
#
#             dcol = PDcol(classes=dataset.classes_)
#             d_res = dcol.xy_measures(dataset.x_data, dataset.y_targ)
#             print('Dcol::', {k: v for k, v in d_res.items() if k in measures})
#
#             print('--exec_no:', exec_no)
#
#             for iid in range(10):
#                 bag = ds_train.load_bag()
#
#                 p_res = pcol.xy_measures(bag.x_data, bag.y_targ)
#                 print('## P_Bag', iid, '::', p_res)
#
#                 d_res = dcol.xy_measures(bag.x_data, bag.y_targ)
#                 print('## D_Bag', iid, '::', {k: v for k, v in d_res.items() if k in measures})
#
#
# # def to_marcos():
# #     data = [[14.38, 1.87, 2.38, 12.0, 102.0, 3.3, 3.64, 0.29, 2.96, 7.5, 1.2, 3.0, 1547.0],
# #      [12.93, 3.8, 2.65, 18.6, 102.0, 2.41, 2.41, 0.25, 1.98, 4.5, 1.03, 3.52, 770.0],
# #      [12.85, 1.6, 2.52, 17.8, 95.0, 2.48, 2.37, 0.26, 1.46, 3.93, 1.09, 3.63, 1015.0],
# #      [13.64, 3.1, 2.56, 15.2, 116.0, 2.7, 3.03, 0.17, 1.66, 5.1, 0.96, 3.36, 845.0],
# #      [14.12, 1.48, 2.32, 16.8, 95.0, 2.2, 2.43, 0.26, 1.57, 5.0, 1.17, 2.82, 1280.0],
# #      [14.23, 1.71, 2.43, 15.6, 127.0, 2.8, 3.06, 0.28, 2.29, 5.64, 1.04, 3.92, 1065.0],
# #      [14.06, 2.15, 2.61, 17.6, 121.0, 2.6, 2.51, 0.31, 1.25, 5.05, 1.06, 3.58, 1295.0],
# #      [13.74, 1.67, 2.25, 16.4, 118.0, 2.6, 2.9, 0.21, 1.62, 5.85, 0.92, 3.2, 1060.0],
# #      [13.86, 1.35, 2.27, 16.0, 98.0, 2.98, 3.15, 0.22, 1.85, 7.22, 1.01, 3.55, 1045.0],
# #      [13.86, 1.35, 2.27, 16.0, 98.0, 2.98, 3.15, 0.22, 1.85, 7.22, 1.01, 3.55, 1045.0],
# #      [14.23, 1.71, 2.43, 15.6, 127.0, 2.8, 3.06, 0.28, 2.29, 5.64, 1.04, 3.92, 1065.0],
# #      [13.87, 1.9, 2.8, 19.4, 107.0, 2.95, 2.97, 0.37, 1.76, 4.5, 1.25, 3.4, 915.0],
# #      [13.5, 1.81, 2.61, 20.0, 96.0, 2.53, 2.61, 0.28, 1.66, 3.52, 1.12, 3.82, 845.0],
# #      [14.06, 2.15, 2.61, 17.6, 121.0, 2.6, 2.51, 0.31, 1.25, 5.05, 1.06, 3.58, 1295.0],
# #      [12.7, 3.87, 2.4, 23.0, 101.0, 2.83, 2.55, 0.43, 1.95, 2.57, 1.19, 3.13, 463.0],
# #      [12.47, 1.52, 2.2, 19.0, 162.0, 2.5, 2.27, 0.32, 3.28, 2.6, 1.16, 2.63, 937.0],
# #      [12.52, 2.43, 2.17, 21.0, 88.0, 2.55, 2.27, 0.26, 1.22, 2.0, 0.9, 2.78, 325.0],
# #      [11.41, 0.74, 2.5, 21.0, 88.0, 2.48, 2.01, 0.42, 1.44, 3.08, 1.1, 2.31, 434.0],
# #      [11.82, 1.72, 1.88, 19.5, 86.0, 2.5, 1.64, 0.37, 1.42, 2.06, 0.94, 2.44, 415.0],
# #      [12.21, 1.19, 1.75, 16.8, 151.0, 1.85, 1.28, 0.14, 2.5, 2.85, 1.28, 3.07, 718.0],
# #      [12.99, 1.67, 2.6, 30.0, 139.0, 3.3, 2.89, 0.21, 1.96, 3.35, 1.31, 3.5, 985.0],
# #      [12.42, 2.55, 2.27, 22.0, 90.0, 1.68, 1.84, 0.66, 1.42, 2.7, 0.86, 3.3, 315.0],
# #      [12.07, 2.16, 2.17, 21.0, 85.0, 2.6, 2.65, 0.37, 1.35, 2.76, 0.86, 3.28, 378.0],
# #      [11.41, 0.74, 2.5, 21.0, 88.0, 2.48, 2.01, 0.42, 1.44, 3.08, 1.1, 2.31, 434.0],
# #      [12.6, 1.34, 1.9, 18.5, 88.0, 1.45, 1.36, 0.29, 1.35, 2.45, 1.04, 2.77, 562.0],
# #      [11.82, 1.47, 1.99, 20.8, 86.0, 1.98, 1.6, 0.3, 1.53, 1.95, 0.95, 3.33, 495.0],
# #      [11.56, 2.05, 3.23, 28.5, 119.0, 3.18, 5.08, 0.47, 1.87, 6.0, 0.93, 3.69, 465.0],
# #      [12.0, 3.43, 2.0, 19.0, 87.0, 2.0, 1.64, 0.37, 1.87, 1.28, 0.93, 3.05, 564.0],
# #      [12.33, 0.99, 1.95, 14.8, 136.0, 1.9, 1.85, 0.35, 2.76, 3.4, 1.06, 2.31, 750.0],
# #      [12.0, 3.43, 2.0, 19.0, 87.0, 2.0, 1.64, 0.37, 1.87, 1.28, 0.93, 3.05, 564.0],
# #      [12.33, 0.99, 1.95, 14.8, 136.0, 1.9, 1.85, 0.35, 2.76, 3.4, 1.06, 2.31, 750.0],
# #      [13.05, 3.86, 2.32, 22.5, 85.0, 1.65, 1.59, 0.61, 1.62, 4.8, 0.84, 2.01, 515.0],
# #      [14.13, 4.1, 2.74, 24.5, 96.0, 2.05, 0.76, 0.56, 1.35, 9.2, 0.61, 1.6, 560.0],
# #      [13.17, 5.19, 2.32, 22.0, 93.0, 1.74, 0.63, 0.61, 1.55, 7.9, 0.6, 1.48, 725.0],
# #      [12.25, 4.72, 2.54, 21.0, 89.0, 1.38, 0.47, 0.53, 0.8, 3.85, 0.75, 1.27, 720.0],
# #      [13.49, 3.59, 2.19, 19.5, 88.0, 1.62, 0.48, 0.58, 0.88, 5.7, 0.81, 1.82, 580.0],
# #      [13.48, 1.67, 2.64, 22.5, 89.0, 2.6, 1.1, 0.52, 2.29, 11.75, 0.57, 1.78, 620.0],
# #      [12.25, 3.88, 2.2, 18.5, 112.0, 1.38, 0.78, 0.29, 1.14, 8.21, 0.65, 2.0, 855.0],
# #      [13.78, 2.76, 2.3, 22.0, 90.0, 1.35, 0.68, 0.41, 1.03, 9.58, 0.7, 1.68, 615.0],
# #      [13.84, 4.12, 2.38, 19.5, 89.0, 1.8, 0.83, 0.48, 1.56, 9.01, 0.57, 1.64, 480.0],
# #      [13.11, 1.9, 2.75, 25.5, 116.0, 2.2, 1.28, 0.26, 1.56, 7.1, 0.61, 1.33, 425.0],
# #      [13.62, 4.95, 2.35, 20.0, 92.0, 2.0, 0.8, 0.47, 1.02, 4.4, 0.91, 2.05, 550.0],
# #      [13.08, 3.9, 2.36, 21.5, 113.0, 1.41, 1.39, 0.34, 1.14, 9.4, 0.57, 1.33, 550.0],
# #      [13.08, 3.9, 2.36, 21.5, 113.0, 1.41, 1.39, 0.34, 1.14, 9.4, 0.57, 1.33, 550.0]]
# #
# #     target = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
# #               1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
# #
# #     classes = [0, 1, 2]
# #
# #     from sklearn.preprocessing import MinMaxScaler
# #     scaler = MinMaxScaler()
# #     scaler.fit(data)
# #     data2 = scaler.transform(data)
# #     print(data2)
# #
# #     pcol = PPcol(classes=classes)
# #     p_res = pcol.xy_measures(data2, target, classes)
# #     print('Pcol::', p_res)
#
# if __name__ == "__main__":
# #    main_test()
#     #to_marcos()
