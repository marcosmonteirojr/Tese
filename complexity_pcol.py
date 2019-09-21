

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

def main_test():
    from pds_dataset import PDataset
    from complexity_dcol import PDcol
    from pds_experiment import _rescale
    measures = ('F1', 'N2', 'N4')

    run_cfg = {'Wine': [15, 17], 'Adult': [6, 7], 'Banana': [10, 15, 18], 'Ecoli': [1, 3, 5, 7, 9, 11, 13]}
    #run_cfg = {'Ecoli': (15, )}
    for ds_name, execs in run_cfg.items():
        print('\n\nDATASET:', ds_name)
        dataset = PDataset.load_data(ds_name)

        for exec_no in execs:
            ds_train, ds_dsel, ds_test = dataset.load_split(exec_no)
            _rescale(ds_train, ds_dsel, ds_test)

            pcol = PPcol(classes=dataset.classes_)
            p_res = pcol.xy_measures(dataset.x_data, dataset.y_targ)
            print('Pcol::', p_res)

            dcol = PDcol(classes=dataset.classes_)
            d_res = dcol.xy_measures(dataset.x_data, dataset.y_targ)
            print('Dcol::', {k: v for k, v in d_res.items() if k in measures})

            print('--exec_no:', exec_no)

            for iid in range(10):
                bag = ds_train.load_bag()

                p_res = pcol.xy_measures(bag.x_data, bag.y_targ)
                print('## P_Bag', iid, '::', p_res)

                d_res = dcol.xy_measures(bag.x_data, bag.y_targ)
                print('## D_Bag', iid, '::', {k: v for k, v in d_res.items() if k in measures})


if __name__ == "__main__":
    main_test()
