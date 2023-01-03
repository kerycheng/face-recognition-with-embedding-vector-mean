import warnings

import pandas as pd

warnings.filterwarnings('ignore')

import numpy as np
np.set_printoptions(suppress=True)

from basic_settings import calculation

class create_dataframe(calculation):
    def __init__(self):
        super().__init__()

    def run(self):
        ca = calculation()
        self.data = ca.data_path

        ca.get_dir_all_embs()
        self.name = ca.dataset_dir
        self.embeddings = ca.embs_list

        ca.get_vector_mean()
        self.vector_mean = ca.vector_mean_list

        ca.self_emb2emb_list()
        self.ese2e = ca.eu_self_emb2emb
        self.ise2e = ca.ip_self_emb2emb

        ca.other_emb2emb_list()
        self.eoe2e = ca.eu_other_emb2emb
        self.ioe2e = ca.ip_other_emb2emb

        ca.self_vm2emb_list()
        self.esv2e = ca.eu_self_vm2emb
        self.isv2e = ca.ip_self_vm2emb

        ca.other_vm2emb_list()
        self.eov2e = ca.eu_other_vm2emb
        self.iov2e = ca.ip_other_vm2emb

        self.create_dataframe()
        self.read_dataframe()

    def create_dataframe(self):
        dict = {
            'People Name': self.name,
            'Embeddings': self.embeddings,
            'Vector Mean': self.vector_mean,
            'EU Self E2E': self.ese2e,
            'IP Self E2E': self.ise2e,
            'EU Other E2E': self.eoe2e,
            'IP Other E2E': self.ioe2e,
            'EU Self V2E': self.esv2e,
            'IP Self V2E': self.isv2e,
            'EU Other V2E': self.eov2e,
            'IP Other V2E': self.iov2e
        }
        df = pd.DataFrame(dict)
        df.to_json(f'{self.data}/dataset.json')

    def read_dataframe(self):
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', -1)

        df = pd.read_json(f'{self.data}/dataset.json')
        print(df)

if __name__ == '__main__':
    cd = create_dataframe()
    cd.run()

