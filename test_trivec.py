from tqdm import trange
import time
import os
import numpy as np
from datetime import datetime
from constants import PARAMS, DATA_CONST
from run_utils import create_parser
from src.negative_sampling import *
from src.knowledge_graph import KnowledgeGraph
from src.trivec_model import TriVec
from src.losses import NegativeSoftPlusLoss
from src.utils import switch_grad_mode, switch_model_mode
from run_utils import evaluation, evaluation_ranking


def make_dirs(path: str):
    if not os.path.exists(path):
        os.makedirs(path)


def test_trivec_model(type):
    test_pos_loader = data.DataLoader(
        torch.Tensor(np.array(kg.get_data_by_type('test_' + type))),
        batch_size=parameters['batch_size'],
        shuffle=False)
    test_neg_sampler = HonestNegativeSampler(
        kg.get_num_of_ent("test"), kg.get_num_of_ent('test'),
        kg.df_drug, neg_per_pos=1)
    test_neg_data = NegDataset(
        torch.Tensor(np.array(kg.get_data_by_type('test_' + type))),
        test_neg_sampler)
    test_neg_loader = data.DataLoader(test_neg_data,
                                      batch_size=parameters['batch_size'],
                                      shuffle=False)

    # Loss and classic metrics
    test_loss, test_metrics = evaluation(
        test_pos_loader, test_neg_loader, model, device, loss_func,
        metrics_separately=args.metrics_separately)
    for metric in test_metrics.keys():
        print(f'Metric {metric} is {test_metrics[metric]} for {type} test data')

if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    parameters = PARAMS.copy()
    parameters['batch_size'] = args.batch_size
    parameters['embed_dim'] = args.embed_dim,
    parameters['epoch'] = args.epoch
    parameters['learning_rate'] = args.learning_rate
    parameters['regularization'] = args.regularization
    parameters['use_proteins'] = args.use_proteins
    parameters['reversed'] = args.reversed
    parameters['metrics_separately'] = args.metrics_separately
    parameters['random_val_neg_sampler'] = args.random_val_neg_sampler
    parameters['val_regenerate'] = args.val_regenerate

    use_cuda = args.gpu and torch.cuda.is_available()
    device = torch.device("cuda" if args.gpu else "cpu")
    print(f'Use device: {device}')

    kg = KnowledgeGraph(data_path=DATA_CONST['work_dir'],
                        use_proteins=args.use_proteins,
                        use_proteins_on_validation=False,
                        use_reversed_edges=args.reversed)
    model = TriVec(ent_total=kg.get_num_of_ent('train'),
                   rel_total=kg.get_num_of_rel('train'))
    model = model.to(device)
    loss_func = NegativeSoftPlusLoss()
    print('Test')
    checkpoint = torch.load(args.model_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    switch_grad_mode(model, requires_grad=False)
    switch_model_mode(model, train=False)
    model.eval()

    test_trivec_model('seen')
    print("===========================")
    test_trivec_model('new')
