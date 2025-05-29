from tensorflow.keras.metrics import TopKCategoricalAccuracy
from tensorflow.keras import models
from scipy.interpolate import interp1d
from pathlib import Path
import numpy as np
import pandas as pd
import tqdm
import yaml

MODEL_PATH = 'my_model.h5'  # path to the saved .h5 file of the model.


def process_gr_input_single(r, g):
    """Reprocess the PDF into PDF input for the CNN model.
    The feature input to the CNN is a 209 x 1 array, with r-range from 1.5 A to 30. A

    Parameters
    ----------
    r: 1-D numpy array, shape (num_peaks,)
        Distances in the raw PDF data. Assuming r is sorted from smallest to largest.
    g : 1-D numpy array, shape (num_peaks,)
        Peak intensity in the raw PDF data.
    Return:
    ----------
    input_PDF: 2-D numpy array, shape (209, 1)
        reprocessed PDF data to input to the CNN model.
    """
    # cut off the PDF at 30 A
    cut_off_indices = np.where((r <= 30) & (r >= 1.5))
    r_cut = r[cut_off_indices]
    g_cut = g[cut_off_indices]

    # interpolating peak intensity in the input cnn r range
    # any peak in the cnn r-range that falls outside of the input range
    # i.e. if r_cut[0] > 1.5 or r_cut[-1] < 30. will be set to 0
    pdf_interp = interp1d(r_cut, g_cut, kind='quadratic', bounds_error=False, fill_value=0.)
    CNN_R_RANGE = np.linspace(1.5, 30., 209)
    input_pdf = pdf_interp(CNN_R_RANGE)

    # NORMALIZE
    min_g = np.amin(input_pdf)
    max_g = np.amax(input_pdf)
    norm_input = 1. / (max_g - min_g) * (input_pdf - min_g)
    return norm_input


def process_gr_input(r, pdf_rt):
    processed_rt = {}
    for i in range(pdf_rt.shape[0]):
        g = pdf_rt.iloc[i, 1:].astype(float).to_numpy()
        g = process_gr_input_single(r, g)
        processed_rt[pdf_rt.iloc[i, 0]] = g
    return pd.DataFrame(processed_rt)


def predict_sg_simple(pdf_rt):
    model = models.load_model(MODEL_PATH)
    preds = model.predict(pdf_rt)
    return preds


def predict_sg(r, g):
    """Predict the space group of a given PDF using the pretrained CNN model.
    Assuming only 1 input .gr file.

    Parameters
    ----------
    r : list
        list of interatomic distances. r should have dimension (209,)
    g : list
        list of PDF peak intensity. g should have dimension (209,)
    out_csv : str
        path to save the output results into a .csv file. Default None (no output file saved)
    Return
    ----------
    results_df : pandas.DataFrame
        CNN model output, with the top 6 space groups of the given PDF
        and their corresponding probability
    """
    # custom metrics to compile the model
    acc_metric = TopKCategoricalAccuracy(k=6)
    # load the model
    model = models.load_model(MODEL_PATH, custom_objects={'top6_acc': acc_metric})
    # process input PDF data
    g_input = process_gr_input(r, g).reshape((1, 209, 1))
    # make prediction
    preds = model.predict(g_input, verbose=1)[0]
    # writing the results into a pd.DataFrame
    output_dict = {}
    output_dict['space_group'] = np.array(['P-1', 'P2_1', 'Cc', 'P2_1/m', 'C2/m', 'P2/c', 'P2_1/c', 'C2/c',
                                           'P2_12_12_1', 'Pna2_1', 'Cmc2_1', 'Pmmm', 'Pbam', 'Pnnm', 'Pbcn',
                                           'Pbca', 'Pnma', 'Cmcm', 'Cmce', 'Cmmm', 'Immm', 'I4/m', 'I4_1/a',
                                           'I-42d', 'P4/mmm', 'P4/nmm', 'P4_2/mnm', 'I4/mmm', 'I4/mcm',
                                           'I4_1/amd', 'R-3', 'R3m', 'P-3m1', 'R-3m', 'R-3c', 'P6_3/m',
                                           'P6_3mc', 'P6/mmm', 'P6_3/mmc', 'Pa-3', 'F-43m', 'Pm-3m', 'Fm-3m',
                                           'Fd-3m', 'Ia-3d'])
    output_dict['space_group_number'] = np.array(['2', '4', '9', '11', '12', '13', '14', '15', '19', '33', '36',
                                                  '47', '55', '58', '60', '61', '62', '63', '64', '65', '71', '87', '88', '122', '123', '129',
                                                  '136', '139', '140', '141', '148', '160', '164', '166', '167', '176', '186', '191', '194',
                                                  '205', '216', '221', '225', '227', '230'])
    output_dict['probs'] = preds
    results_df = pd.DataFrame(output_dict)
    results_df_sorted = results_df.sort_values(by=['space_group_number'], ascending=False)
    results_df_sorted = results_df_sorted.reset_index(drop=True)
    return results_df_sorted


def eval_rmax():
    for idx in tqdm.tqdm(range(len(configs['rmax_test']))):
        pdf_dir = str(PATH_PDF) + '/pdf_rmax_' + str(idx+1) + '.csv'
        pdf_rt = pd.read_csv(pdf_dir, dtype=str, header=None)
        r_dir = str(Path.cwd()) + '/r/' + str(idx+1) + '.csv'
        r = pd.read_csv(r_dir, header=None).to_numpy().flatten()
        pred_dir = str(PATH_PRED) + '/pred_rmax_' + str(idx+1) + '.csv'
        pdf_rt = process_gr_input(r, pdf_rt)
        pdf_rt = pdf_rt.to_numpy().T
        preds = predict_sg_simple(pdf_rt)
        pd.DataFrame(preds).to_csv(pred_dir, header=False, index=False)


def eval_qmax():
    for idx in tqdm.tqdm(range(len(configs['qmax_test']))):
        pdf_dir = str(PATH_PDF) + '/pdf_qmax_' + str(idx+1) + '.csv'
        pdf_rt = pd.read_csv(pdf_dir, dtype=str, header=None)
        r_dir = str(Path.cwd()) + '/r/original.csv'
        r = pd.read_csv(r_dir, header=None).to_numpy().flatten()
        pred_dir = str(PATH_PRED) + '/pred_qmax_' + str(idx+1) + '.csv'
        pdf_rt = process_gr_input(r, pdf_rt)
        pdf_rt = pdf_rt.to_numpy().T
        preds = predict_sg_simple(pdf_rt)
        pd.DataFrame(preds).to_csv(pred_dir, header=False, index=False)


def eval_adp():
    for idx in tqdm.tqdm(range(len(configs['adp_test']))):
        pdf_dir = str(PATH_PDF) + '/pdf_adp_' + str(idx+1) + '.csv'
        pdf_rt = pd.read_csv(pdf_dir, dtype=str, header=None)
        r_dir = str(Path.cwd()) + '/r/original.csv'
        r = pd.read_csv(r_dir, header=None).to_numpy().flatten()
        pred_dir = str(PATH_PRED) + '/pred_adp_' + str(idx+1) + '.csv'
        pdf_rt = process_gr_input(r, pdf_rt)
        pdf_rt = pdf_rt.to_numpy().T
        preds = predict_sg_simple(pdf_rt)
        pd.DataFrame(preds).to_csv(pred_dir, header=False, index=False)


def eval_qdamp():
    for idx in tqdm.tqdm(range(len(configs['qdamp_test']))):
        pdf_dir = str(PATH_PDF) + '/pdf_qdamp_' + str(idx+1) + '.csv'
        pdf_rt = pd.read_csv(pdf_dir, dtype=str, header=None)
        r_dir = str(Path.cwd()) + '/r/original.csv'
        r = pd.read_csv(r_dir, header=None).to_numpy().flatten()
        pred_dir = str(PATH_PRED) + '/pred_qdamp_' + str(idx+1) + '.csv'
        pdf_rt = process_gr_input(r, pdf_rt)
        pdf_rt = pdf_rt.to_numpy().T
        preds = predict_sg_simple(pdf_rt)
        pd.DataFrame(preds).to_csv(pred_dir, header=False, index=False)


if __name__ == "__main__":
    PATH_CIF = Path(Path.cwd() / 'db_cif_clean/')
    PATH_PDF = Path(Path.cwd() / 'db_rt/')
    PATH_PRED = Path(Path.cwd() / 'pred_rt/')
    with open("configs.yaml", 'r') as stream:
        configs = yaml.safe_load(stream)
    eval_rmax()
    eval_qmax()
    eval_adp()
    eval_qdamp()
