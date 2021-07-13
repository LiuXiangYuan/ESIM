import torch
import numpy as np


@torch.no_grad()
def prediction(model_list, test_loader, device):
    pres_all = []

    for _, batch in enumerate(test_loader):
        pres_fold = 0.0

        premises = batch["premise"].to(device)
        premises_lengths = batch["premise_length"].to(device)
        hypotheses = batch["hypothesis"].to(device)
        hypotheses_lengths = batch["hypothesis_length"].to(device)

        for i, model in enumerate(model_list):
            outputs, _ = model(premises, premises_lengths, hypotheses, hypotheses_lengths)
            outputs = outputs.squeeze(1)
            outputs = outputs.sigmoid().detach().cpu().numpy()[0]

            pres_fold += outputs / len(model_list)

        pres_all.append(pres_fold)

    predict = np.array(pres_all, dtype=np.float)

    return predict
