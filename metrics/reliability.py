import numpy as np


def irr(preds, confs0=None, accs0=None, cohers0=None):
    res_min = 0
    res_max = 0
    if confs0 is None:
        confs0 = np.ones(preds.shape)
    if cohers0 is None:
        cohers0 = np.zeros(preds.shape[1])
    if accs0 is None:
        accs0 = np.ones(preds.shape[1])
    for row in range(len(preds)):
        count_min = 0
        count_max = 0
        r = preds[row]
        # Drop NaN values for faster loop
        idx = ~np.isnan(r)
        r = r[idx]
        accs = accs0[idx]
        confs = confs0[row][idx]
        cohers = cohers0[idx]
        den = 0

        # If more than one annotator, estimate agreement
        if len(r) > 1:
            for i in range(len(r)):
                # Loop without repetition
                for j in range(i + 1, len(r)):
                    den += 1
                    if r[i] == r[j]:
                        count_min_temp = count_sigma(
                            confs[i], confs[j], cohers[i], cohers[j]
                        )
                        count_max_temp = count_sigma(
                            confs[i], confs[j], cohers[i], cohers[j], mode="max",
                        )
                        acc = accs[i] * accs[j]
                        acc /= accs[i] * accs[j] + (1 - accs[i]) * (1 - accs[j])
                        count_min_temp *= acc
                        count_max_temp *= acc
                        count_min += count_min_temp
                        count_max += count_max_temp
                        # count += countn
        else:
            # If only one annotator is present, disagreement is zero but his/her
            # accuracy and confidence should be incorporated anyway
            den = 1
            count_min = float(accs)
            count_max = count_min

        count_min /= den
        count_max /= den
        res_min += count_min
        res_max += count_max
    return res_min / preds.shape[0], res_max / preds.shape[0]


def count_sigma(conf_i, conf_j, cohers_i, cohers_j, mode="min"):
    count = 0
    conf_corr_i = conf_i
    conf_corr_j = conf_j
    if mode == "min":
        conf_corr_i -= cohers_i
        conf_corr_j -= cohers_j
    else:
        conf_corr_i += cohers_i
        conf_corr_j += cohers_j

    if conf_corr_i < 0:
        conf_corr_i = 0
    if conf_corr_j < 0:
        conf_corr_j = 0
    if conf_corr_i > 1:
        conf_corr_i = 1
    if conf_corr_j > 1:
        conf_corr_j = 1

    countn = conf_corr_i * conf_corr_j
    countd = conf_corr_i * conf_corr_j
    countd += conf_corr_i * (1 - conf_corr_j) / 2
    countd += conf_corr_j * (1 - conf_corr_i) / 2
    countd += (1 - conf_corr_i) * (1 - conf_corr_j) / 4
    count += countn / countd
    return count
