from modAL.uncertainty import entropy_sampling, margin_sampling, uncertainty_sampling


def get_strategy_dict():

    qs_dict = {
        "Uncertainty Sampling": uncertainty_sampling,
        "Margin Sampling": margin_sampling,
        "Entropy Sampling": entropy_sampling,
    }
    return qs_dict
