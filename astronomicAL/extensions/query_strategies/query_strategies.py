from .. import query_strategies


def get_strategy_dict():

    qs_dict_ = query_strategies.__dict__

    qs_dict = {
        k:v for k,v in qs_dict_.items()
        if (("_" not in k) and (k.lower() != k))
    }
    print(qs_dict)
    return qs_dict
