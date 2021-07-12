from collections import OrderedDict

CONFIGS = {}


CONFIGS['relu_reward'] = OrderedDict(
    relu_improvement_reward=True,
    time_in_state=True,
)
CONFIGS['prior-lengthscale'] = OrderedDict(
    length_scale_prior_lower=0.05,
    length_scale_prior_upper=0.2,
)
