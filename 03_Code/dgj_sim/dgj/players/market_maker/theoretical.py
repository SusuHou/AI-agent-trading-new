"""Theoretical market maker, equation (3.4), and the linear benchmark rule.

    p = xi/(xi^2+theta) y + xi^2/(xi^2+theta) v_bar + theta/(xi^2+theta) E[v|y]
    benchmark: p = v_bar + lambda y
"""


def optimal_price(total_order_flow, value_mean, expected_value_given_flow, investor_slope, pricing_error_weight):
    denominator = investor_slope * investor_slope + pricing_error_weight
    return (
        investor_slope / denominator * total_order_flow
        + investor_slope * investor_slope / denominator * value_mean
        + pricing_error_weight / denominator * expected_value_given_flow
    )


def linear_price(total_order_flow, value_mean, price_impact):
    return value_mean + price_impact * total_order_flow
