"""Minimal cache state for L2P step scheduling."""


def cache_init(timesteps, interval=7, max_order=2, first_enhance=3):
    cache_dic = {
        "fresh_threshold": interval,
        "max_order": max_order,
        "first_enhance": first_enhance,
        "cache_type": "random",
    }
    current = {
        "final_time": timesteps[-2],
        "activated_steps": [0],
    }
    return cache_dic, current
