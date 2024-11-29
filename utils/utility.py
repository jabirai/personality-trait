def scale_to_range(data, min_val=1, max_val=10):
    values = list(data.values())
    
    original_min = min(values)
    original_max = max(values)

    scaled_data = {
        key: round(min_val + (value - original_min) * (max_val - min_val) / (original_max - original_min),1)
        for key, value in data.items()
    }
    return scaled_data