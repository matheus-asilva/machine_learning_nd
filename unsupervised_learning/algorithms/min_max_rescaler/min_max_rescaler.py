def featureScaling(arr):
    if min(arr) == max(arr):
        return None
    return [float(x - min(arr))/(max(arr) - min(arr)) for x in arr]

# tests of your feature scaler--line below is input data
data = [115, 140, 175]
print featureScaling(data)