from pandas import DataFrame 

def observations_to_dataframe(observations):
    """ Some formatting to pretty display the observations as dataframe
    """
    data = {['x', 'y', 'Theta', 'v'][i % 4] + ' (' + str(int(i / 4)) + ')': [] for i in range(16)}
    for observation in observations:
        for i, value in enumerate(observation):
            data[list(data.keys())[i]].append(value)

    return DataFrame(data=data)

def actions_to_dataframe(observations):
    """ Some formatting to pretty display the actions as dataframe
    """
    data = {'Acceleration': [], 'Steering angle': []}
    for observation in observations:
        for i, value in enumerate(observation):
            data[list(data.keys())[i]].append(value)

    return DataFrame(data=data)