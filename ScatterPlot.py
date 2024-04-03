import matplotlib.pyplot as plt

def read_data_map():
    data_map = {}

    with open('dataMap.txt', 'r') as file:
        for line in file:
            parts = line.split(',')
            target = int(parts[2])

            data_point = [float(parts[0][1:]), float(parts[1][:-1])]  # Remove the leading '[' and trailing ']'

            if target not in data_map:
                data_map[target] = []
            data_map[target].append(data_point)

    return data_map

def plot_data_map(data_map):
    colors = {1: 'red' , 2: 'green', 3: 'blue', 4: 'yellow'}
    
    for target, data_points in data_map.items():
        x_data = [point[0] for point in data_points]
        y_data = [point[1] for point in data_points]

        plt.scatter(x_data, y_data, color = colors[target] , marker = '+' , s = 50 ,  label=f'Target {target}')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Scatter Plot')
    plt.legend()
    plt.show()

data_map = read_data_map()
plot_data_map(data_map)