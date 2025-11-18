

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
import pickle

OUTPUT_PATH = "./output/"

def model_to_pickle(model, pickle_path):
    with open(pickle_path, "wb") as f:
        pickle.dump(model, f)

def pickle_to_model(pickle_path):
    with open(pickle_path, "rb") as f:
        return pickle.load(f)

def pickle_to_csv(file_name, csv_path, pickle_path):
    # pickle_to_csv(pickle_name, "./output.csv")
    tmp = {}
    for name in list(file_name):
        with open(pickle_path + name, "rb") as f:
            tmp[name] = pickle.load(f).accuracy
            plt.plot(tmp[name])
    plt.show()
    pd.DataFrame.from_dict(tmp, orient='index').to_csv(str(csv_path))

def pickle_to_dict(pickle_name, pickle_path):
    tmp = {}
    for name in list(pickle_name):
        with open(pickle_path + name, "rb") as f:
            tmp[name] = np.array(pickle.load(f).accuracy)
            plt.plot(tmp[name])
    plt.show()
    return tmp

def pd_one_epoch_to_csv(args, environment, export_data, data_type, \
    epoch, data_per_epoch, data_per_subsystem, PATH):
    for i in range(data_per_epoch):
        export_data_temp = []
        for j,subsystem in enumerate(environment.subsystems_list):
            export_data_temp.append(
                eval('subsystem.' + data_type + '_list')[epoch* data_per_epoch + i]
            )
        export_data_line = np.array(export_data_temp, dtype=object)
        export_data.append(export_data_line.reshape(-1,))
    data = np.array(export_data)
    data = pd.DataFrame(data=data) 
    data.to_csv(PATH,index = True)
    return export_data

if __name__ == "__main__":
    tmp_name = ["mix_mnist", "mix_fmnist", "one_mnist", "one_fmnist"]
    data = pickle_to_dict(tmp_name, OUTPUT_PATH)
    mnist_target = np.linspace(0.2, 0.85, 50)
    fmnist_target = np.linspace(0.2, 0.7, 50)

    one_mesh = np.zeros([len(mnist_target), len(fmnist_target)])
    mix_mesh = np.zeros([len(mnist_target), len(fmnist_target)])

    for i in range(len(mnist_target)):
        for j in range(len(fmnist_target)):
            one_mesh[i, j] = np.where(data["one_mnist"] > mnist_target[i])[0][0] + \
                            np.where(data["one_fmnist"] > fmnist_target[j])[0][0]
            mix_mesh[i, j] = max(np.where(data["mix_mnist"] > mnist_target[i])[0][0],
                                np.where(data["mix_fmnist"] > fmnist_target[j])[0][0])

    fig = plt.figure()
    mnist_target, fmnist_target = np.meshgrid(mnist_target, fmnist_target)

    np.savetxt(OUTPUT_PATH+'mnist.csv', mnist_target, delimiter=',')
    np.savetxt(OUTPUT_PATH+'fmnist.csv', fmnist_target, delimiter=',')
    np.savetxt(OUTPUT_PATH+'one.csv', one_mesh, delimiter=',')
    np.savetxt(OUTPUT_PATH+'mix.csv', mix_mesh, delimiter=',')

    ax1 = Axes3D(fig)
    ax1.plot_surface(mnist_target, fmnist_target, one_mesh, linewidth=1,
                    antialiased=True, cmap=plt.cm.inferno, alpha=1)
    plt.show()

    fig = plt.figure()
    ax2 = Axes3D(fig)
    ax2.plot_surface(mnist_target, fmnist_target, mix_mesh, linewidth=1,
                    antialiased=True, cmap=plt.cm.winter, alpha=1)
    plt.show()
