import os
import pandas as pd
import matplotlib.pyplot as plt


def plot_determination_coefficients(file_path):
    df = pd.read_csv(file_path)

    plt.figure()
    df.plot()
    plt.title('Determination Coefficients')
    plt.xlabel('Delay')
    plt.ylabel('Determination Coefficient')

    plot_path = os.path.join(os.path.dirname(file_path), 'determination_coefficients_plot.png')
    plt.savefig(plot_path)
    plt.close('all')


def visit_directories_and_plot(root_dir):
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename == 'determination_coefficients.csv':
                file_path = os.path.join(dirpath, filename)
                plot_determination_coefficients(file_path)


if __name__ == '__main__':
    root_directory = 'results'
    visit_directories_and_plot(root_directory)
