import numpy as np
from matplotlib import pyplot as plt


def line_chart_acc(rms_acc, spe_acc, rms_axis):
    fig, ax = plt.subplots()
    ax.set_title("test_accuracy_comparison", fontsize=20)
    ax.set_xlabel("sample rate, window length", fontsize=14)
    ax.set_ylabel("accuracy", fontsize=14)  # 纵坐标标签
    ax.plot(rms_axis, rms_acc, marker='*', label='RMS')
    ax.plot(rms_axis, spe_acc, marker='o', label='Spectograms')

    plt.legend()  # 让图例生效
    # 添加网格线
    plt.grid(True, alpha=0.5, axis='both', linestyle=':')
    plt.show()


def line_chart_avr_time(rms_avr_time, spe_avr_time, rms_axis):
    fig, ax = plt.subplots()
    ax.set_title("Average time of model evaluation time", fontsize=20)
    ax.set_xlabel("sample rate, window length", fontsize=14)
    ax.set_ylabel("average time", fontsize=14)
    ax.plot(rms_axis, rms_avr_time, marker='*', label='RMS')
    ax.plot(rms_axis, spe_avr_time, marker='o', label='Spectograms')

    plt.legend()

    plt.grid(True, alpha=0.5, axis='both', linestyle=':')
    plt.show()

def line_chart_pred_time(rms_pred_time, spe_pred_time, rms_axis):
    fig, ax = plt.subplots()
    ax.set_title("Prediction time of model", fontsize=20)
    ax.set_xlabel("sample rate, window length", fontsize=14)
    ax.set_ylabel("prediction time", fontsize=14)
    ax.plot(rms_axis, rms_pred_time, marker='*', label='RMS')
    ax.plot(rms_axis, spe_pred_time, marker='o', label='Spectograms')

    plt.legend()

    plt.grid(True, alpha=0.5, axis='both', linestyle=':')
    plt.show()


def line_chart_all_time(rms_all_time, spe_all_time, rms_axis):
    fig, ax = plt.subplots()
    ax.set_title("Training time of model", fontsize=20)
    ax.set_xlabel("sample rate, window length", fontsize=14)
    ax.set_ylabel("training time", fontsize=14)
    ax.plot(rms_axis, rms_all_time, marker='*', label='RMS')
    ax.plot(rms_axis, spe_all_time, marker='o', label='Spectograms')

    plt.legend()

    plt.grid(True, alpha=0.5, axis='both', linestyle=':')
    plt.show()


def main():
    rms_acc = np.loadtxt('./RMS_test_acc.txt')
    rms_axis = np.loadtxt('./RMS_x_axis.txt', dtype=str, delimiter=" ")
    spe_acc = np.loadtxt('./SPE_test_acc.txt')
    rms_avr_time = np.loadtxt('./RMS_avr_evaluate_time.txt')
    rms_pred_time = np.loadtxt('./RMS_pred_time.txt')
    spe_avr_time = np.loadtxt('./SPE_avr_evaluate_time.txt')
    spe_pred_time = np.loadtxt('./SPE_pred_time.txt')
    rms_all_time = np.loadtxt('./RMS_all_time.txt')
    spe_all_time = np.loadtxt('./SPE_all_time.txt')
    #line_chart_acc(rms_acc, spe_acc, rms_axis)
    #line_chart_avr_time(rms_avr_time, spe_avr_time, rms_axis)
    #line_chart_pred_time(rms_pred_time, spe_pred_time, rms_axis)
    line_chart_all_time(rms_all_time, spe_all_time, rms_axis)
if __name__ == "__main__":
  main()
