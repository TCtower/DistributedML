import matplotlib
import matplotlib.pyplot as plt
import numpy as np

file_names = [
    "test_IID_mean_w5_r1_1.out",
    "test_IID_mean_w5_r001_1.out",
    "test_IID_mean_w5_r0001_1.out",
    "test_IID_mean_w5_r00001_1.out",
    "test_iid_param_w5_r1_2.out",
    "test_iid_param_w5_r001_2.out",
    "test_iid_param_w5_r0001_2.out",
    "test_iid_param_w5_r00001_2.out",

    # "test_param_w5_r1_1.out",
    # "test_param_w5_r001_1.out",
    # "test_param_w5_r0001_1.out",
]

for name in file_names:
    file_name = "../dig/res/" + name
    f = open(file_name, "r")

    x = []
    y_avg = []
    idx = 0
    cnt = 0

    lines = f.readlines()
    for i in range(len(lines)): 
        line = lines[i]
        if not ("accu" in line):
            continue
        
        # if not ("accu" in  line):
        #     continue


        # print(line[:-1])
        idx += 1
        if i == (len(lines) - 1):
            break

        print(lines[i].split(" ")[-1][:-1])
        if lines[i].split(" ")[-1][:-1] != "nan":
            cnt += eval(lines[i].split(" ")[-1][:-1])


        if idx % 1 == 0:
            x.append(idx * 25)
            y_avg.append(cnt / 1.0)
            cnt = 0

    if "iid" not in name:
        plt.plot(x, y_avg, label=name, linestyle="dashed")
    elif "layer" in name:
        plt.plot(x, y_avg, label=name, linestyle='dotted')
    else:
        plt.plot(x, y_avg, label=name)

print(x)
print(y_avg)
label_size = 12
font2 = {
    'family': 'Times New Roman',
    'weight': 'normal',
    'size': label_size,
}

# different
plt.xlabel("Iteration", font2)
# plt.ylabel("Activation Rate", font2)
# plt.ylabel("Nonzero Gradient Weight Rate", font2)
plt.ylabel("Accuracy", font2)
plt.title("AlexNet", font2)
plt.legend()
# plt.ylim(0, 2)

# plt.savefig('./out/' + dig_name[i])

plt.show()
f.close()