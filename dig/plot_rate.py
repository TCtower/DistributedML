import matplotlib
import matplotlib.pyplot as plt
import numpy as np

file_names = [
    "test_full.out",
    "test_fc1.out",
    "test_fc12.out",
    "test_conv2.out",
    "test_conv12.out",
]

label_name = [
    "conv1", "conv2", "conv3", "conv4", "conv5",
    "fc1", "fc2", "fc3"
]

file_name = "../dig/res/test_full.out"
f = open(file_name, "r")

x = []
y_avg = []
idx = 0

lines = f.readlines()
for layer in range(0, 8):
    x = []
    y_avg = []
    idx = 0
    for i in range(len(lines)): 
        line = lines[i]
        if not ("Train" in  line):
            continue

        # print(line[:-1])
        idx += 1
        if i == (len(lines) - 1):
            break
        
        x.append(idx)
        # y_avg.append(eval(lines[i + 1 + layer][:-1]))
        # y_avg.append(eval(lines[i + 1 + 8 + layer * 2].split(" ")[-1][:-1]))
        y_avg.append(eval(lines[i + 1 + 8 + layer * 2 + 1].split(" ")[-1][:-1]))

        # for j in range(1, 25):
        #     mid_line = lines[i + j]
        #     print(mid_line[:-1])
    plt.plot(x, y_avg, label=label_name[layer])
print(x)
print(y_avg)
label_size = 12
font2 = {
    'family': 'Times New Roman',
    'weight': 'normal',
    'size': label_size,
}

plt.xlabel("Iteration", font2)
# plt.ylabel("Activation Rate", font2)
# plt.ylabel("Nonzero Gradient Weight Rate", font2)
plt.ylabel("Nonzero Gradient Bias Rate", font2)
plt.title("AlexNet", font2)
plt.legend()
plt.ylim(0, 1)

# plt.savefig('./out/' + dig_name[i])

plt.show()
f.close()