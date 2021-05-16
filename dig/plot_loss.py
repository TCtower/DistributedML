import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import ast

file_names = [
    # "iid_dymparam_r001.out",
    # "iid_dymparam_r0001.out",
    # "iid_dymparam_r00001.out",
    # "test_noniid.out",
    "test_iid.out",

    # "test_param_w5_r1_1.out",
    # "test_param_w5_r001_1.out",
    # "test_param_w5_r0001_1.out",
]


def get_iter_mode(lines):
    iter_mode = 0

    for i in range(len(lines)):
        line = lines[i]

        if (iter_mode == 0) and ("Train" in line):
            # set once
            iter_mode = eval(lines[i].split(" ")[2].split(":")[0])
            break

    return iter_mode + 1


def get_sim(iter_mode, lines):
    x = []
    y = []
    idx = 0
    for i in range(len(lines)):
        line = lines[i]

        if "Cosine" in line:
            # print("cosine")

            tot_sim = 0
            for j in range(i + 1, i + 6):
                if "nan" in lines[j][:-1]:
                    tot_sim = 0
                    break

                sim_list = ast.literal_eval(lines[j][:-1])
                tmp_sim = 0
                for num in sim_list:
                    if num == 1.0:
                        continue

                    tmp_sim += num / 4.0
                # print(sim_list)

                tot_sim += tmp_sim / 5.0

            # print(tot_sim)
            # if tot_sim > 0.99:
            #     tot_sim = 0

            idx += 1
            x.append(idx * iter_mode)
            y.append(tot_sim)

    return x, y

def get_sim_chunk(iter_mode, lines):
    x = []
    y = []
    idx = 0
    for i in range(len(lines)):
        line = lines[i]

        if "chunk" in line:
            # print("cosine")

            tot_sim = 0
            # just one line
            idx += 1
            tmp_sim = []
            for j in range(i + 1, i + 2):

                sim_list = ast.literal_eval(lines[j][:-1])
                # print(sim_list)
                for k in range(len(sim_list)):
                    if idx == 1:
                        y.append([sim_list[k]])
                    else:
                        # print(y)
                        # y[k].append((y[k][-1] * (idx - 1) + sim_list[k]) / idx)
                        y[k].append(sim_list[k])
                # print(tmp_sim)

            # print(tot_sim)
            # if tot_sim > 0.99:
            #     tot_sim = 0

            x.append(idx * iter_mode)
            # y.append(tot_sim)
    return x, y

def get_al(iter_mode, lines, metric):
    x = []
    y = []
    idx = 0

    for i in range(len(lines)):
        line = lines[i]

        if not (metric in line):
            continue

        idx += 1
        if i == (len(lines) - 1):
            break

        # print(lines[i].split(" ")[-1][:-1])
        cnt = 0
        if lines[i].split(" ")[-1][:-1] != "nan":
            cnt = eval(lines[i].split(" ")[-1][:-1])

        x.append(idx * iter_mode)
        y.append(cnt)

    return x, y


def get_pl(iter_mode, lines, name):
    x = []
    y = []
    idx = 0
    accum = 0
    for i in range(len(lines)):
        line = lines[i]

        if "Train" in line:
            idx += 1
            x.append(idx * iter_mode)

            if "dymparam" not in name:
                y.append(1)
            else:
                # print(lines[i + 9][:-1])
                accum += eval(lines[i + 9][:-1].split(" ")[-1])
                y.append(accum)

    return x, y


iter_mode = 0

label_size = 12
font2 = {
    'family': 'Times New Roman',
    'weight': 'normal',
    'size': label_size,
}

fig, ax1 = plt.subplots()
ax1.set_xlabel('Iteration', font2)
ax1.set_ylabel('Accuracy', font2)
plt.ylim(0.0, 100.0)

ax2 = ax1.twinx()
ax2.set_ylabel('Chunk Similarity (Per 10 Iterations)', font2)
# plt.ylim(1, 10)

for name in file_names:
    file_name = "../dig/res/" + name
    f = open(file_name, "r")

    lines = f.readlines()

    iter_mode = get_iter_mode(lines)
    # x_sim, y_sim = get_sim(iter_mode, lines)
    x_accu, y_accu = get_al(iter_mode, lines, "accu")
    # x_pl, y_pl = get_pl(iter_mode, lines, name)
    x_sim_chunk, y_sim_chunk = get_sim_chunk(iter_mode, lines)

    ax1.plot(x_accu, y_accu, label=name[:-4])
    # ax2.plot(x_pl, y_pl, label=name[:-4], linestyle="dashed")
    # ax2.plot(x_sim_chunk, x_sim_chunk, label=name[:-4], linestyle="dashed")
    print(x_sim_chunk, y_sim_chunk)
    for i in range(len(y_sim_chunk)):
        ax2.plot(x_sim_chunk, y_sim_chunk[i], label="chunk_" + str(i), linestyle="dashed")
    # plt.plot(x_sim, y_sim, label=name, linestyle="dashed")

    # if "iid" not in name:
    #     plt.plot(x, y_avg, label=name, linestyle="dashed")
    # elif "layer" in name:
    #     plt.plot(x, y_avg, label=name, linestyle='dotted')
    # else:
    #     plt.plot(x, y_avg, label=name)

# different

plt.title("AlexNet", font2)
fig.tight_layout()
plt.legend(loc='upper left')
# plt.ylim(0, 2)

# plt.savefig('./out/' + dig_name[i])

plt.show()


f.close()