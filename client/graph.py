

token_data_dynamic = []
with open('results/token_data_dynamic.csv', 'r') as f:
    lns = f.readlines()
    for l in lns:
        token_len, tme = l.strip().split(',')
        token_len = int(token_len)
        tme = float(tme)
        token_data_dynamic.append((token_len, tme))

token_data_dynamic.sort(key=lambda x:x[0])
dynamic_x = [i[0] for i in token_data_dynamic]
dynamic_y = [i[1] for i in token_data_dynamic]


token_data_nobatch = []
with open('results/token_data_nobatch.csv', 'r') as f:
    lns = f.readlines()
    for l in lns:
        token_len, tme = l.strip().split(',')
        token_len = int(token_len)
        tme = float(tme)
        token_data_nobatch.append((token_len, tme))

token_data_nobatch.sort(key=lambda x:x[0])

nobatch_x = [i[0] for i in token_data_nobatch]
nobatch_y = [i[1] for i in token_data_nobatch]

token_data_static = []
with open('results/token_data_static.csv', 'r') as f:
    lns = f.readlines()
    for l in lns:
        token_len, tme = l.strip().split(',')
        token_len = int(token_len)
        tme = float(tme)
        token_data_static.append((token_len, tme))

token_data_static.sort(key=lambda x:x[0])

static_x = [i[0] for i in token_data_static]
static_y = [i[1] for i in token_data_static]
import matplotlib.pyplot as plt

height = 0.8
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

ax.set_xlabel('# tokens requested')
ax.set_ylabel('request latency (s)')

# g = 'static'
# if g == 'dynamic':
#     ax.plot(dynamic_x, dynamic_y, label='Dynamic', color='r')
#     plt.title("Dynamic Batching")
#     fig.savefig("dynamic.pdf")
# elif g == 'static':
#     ax.plot(static_x, static_y, color='g')
#     plt.title("Static Batching")
#     fig.savefig("static.pdf")
# else:
#     ax.plot(nobatch_x, nobatch_y, label='No batch', color='b')
#     plt.title("No Batching")
#     fig.savefig("nobatch.pdf")

ax.plot(dynamic_x, dynamic_y, label='Dynamic', color='r')
ax.plot(static_x, static_y, label='Static', color='g')
ax.plot(nobatch_x, nobatch_y, label='No batch', color='b')
plt.legend(loc='best', fontsize=8)
plt.title("Latency vs Requested Tokens")
fig.savefig("all.pdf")