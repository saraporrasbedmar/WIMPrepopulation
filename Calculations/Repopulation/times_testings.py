import numpy as np
import time
import matplotlib.pyplot as plt


lenght_array = np.logspace(np.log10(5e4), np.log10(5e6), num=3)
times_sort = []
times_argmax = []

for i in lenght_array:

    stuff = np.random.random(int(i))

    init_time = time.process_time()
    stuff = np.sort(stuff)
    times_sort.append((time.process_time() - init_time))


plt.figure()
plt.plot(lenght_array, times_sort, label='sort', color='k')

num_max_values = [1, 10, 100, 1000]
colors = ['b', 'orange', 'green', 'red']

for nmax, max_val in enumerate(num_max_values):
    times_sort = []

    for i in lenght_array:
        stuff = np.random.random(int(i))

        init_time = time.process_time()
        for j in range(max_val):
            bright_Js = np.where(
                np.max(stuff) == stuff)[0][0]
            stuff[bright_Js] = 0.

        times_sort.append((time.process_time() - init_time))

    plt.plot(lenght_array, times_sort, color=colors[nmax], linestyle='--')

for nmax, max_val in enumerate(num_max_values):
    times_sort = []

    for i in lenght_array:
        stuff = np.random.random(int(i))

        init_time = time.process_time()
        for j in range(max_val):
            bright_Js = np.argmax(stuff)
            stuff[bright_Js] = 0.

        times_sort.append((time.process_time() - init_time))

    plt.plot(lenght_array, times_sort, color=colors[nmax], linestyle='-.')

plt.xscale('log')
plt.yscale('log')

plt.axvline(5e5, color='grey')

plt.legend()

plt.show()
