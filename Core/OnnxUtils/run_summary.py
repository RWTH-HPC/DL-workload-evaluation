import csv
import sys
from BenchmarkDatabase.elements import Time

def summarise_training_run(op_dict):
    print("Training summary:")
    tt = Time.fromString("0ms")
    timing_list = []
    for key in op_dict.keys():
        t = Time.fromString("0ms")
        for (time, shape) in op_dict[key]:
            # print("{} {}{}".format(key, time, shape))
            t = t + Time.fromString(time)
        tt = tt + t
        timing_list.append((key, t))
    timing_list.sort(key=lambda x: x[1], reverse=True)
    for (key, t) in timing_list:
        print("{} total: {}".format(key, t))

    print("All operations summed: {}".format(tt))

if __name__ == '__main__':
    with open(sys.argv[1], newline='') as csvfile:
        sreader = csv.reader(csvfile, delimiter=',')
        op_dict = {}
        for row in sreader:
            print(row)
            key = row[1]
            if key not in op_dict:
                op_dict[key] = []
            op_dict[key].append((row[0], row[2]))




