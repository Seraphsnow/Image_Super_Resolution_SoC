import csv

listed_A = []
listed_y = []

part1 = []
y_part1 = []
part2 = []
y_part2 = []
part3 = []
y_part3 = []

with open("framingham.csv", "r") as file:
    csvFile = csv.reader(file)
    flag = False
    for lines in csvFile:
        if flag:    
            listed_y.append(lines[-1])
            list1 = [1] + lines[:-1]
            listed_A.append(list1)
        flag = True

part_size_1 = int(0.7*4239)
part_size_2 =  int(0.85*4239)

for i in range(len(listed_A)):
	if i < part_size_1:
		part1.append(listed_A[i])
		y_part1.append(listed_y[i])
	elif i < part_size_2:
		part2.append(listed_A[i])
		y_part2.append(listed_y[i])
	else:
		part3.append(listed_A[i])
		y_part3.append(listed_y[i])


