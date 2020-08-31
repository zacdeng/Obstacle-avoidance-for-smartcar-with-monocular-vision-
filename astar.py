# -*- coding: utf-8 -*-

import cv2
import numpy as np
import os

def sliding_window(image, stepSize, windowSize):
	# slide a window across the image
	for y in range(0, image.shape[0], stepSize):
		for x in range(0, image.shape[1], stepSize):
			# yield the current window
			yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

#A* Search algorithm implementation to find the minimum path between 2 points
def astar(m,startp,endp):
    w,h = 33,25		# 10x10(blocks) is the dimension of the input images
    sx,sy = startp 	#Start Point
    ex,ey = endp 	#End Point
    #[parent node, x, y, g, f]
    node = [None,sx,sy,0,abs(ex-sx)+abs(ey-sy)]
    closeList = [node]
    createdList = {}
    createdList[sy*w+sx] = node
    k=0
    while(closeList):
        node = closeList.pop(0)
        x = node[1]
        y = node[2]
        l = node[3]+1
        k+=1
        #find neighbours
        if k!=0:
            neighbours = ((x,y+1),(x,y-1),(x+1,y),(x-1,y))
        else:
            neighbours = ((x+1,y),(x-1,y),(x,y+1),(x,y-1))
        for nx,ny in neighbours:
            if nx==ex and ny==ey:
                path = [(ex,ey)]
                while node:
                    path.append((node[1],node[2]))
                    node = node[0]
                return list(reversed(path))
            if 0<=nx<w and 0<=ny<h and m[ny][nx]==0:
                if ny*w+nx not in createdList:
                    nn = (node,nx,ny,l,l+abs(nx-ex)+abs(ny-ey))
                    createdList[ny*w+nx] = nn
                    #adding to closelist ,using binary heap
                    nni = len(closeList)
                    closeList.append(nn)
                    while nni:
                        i = (nni-1)>>1
                        if closeList[i][4]>nn[4]:
                            closeList[i],closeList[nni] = nn,closeList[i]
                            nni = i
                        else:
                            break
    return []


def astardetect(image_filename):
	'''
    Returns:
    1 - List of tuples which is the coordinates for occupied grid.
    2 - Dictionary with information of path.
    '''
	occupied_grids = []  # List to store coordinates of occupied grid
	planned_path = {}  # Dictionary to store information regarding path planning

	# load the image and define the window width and height
	image = cv2.imread(image_filename)
	# h, w, ch = image.shape
	# matSrc = np.float32([[110, 0], [210, 0], [0, 220], [320, 220]])
	# matDst = np.float32([[20, 0], [300, 0], [20, 220], [300, 220]])
	# matSpec = cv2.getPerspectiveTransform(matSrc, matDst)
	# dst2 = cv2.warpPerspective(image, matSpec, (w, h))
	dst = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	low_hsv = np.array([0, 0, 46])
	high_hsv = np.array([180, 43, 220])
	thresh1 = cv2.inRange(dst, low_hsv, high_hsv)

	(winW, winH) = (10, 10)  # Size of individual cropped images
	obstacles = []  # List to store obstacles (black tiles)
	index = [1, 1]
	blank_image = np.zeros((10, 10, 3), np.uint8)
	list_images = [[blank_image for i in range(25)] for i in range(33)]  # array of list of images
#	maze = [[0 for i in range(17)] for i in range(13)]  # matrix to represent the grids of individual cropped images
	maze = np.zeros((25,33), np.uint8)
	
	for (x, y, window) in sliding_window(thresh1, stepSize=10, windowSize=(winW, winH)):
	# for (x, y, window) in sliding_window(image, stepSize=20, windowSize=(winW, winH)):
		# if the window does not meet our desired window size, ignore it
		if window.shape[0] != winH or window.shape[1] != winW:
			continue

		#        dst = cv2.cvtColor(clone, cv2.COLOR_BGR2HSV)
		#        low_hsv = np.array([0, 0, 46])
		#        high_hsv = np.array([180, 43, 220])
		#        thresh1 = cv2.inRange(dst, low_hsv, high_hsv)
		# GrayImage = cv2.cvtColor(clone, cv2.COLOR_BGR2GRAY)
		# ret, thresh1 = cv2.threshold(GrayImage, 127, 255, cv2.THRESH_BINARY)
		crop_img = thresh1[y:y + winH, x:x + winW]  # crop the image
		list_images[index[0] - 1][index[1] - 1] = crop_img.copy()  # Add it to the array of images
		# cv2.rectangle(thresh1, (x, y), (x + winW, y + winH), (0, 255, 0), 2)

		average_color_per_row = np.average(crop_img, axis=0)
		average_color = np.average(average_color_per_row, axis=0)
		average_color = np.uint8(average_color)  # Average color of the grids

		if (average_color < 150):  # Check if grids are colored
			maze[index[1] - 1][index[0] - 1] = 1  # 标记为不可到达点
			# 将障碍物标记点周围一圈都打上障碍物标签，类似膨胀层作用
			for i in range(1, 2):
				for j in range(1, 3):
					if 3 < index[1] < 22 and 3 < index[0] < 30:
						maze[index[1] - 1 + i][index[0] - 1 + j] = 1
						maze[index[1] - 1 - i][index[0] - 1 - j] = 1
						maze[index[1] - 1 - i][index[0] - 1 + j] = 1
						maze[index[1] - 1 + i][index[0] - 1 - j] = 1
						maze[index[1] - 1 + i][index[0] - 1] = 1
						maze[index[1] - 1 - i][index[0] - 1] = 1
						maze[index[1] - 1][index[0] - 1 + j] = 1
						maze[index[1] - 1][index[0] - 1 - j] = 1
					elif index[0] < 3:
						maze[index[1] - 1][index[0] - 1 + j] = 1
					elif index[0] > 30:
						maze[index[1] - 1][index[0] - 1 - j] = 1
					else:
						pass
			obstacles.append(tuple(index))  # add to obstacles list
			obstacles_state = True
		elif (average_color > 150) and (maze[index[1] - 1][index[0] - 1] == 1):
			obstacles_state = True
		else:
			obstacles_state = False
			occupied_grids.append(tuple(index))  # These grids are termed as occupied_grids

		print('index:',index,'  x:',x,'y:',y,'  average_color:',average_color, '  obstacles_state:',obstacles_state)
		# cv2.imshow("Window", thresh1)
		# cv2.waitKey(1)

		# Iterate
		index[0] = index[0] + 1
		if (index[0] > 32):
			index[1] = index[1] + 1
			index[0] = 1

	# First part done
	##############################################################################
	startimage = [17, 22]
	list_colored_grids = []
	len_result = 1000
	target_point = 0
	target_point_xy = 0
	path = []
	path_xy = []
	step = 0
	left_count = 0
	right_count = 0
	nopath_flag = 0
	for n in occupied_grids:
		if n not in obstacles:
			list_colored_grids.append(n)  # Grids with objects (not black obstacles)
		else:
			pass

	for x in range(33):
		if x <= 17:
			left_flag = True
			right_flag = False
		else:
			left_flag = False
			right_flag = True
		for y in range(0, 10):
			if maze[y][x] == 0 and left_flag == True:
				left_count = left_count + 1
			elif maze[y][x] == 0 and right_flag == True:
				right_count = right_count + 1
			else:
				pass
	# print(list_colored_grids)
	# print('left:',left_count,'right:',right_count)
	for n in list_colored_grids:
		if n != startimage and 8 <= n[1] <= 9:
			if right_count >= left_count and 17 < n[0] < 33:
				grid = n
				result = astar(maze, (startimage[0] - 1, startimage[1] - 1), (grid[0] - 1, grid[1] - 1))
				list2 = []
			elif right_count < left_count and 0 <= n[0] <= 17:
				grid = n
				result = astar(maze, (startimage[0] - 1, startimage[1] - 1), (grid[0] - 1, grid[1] - 1))
				list2 = []
			else:
				grid = (0, 0)
				result = []
				list2 = []
			for t in result:
				x, y = t[0], t[1]
				list2.append(tuple((x + 1, y + 1)))  # Contains min path + startimage + endimage
				result = list(list2[1:-1])  # Result contains the minimum path required
			planned_path = list([str(grid), result, len(result) + 1])
			# print(planned_path)
			if len(result) + 1 != 1:
				if len(result) + 1 < len_result:
					len_result = len(result) + 1
					x_out = 10 * (grid[0] - 1)
					y_out = 10 * (grid[1] - 1)
					target_point = str(grid)
					target_point_xy = str((x_out, y_out))
					path = result
					path.insert(0, (17, 22))
					path_xy = []
					for target in result:
						x_target_out = 10 * (target[0] - 1)
						y_target_out = 10 * (target[1] - 1)
						path_xy.append((x_target_out, y_target_out))
					step = len_result
		# print('Available Path:',planned_path)
		else:
			pass
	print(
		'--------------------------------------------------------------------------------------------------------------')
	print('Shortest Path:')
	print('  target_point:', target_point)
	print('  target_point_xy:', target_point_xy)
	print('  path', path)
	print('  path_xy', path_xy)
	print('  step:', step)

	'''Show the path and obstacle layer'''
	path_img = image.copy()
	xy_begin = (160, 220)

	for x in range(33):
		for y in range(25):
			x_draw = 10 * x
			y_draw = 10 * y
			if maze[y][x] == 1:
				cv2.line(path_img, (x_draw, y_draw), (x_draw + 10, y_draw), (0, 255, 0), 2)
				cv2.line(path_img, (x_draw + 10, y_draw), (x_draw + 10, y_draw + 10), (0, 255, 0), 2)
				cv2.line(path_img, (x_draw + 10, y_draw + 10), (x_draw, y_draw + 10), (0, 255, 0), 2)
				cv2.line(path_img, (x_draw, y_draw + 10), (x_draw, y_draw), (0, 255, 0), 2)

	for xy in path_xy:
		xy_next = xy
		cv2.line(path_img, xy_begin, xy_next, (0, 0, 255), 2)
		xy_begin = xy

	'''return current position and next goal'''
	current_position = (160, 230)
	if len(path_xy) >= 4:
		next_position = path_xy[3]
	else:
		next_position = (160, 220)
		nopath_flag = 1
		print('------------------Error: No path found------------------')

	cv2.imshow("Path", path_img)
	cv2.waitKey(1000)
	cv2.destroyAllWindows()
	# Uncomment this portion to print the planned_path (Second Part Solution)
	# print("Occupied Grids : ")
	# print(occupied_grids)
	return current_position, next_position, nopath_flag

# change filename to check for other images
image_filename = "test_images"
# astardetect(image_filename)

files= os.listdir(image_filename) # 得到文件夹下的所有文件名称
for file in files: # 遍历文件夹
	if not os.path.isdir(file): # 判断是否是文件夹，不是文件夹才打开
		current_point, next_point, nopath_flag = astardetect(image_filename+"/"+file)
		print('current position:', current_point, '   next position:', next_point)
