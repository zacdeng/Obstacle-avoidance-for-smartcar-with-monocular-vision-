def get_best_point2(ori_bin_img):
    """找到最佳的驾驶点

    Args:
        ori_bin_img (np.array): 输入二值化图

    Returns:
        [point]: 图像上x，y位置
    """
    costmap=cv2.distanceTransform(ori_bin_img,cv2.DIST_L2,3) #产生ESDF图 采用欧式距离 可以换用其他距离测一下
    costmap_roi = costmap[119:210,:] #划定ROI区域，需要实际选取大小 现在表示选取 0-179行的位置 需要自己调整
    min_value,max_value,minloc,maxloc = cv2.minMaxLoc(costmap_roi) #找最大的点
    '''解决障碍物在视线正前方时决策左右摇摆的方法
    动态ROI策略：
        如果两次的目标点分布在图像中线两侧，且距中线距离在一定范围内，认为是目标点左右摇摆的情况
        这种情况直接改变图像ROI范围，将其锁定在一侧
    '''
    return maxloc