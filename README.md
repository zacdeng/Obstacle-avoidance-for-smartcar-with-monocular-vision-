# Obstacle-avoidance-for-smartcar-with-monocular-vision

## 单目视觉的智能车避障方案

背景：第十五届“恩智浦”杯全国大学生智能汽车竞赛-创意组别-百度AI项目国家一等奖

问题描述：组委会提供的智能车硬件方面主要使用百度的EdgeBoard，算力较弱且只配备了一个单目摄像头，在比赛中小车运动时需要进行精准的避障

解决方法：主要尝试了两种方法，第一个是在ROS中普遍运用的A*算法，第二个是借鉴了无人机中ESDF的思想而演化的方法

### A*方法

- 效果展示

![astar](https://i.loli.net/2020/08/31/SWVzsgi49Ev7Jf3.gif)

- 主要思想

1. 将摄像头采集到的图片通过二值化转化为黑白图片（这里通过HSV实现）
2. 划定膨胀层
3. A*实现路径规划

- 主要问题

  使用A*算法需要将图片进行栅格化处理，即使通过二值化处理后还是需要对每个栅格的图片图片的颜色进行一个判断（黑或者是白）来决定是否将其列为可移动范围或者障碍物，考虑到EdgeBoard的算力有限，精细的划分会导致其计算时间过长，在小车移动时无法实时给出移动路径，栅格过大又会导致控制不精细...（如果有朋友有相关改进方法或者想法的欢迎留言或Email交流~）

### 动态ROI-ESDF方法

- 效果展示

![avoid](https://raw.githubusercontent.com/zacdeng/Obstacle-avoidance-for-smartcar-with-monocular-vision-/master/avoid.gif)

- 主要思想

1. 将摄像头采集到的图片通过二值化转化为黑白图片（这里通过HSV实现）
2. 结合ESDF的思想，将其转化为每个点到其附近障碍物的距离的灰度图表示
3. 找到距障碍物距离最小的点作为target

- 效果说明

  这是本次参赛过程中所采用的方法，首先其代码量非常少...（核心代码只有两行，均可以通过OpenCV实现）再就是其运行速度快，且精度高，在实际使用中效果非常好
