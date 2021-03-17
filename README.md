# pattern-recognition

## perceptron

实现感知器算法，并产生多维正态分布的数据

![动态变换](./perceptron/perceptron.gif)


## linear

### [linear regression](./linear/linear_regression.ipynb)

1. 实现线性回归的函数
   - SSErr(2):通过定义求广义逆，得到分类面法向量`w`
   - GD:梯度下降
   - LMSalg:实现Widrow-Hoff算法（LMS算法），即用随机梯度下降法获得最佳解。 
2. 进行性能测试，计时与勘误
3. 可视化显示，通过不同颜色的色点代表分类与正误，在二维平面绘制分类线

![分类演示](./linear/linear_regression.png)

### [fisher](./linear/fisher.ipynb)

1. 实现fisher算法求最佳投影面
   1. 通过均值衡量类间差距度
   2. 通过方差衡量类内相似度
   3. 利用拉格朗日乘子，得到最佳投影面的公式

